# src/visualization/reports/automated_reports.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import json
import logging
from typing import Dict, List, Optional
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import schedule
import time
from jinja2 import Template
import base64
import io

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReportGenerator:
    """Automated report generation for chip performance analytics"""
    
    def __init__(self, output_dir: str = "data/outputs/reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Email configuration (to be set via environment variables)
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.email_user = os.getenv('EMAIL_USER', '')
        self.email_password = os.getenv('EMAIL_PASSWORD', '')
        
        # Report templates
        self.html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{{ report_title }}</title>
            <style>
                body { 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }
                .header {
                    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    text-align: center;
                    margin-bottom: 30px;
                }
                .metric-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }
                .metric-card {
                    background: #f8fafc;
                    border: 1px solid #e2e8f0;
                    border-radius: 8px;
                    padding: 20px;
                    text-align: center;
                }
                .metric-value {
                    font-size: 2em;
                    font-weight: bold;
                    color: #1e3a8a;
                }
                .metric-label {
                    color: #64748b;
                    font-size: 0.9em;
                    margin-top: 5px;
                }
                .section {
                    background: white;
                    border: 1px solid #e2e8f0;
                    border-radius: 8px;
                    padding: 25px;
                    margin-bottom: 25px;
                }
                .section-title {
                    font-size: 1.4em;
                    font-weight: bold;
                    color: #1e3a8a;
                    margin-bottom: 15px;
                    border-bottom: 2px solid #3b82f6;
                    padding-bottom: 5px;
                }
                .alert {
                    padding: 15px;
                    border-radius: 5px;
                    margin: 10px 0;
                }
                .alert-success { background: #f0fdf4; border-left: 4px solid #22c55e; }
                .alert-warning { background: #fffbeb; border-left: 4px solid #f59e0b; }
                .alert-danger { background: #fef2f2; border-left: 4px solid #ef4444; }
                .chart-container {
                    text-align: center;
                    margin: 20px 0;
                }
                .footer {
                    text-align: center;
                    color: #64748b;
                    font-size: 0.9em;
                    margin-top: 40px;
                    padding: 20px;
                    border-top: 1px solid #e2e8f0;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 15px 0;
                }
                th, td {
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #e2e8f0;
                }
                th {
                    background: #f1f5f9;
                    font-weight: 600;
                    color: #1e3a8a;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{{ report_title }}</h1>
                <p>{{ report_subtitle }}</p>
                <p><strong>Report Period:</strong> {{ report_period }}</p>
            </div>
            
            {{ content }}
            
            <div class="footer">
                <p>Generated automatically by Semiconductor Performance Analytics Platform</p>
                <p>Report generated on {{ generation_time }}</p>
            </div>
        </body>
        </html>
        """
    
    def load_data(self, data_path: str = None) -> pd.DataFrame:
        """Load chip performance data"""
        if data_path is None:
            # Try multiple data sources
            possible_paths = [
                'data/raw/chip_test_data/secom_real_data.csv',
                'data/raw/chip_test_data/chip_performance_data.csv',
                'data/streaming/realtime_data.db'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    if path.endswith('.csv'):
                        df = pd.read_csv(path)
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        return df
                    elif path.endswith('.db'):
                        # Load from SQLite database
                        import sqlite3
                        conn = sqlite3.connect(path)
                        df = pd.read_sql_query("SELECT * FROM realtime_chip_data", conn)
                        conn.close()
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        return df
            
            raise FileNotFoundError("No data files found")
        else:
            df = pd.read_csv(data_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
    
    def generate_executive_summary(self, df: pd.DataFrame) -> Dict:
        """Generate executive summary metrics"""
        total_chips = len(df)
        pass_rate = (df['test_result'] == 'PASS').mean() * 100
        avg_performance = df['performance_score'].mean()
        avg_temperature = df['temperature_celsius'].mean()
        avg_efficiency = df['efficiency_score'].mean()
        thermal_issues = (df['thermal_throttling'] == True).sum()
        
        # Quality grades
        performance_scores = df['performance_score']
        excellent = (performance_scores >= performance_scores.quantile(0.9)).sum()
        good = ((performance_scores >= performance_scores.quantile(0.7)) & 
                (performance_scores < performance_scores.quantile(0.9))).sum()
        poor = (performance_scores <= performance_scores.quantile(0.1)).sum()
        
        return {
            'total_units': total_chips,
            'yield_rate': pass_rate,
            'defect_rate': 100 - pass_rate,
            'avg_performance': avg_performance,
            'avg_temperature': avg_temperature,
            'avg_efficiency': avg_efficiency,
            'thermal_incidents': thermal_issues,
            'excellent_units': excellent,
            'good_units': good,
            'poor_units': poor,
            'data_period': f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}"
        }
    
    def create_performance_charts(self, df: pd.DataFrame) -> Dict[str, str]:
        """Create performance visualization charts"""
        charts = {}
        
        # 1. Performance distribution
        fig1 = px.histogram(df, x='performance_score', nbins=30, 
                           title="Performance Score Distribution")
        fig1.update_layout(showlegend=False, height=400)
        charts['performance_dist'] = self._fig_to_base64(fig1)
        
        # 2. Yield by chip type
        yield_by_type = df.groupby(['chip_type', 'test_result']).size().reset_index(name='count')
        fig2 = px.bar(yield_by_type, x='chip_type', y='count', color='test_result',
                     title="Manufacturing Yield by Chip Type",
                     color_discrete_map={'PASS': '#10b981', 'FAIL': '#ef4444'})
        charts['yield_by_type'] = self._fig_to_base64(fig2)
        
        # 3. Daily yield trend
        daily_yield = df.groupby(df['timestamp'].dt.date).agg({
            'test_result': lambda x: (x == 'PASS').mean() * 100
        }).reset_index()
        fig3 = px.line(daily_yield, x='timestamp', y='test_result',
                      title="Daily Manufacturing Yield Trend")
        fig3.update_layout(yaxis_title="Yield Rate (%)")
        charts['daily_yield'] = self._fig_to_base64(fig3)
        
        # 4. Temperature vs Performance
        fig4 = px.scatter(df, x='temperature_celsius', y='performance_score',
                         color='chip_type', title="Temperature vs Performance Analysis")
        charts['temp_vs_perf'] = self._fig_to_base64(fig4)
        
        return charts
    
    def _fig_to_base64(self, fig) -> str:
        """Convert plotly figure to base64 string"""
        img_bytes = fig.to_image(format="png", width=800, height=400)
        img_base64 = base64.b64encode(img_bytes).decode()
        return f"data:image/png;base64,{img_base64}"
    
    def generate_daily_report(self, date: datetime = None) -> str:
        """Generate daily performance report"""
        if date is None:
            date = datetime.now() - timedelta(days=1)
        
        try:
            df = self.load_data()
            
            # Filter for specific date
            daily_data = df[df['timestamp'].dt.date == date.date()]
            
            if daily_data.empty:
                return self._generate_no_data_report(date, "daily")
            
            summary = self.generate_executive_summary(daily_data)
            charts = self.create_performance_charts(daily_data)
            
            # Generate alerts
            alerts = self._generate_alerts(summary)
            
            # Create content
            content = f"""
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{summary['total_units']:,}</div>
                    <div class="metric-label">Units Tested</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary['yield_rate']:.1f}%</div>
                    <div class="metric-label">Manufacturing Yield</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary['avg_performance']:.0f}</div>
                    <div class="metric-label">Avg Performance</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary['avg_temperature']:.1f}°C</div>
                    <div class="metric-label">Avg Temperature</div>
                </div>
            </div>
            
            {alerts}
            
            <div class="section">
                <div class="section-title">📊 Performance Analysis</div>
                <div class="chart-container">
                    <img src="{charts['performance_dist']}" alt="Performance Distribution" style="max-width: 100%;">
                </div>
                <div class="chart-container">
                    <img src="{charts['yield_by_type']}" alt="Yield by Type" style="max-width: 100%;">
                </div>
            </div>
            
            <div class="section">
                <div class="section-title">🎯 Quality Metrics</div>
                <table>
                    <tr><th>Metric</th><th>Value</th><th>Status</th></tr>
                    <tr><td>Manufacturing Yield</td><td>{summary['yield_rate']:.1f}%</td><td>{'✅ Good' if summary['yield_rate'] >= 95 else '⚠️ Monitor' if summary['yield_rate'] >= 90 else '🔴 Critical'}</td></tr>
                    <tr><td>Defect Rate</td><td>{summary['defect_rate']:.1f}%</td><td>{'✅ Good' if summary['defect_rate'] <= 5 else '⚠️ Monitor' if summary['defect_rate'] <= 10 else '🔴 Critical'}</td></tr>
                    <tr><td>Thermal Incidents</td><td>{summary['thermal_incidents']}</td><td>{'✅ Good' if summary['thermal_incidents'] == 0 else '⚠️ Monitor' if summary['thermal_incidents'] <= 5 else '🔴 Critical'}</td></tr>
                    <tr><td>Excellent Units</td><td>{summary['excellent_units']}</td><td>{'✅ Good' if summary['excellent_units'] > summary['total_units'] * 0.1 else '⚠️ Monitor'}</td></tr>
                </table>
            </div>
            """
            
            # Generate HTML report
            template = Template(self.html_template)
            html_content = template.render(
                report_title="Daily Performance Report",
                report_subtitle="Semiconductor Manufacturing Analytics",
                report_period=date.strftime("%B %d, %Y"),
                content=content,
                generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
            # Save report
            filename = f"daily_report_{date.strftime('%Y%m%d')}.html"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Daily report generated: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating daily report: {e}")
            return None
    
    def generate_weekly_report(self, week_start: datetime = None) -> str:
        """Generate weekly performance report"""
        if week_start is None:
            week_start = datetime.now() - timedelta(days=7)
        
        week_end = week_start + timedelta(days=7)
        
        try:
            df = self.load_data()
            
            # Filter for week
            weekly_data = df[
                (df['timestamp'].dt.date >= week_start.date()) &
                (df['timestamp'].dt.date < week_end.date())
            ]
            
            if weekly_data.empty:
                return self._generate_no_data_report(week_start, "weekly")
            
            summary = self.generate_executive_summary(weekly_data)
            charts = self.create_performance_charts(weekly_data)
            
            # Weekly trends
            daily_stats = weekly_data.groupby(weekly_data['timestamp'].dt.date).agg({
                'test_result': lambda x: (x == 'PASS').mean() * 100,
                'performance_score': 'mean',
                'temperature_celsius': 'mean'
            }).reset_index()
            
            # Generate comprehensive analysis
            content = f"""
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{summary['total_units']:,}</div>
                    <div class="metric-label">Total Units Tested</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary['yield_rate']:.1f}%</div>
                    <div class="metric-label">Average Yield</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary['avg_performance']:.0f}</div>
                    <div class="metric-label">Performance Index</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{summary['avg_efficiency']:.1f}</div>
                    <div class="metric-label">Power Efficiency</div>
                </div>
            </div>
            
            <div class="section">
                <div class="section-title">📈 Weekly Trends</div>
                <div class="chart-container">
                    <img src="{charts['daily_yield']}" alt="Daily Yield Trend" style="max-width: 100%;">
                </div>
                <div class="chart-container">
                    <img src="{charts['temp_vs_perf']}" alt="Temperature vs Performance" style="max-width: 100%;">
                </div>
            </div>
            
            <div class="section">
                <div class="section-title">🏭 Manufacturing Analysis</div>
                <div class="chart-container">
                    <img src="{charts['yield_by_type']}" alt="Yield by Chip Type" style="max-width: 100%;">
                </div>
                
                <h4>Weekly Performance Summary:</h4>
                <ul>
                    <li><strong>Excellent Performance:</strong> {summary['excellent_units']:,} units ({summary['excellent_units']/summary['total_units']*100:.1f}%)</li>
                    <li><strong>Good Performance:</strong> {summary['good_units']:,} units ({summary['good_units']/summary['total_units']*100:.1f}%)</li>
                    <li><strong>Poor Performance:</strong> {summary['poor_units']:,} units ({summary['poor_units']/summary['total_units']*100:.1f}%)</li>
                    <li><strong>Thermal Incidents:</strong> {summary['thermal_incidents']} cases</li>
                </ul>
            </div>
            """
            
            # Generate HTML report
            template = Template(self.html_template)
            html_content = template.render(
                report_title="Weekly Performance Report",
                report_subtitle="Semiconductor Manufacturing Analytics",
                report_period=f"{week_start.strftime('%B %d')} - {week_end.strftime('%B %d, %Y')}",
                content=content,
                generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
            # Save report
            filename = f"weekly_report_{week_start.strftime('%Y%m%d')}.html"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Weekly report generated: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating weekly report: {e}")
            return None
    
    def _generate_alerts(self, summary: Dict) -> str:
        """Generate alert messages based on summary data"""
        alerts = []
        
        if summary['yield_rate'] < 90:
            alerts.append(f'<div class="alert alert-danger">🔴 <strong>Critical:</strong> Manufacturing yield is {summary["yield_rate"]:.1f}% (Target: >95%)</div>')
        elif summary['yield_rate'] < 95:
            alerts.append(f'<div class="alert alert-warning">⚠️ <strong>Warning:</strong> Manufacturing yield is {summary["yield_rate"]:.1f}% (Target: >95%)</div>')
        
        if summary['thermal_incidents'] > 10:
            alerts.append(f'<div class="alert alert-danger">🔴 <strong>Critical:</strong> {summary["thermal_incidents"]} thermal incidents detected</div>')
        elif summary['thermal_incidents'] > 0:
            alerts.append(f'<div class="alert alert-warning">⚠️ <strong>Monitor:</strong> {summary["thermal_incidents"]} thermal incidents detected</div>')
        
        if summary['defect_rate'] > 10:
            alerts.append(f'<div class="alert alert-danger">🔴 <strong>Critical:</strong> Defect rate is {summary["defect_rate"]:.1f}%</div>')
        elif summary['defect_rate'] > 5:
            alerts.append(f'<div class="alert alert-warning">⚠️ <strong>Warning:</strong> Defect rate is {summary["defect_rate"]:.1f}%</div>')
        
        if not alerts:
            alerts.append('<div class="alert alert-success">✅ <strong>Excellent:</strong> All metrics within target ranges</div>')
        
        return '\n'.join(alerts)
    
    def _generate_no_data_report(self, date: datetime, report_type: str) -> str:
        """Generate report when no data is available"""
        content = f"""
        <div class="alert alert-warning">
            ⚠️ <strong>No Data Available:</strong> No chip testing data found for the specified {report_type} period.
            Please check data collection systems and ensure data is being properly ingested.
        </div>
        """
        
        template = Template(self.html_template)
        html_content = template.render(
            report_title=f"{report_type.title()} Report - No Data",
            report_subtitle="Semiconductor Manufacturing Analytics",
            report_period=date.strftime("%B %d, %Y"),
            content=content,
            generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        filename = f"{report_type}_report_nodata_{date.strftime('%Y%m%d')}.html"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return filepath
    
    def send_email_report(self, report_path: str, recipients: List[str], 
                         subject: str = None) -> bool:
        """Send report via email"""
        if not self.email_user or not self.email_password:
            logger.warning("Email credentials not configured")
            return False
        
        try:
            if subject is None:
                subject = f"Chip Performance Report - {datetime.now().strftime('%Y-%m-%d')}"
            
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject
            
            # Email body
            body = f"""
            Please find attached the latest chip performance report.
            
            This report was automatically generated by the Semiconductor Performance Analytics Platform.
            
            For questions or issues, please contact the analytics team.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach report
            with open(report_path, 'rb') as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
            
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {os.path.basename(report_path)}'
            )
            
            msg.attach(part)
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_user, self.email_password)
            text = msg.as_string()
            server.sendmail(self.email_user, recipients, text)
            server.quit()
            
            logger.info(f"Report emailed to {recipients}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False

class ReportScheduler:
    """Schedule automated report generation"""
    
    def __init__(self, report_generator: ReportGenerator):
        self.report_generator = report_generator
        self.email_recipients = []
    
    def add_recipient(self, email: str):
        """Add email recipient for reports"""
        self.email_recipients.append(email)
    
    def schedule_daily_reports(self, time: str = "08:00"):
        """Schedule daily reports"""
        schedule.every().day.at(time).do(self._generate_and_send_daily)
        logger.info(f"Daily reports scheduled for {time}")
    
    def schedule_weekly_reports(self, day: str = "monday", time: str = "09:00"):
        """Schedule weekly reports"""
        getattr(schedule.every(), day.lower()).at(time).do(self._generate_and_send_weekly)
        logger.info(f"Weekly reports scheduled for {day} at {time}")
    
    def _generate_and_send_daily(self):
        """Generate and send daily report"""
        try:
            report_path = self.report_generator.generate_daily_report()
            if report_path and self.email_recipients:
                self.report_generator.send_email_report(
                    report_path, 
                    self.email_recipients,
                    f"Daily Chip Performance Report - {datetime.now().strftime('%Y-%m-%d')}"
                )
        except Exception as e:
            logger.error(f"Error in daily report generation: {e}")
    
    def _generate_and_send_weekly(self):
        """Generate and send weekly report"""
        try:
            report_path = self.report_generator.generate_weekly_report()
            if report_path and self.email_recipients:
                self.report_generator.send_email_report(
                    report_path, 
                    self.email_recipients,
                    f"Weekly Chip Performance Report - Week of {datetime.now().strftime('%Y-%m-%d')}"
                )
        except Exception as e:
            logger.error(f"Error in weekly report generation: {e}")
    
    def run_scheduler(self):
        """Run the report scheduler"""
        logger.info("Starting report scheduler...")
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

# Example usage
if __name__ == "__main__":
    # Create report generator
    report_gen = ReportGenerator()
    
    # Generate reports
    print("📊 Generating daily report...")
    daily_report = report_gen.generate_daily_report()
    if daily_report:
        print(f"✅ Daily report generated: {daily_report}")
    
    print("📊 Generating weekly report...")
    weekly_report = report_gen.generate_weekly_report()
    if weekly_report:
        print(f"✅ Weekly report generated: {weekly_report}")
    
    # Example scheduler setup
    scheduler = ReportScheduler(report_gen)
    scheduler.add_recipient("analytics@company.com")
    scheduler.schedule_daily_reports("08:00")
    scheduler.schedule_weekly_reports("monday", "09:00")
    
    print("📅 Report scheduler configured")
    print("Use scheduler.run_scheduler() to start automated reporting")