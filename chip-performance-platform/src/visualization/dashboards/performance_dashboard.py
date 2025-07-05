# src/visualization/dashboards/performance_dashboard.py - ENTERPRISE GRADE
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add src to path for imports
sys.path.append('src')

# Page configuration
st.set_page_config(
    page_title="Semiconductor Performance Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise-grade CSS styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1e3a8a;
        --secondary-color: #3b82f6;
        --accent-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
        --background-color: #f8fafc;
        --card-background: #ffffff;
        --text-primary: #1f2937;
        --text-secondary: #6b7280;
        --border-color: #e5e7eb;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom header */
    .main-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        padding: 2rem 1rem;
        border-radius: 0.75rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.025em;
    }
    
    .main-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
        font-weight: 400;
    }

    /* Enterprise metrics styling */
    .metric-container {
        background: var(--card-background);
        border: 1px solid var(--border-color);
        border-radius: 0.75rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: var(--background-color);
        border-radius: 0.5rem;
        padding: 0.25rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 0.5rem;
        color: var(--text-secondary);
        font-weight: 500;
        padding: 0.75rem 1.5rem;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--card-background);
        color: var(--primary-color);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--card-background);
        border-right: 1px solid var(--border-color);
    }

    /* Chart containers */
    .chart-container {
        background: var(--card-background);
        border: 1px solid var(--border-color);
        border-radius: 0.75rem;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* Status indicators */
    .status-excellent { color: var(--accent-color); font-weight: 600; }
    .status-good { color: #3b82f6; font-weight: 600; }
    .status-warning { color: var(--warning-color); font-weight: 600; }
    .status-critical { color: var(--danger-color); font-weight: 600; }

    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--border-color);
    }

    /* Alert styling */
    .alert-success {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 0.5rem;
        color: #166534;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .alert-warning {
        background-color: #fffbeb;
        border: 1px solid #fed7aa;
        border-radius: 0.5rem;
        color: #ea580c;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .alert-danger {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 0.5rem;
        color: #dc2626;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    /* Hide default streamlit elements */
    .stAlert, .stSuccess, .stWarning, .stError {
        border-radius: 0.5rem;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load chip performance data"""
    real_data_path = 'data/raw/chip_test_data/secom_real_data.csv'
    synthetic_data_path = 'data/raw/chip_test_data/chip_performance_data.csv'
    
    if os.path.exists(real_data_path):
        df = pd.read_csv(real_data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    elif os.path.exists(synthetic_data_path):
        df = pd.read_csv(synthetic_data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    else:
        st.error("⚠️ Unable to load performance data. Please contact system administrator.")
        st.stop()

def create_enterprise_header():
    """Create professional enterprise header"""
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">⚡ Semiconductor Performance Analytics</h1>
        <p class="main-subtitle">Real-time monitoring and intelligence for chip manufacturing excellence</p>
    </div>
    """, unsafe_allow_html=True)

def create_control_panel(df):
    """Create professional control panel"""
    st.sidebar.markdown("### 🎛️ Analysis Controls")
    
    # Date range
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()
    
    date_range = st.sidebar.date_input(
        "📅 Reporting Period",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Chip type selection
    chip_types = st.sidebar.multiselect(
        "🔧 Chip Technologies",
        options=sorted(df['chip_type'].unique()),
        default=sorted(df['chip_type'].unique())
    )
    
    # Manufacturer selection
    manufacturers = st.sidebar.multiselect(
        "🏭 Manufacturing Partners",
        options=sorted(df['manufacturer'].unique()),
        default=sorted(df['manufacturer'].unique())
    )
    
    # Performance threshold
    performance_threshold = st.sidebar.slider(
        "⚡ Performance Baseline",
        min_value=int(df['performance_score'].min()),
        max_value=int(df['performance_score'].max()),
        value=int(df['performance_score'].min()),
        step=100,
        help="Minimum performance score for analysis"
    )
    
    return date_range, chip_types, manufacturers, performance_threshold

def filter_data(df, date_range, chip_types, manufacturers, performance_threshold):
    """Apply business filters to data"""
    filtered_df = df.copy()
    
    # Date filter
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['timestamp'].dt.date >= start_date) &
            (filtered_df['timestamp'].dt.date <= end_date)
        ]
    
    # Business filters
    filtered_df = filtered_df[
        (filtered_df['chip_type'].isin(chip_types)) &
        (filtered_df['manufacturer'].isin(manufacturers)) &
        (filtered_df['performance_score'] >= performance_threshold)
    ]
    
    return filtered_df

def calculate_enterprise_metrics(df):
    """Calculate enterprise-level KPIs"""
    if df.empty:
        return {}
    
    total_chips = len(df)
    pass_rate = (df['test_result'] == 'PASS').mean() * 100
    avg_performance = df['performance_score'].mean()
    avg_temperature = df['temperature_celsius'].mean()
    avg_efficiency = df['efficiency_score'].mean()
    thermal_issues = (df['thermal_throttling'] == True).sum()
    
    # Performance categories
    excellent_performance = (df['performance_score'] >= df['performance_score'].quantile(0.9)).sum()
    poor_performance = (df['performance_score'] <= df['performance_score'].quantile(0.1)).sum()
    
    # Quality metrics
    defect_rate = (df['test_result'] == 'FAIL').mean() * 100
    
    return {
        'total_units': total_chips,
        'yield_rate': pass_rate,
        'defect_rate': defect_rate,
        'avg_performance': avg_performance,
        'avg_temperature': avg_temperature,
        'avg_efficiency': avg_efficiency,
        'thermal_incidents': thermal_issues,
        'excellent_units': excellent_performance,
        'underperforming_units': poor_performance,
        'thermal_compliance': ((~df['thermal_throttling']).sum() / total_chips * 100) if total_chips > 0 else 100
    }

def display_executive_metrics(metrics):
    """Display executive-level metrics"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="📊 Production Volume",
            value=f"{metrics['total_units']:,}",
            help="Total units analyzed in reporting period"
        )
    
    with col2:
        yield_rate = metrics['yield_rate']
        yield_delta = "Normal" if yield_rate >= 95 else "Below Target"
        st.metric(
            label="✅ Manufacturing Yield",
            value=f"{yield_rate:.1f}%",
            delta=yield_delta,
            delta_color="normal" if yield_rate >= 95 else "inverse"
        )
    
    with col3:
        performance = metrics['avg_performance']
        st.metric(
            label="⚡ Performance Index",
            value=f"{performance:.0f}",
            help="Average performance score across all units"
        )
    
    with col4:
        efficiency = metrics['avg_efficiency']
        st.metric(
            label="🔋 Power Efficiency",
            value=f"{efficiency:.1f}",
            help="Performance per watt ratio"
        )
    
    with col5:
        thermal_compliance = metrics['thermal_compliance']
        st.metric(
            label="🌡️ Thermal Compliance",
            value=f"{thermal_compliance:.1f}%",
            delta="Excellent" if thermal_compliance >= 98 else "Monitor",
            delta_color="normal" if thermal_compliance >= 98 else "inverse"
        )

def show_executive_dashboard(df):
    """Executive overview dashboard"""
    metrics = calculate_enterprise_metrics(df)
    
    st.markdown('<h2 class="section-header">Executive Summary</h2>', unsafe_allow_html=True)
    
    # Key metrics
    display_executive_metrics(metrics)
    
    st.markdown("---")
    
    # Status indicators
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Overall health status
        yield_rate = metrics['yield_rate']
        if yield_rate >= 95:
            status = "🟢 Excellent"
            status_class = "status-excellent"
        elif yield_rate >= 90:
            status = "🟡 Good"
            status_class = "status-good"
        elif yield_rate >= 85:
            status = "🟠 Monitor"
            status_class = "status-warning"
        else:
            status = "🔴 Critical"
            status_class = "status-critical"
        
        st.markdown(f"**Manufacturing Health**")
        st.markdown(f'<p class="{status_class}">{status}</p>', unsafe_allow_html=True)
    
    with col2:
        # Quality status
        defect_rate = metrics['defect_rate']
        if defect_rate <= 2:
            quality_status = "🟢 Superior"
            quality_class = "status-excellent"
        elif defect_rate <= 5:
            quality_status = "🟡 Acceptable"
            quality_class = "status-good"
        elif defect_rate <= 10:
            quality_status = "🟠 Review Required"
            quality_class = "status-warning"
        else:
            quality_status = "🔴 Action Required"
            quality_class = "status-critical"
        
        st.markdown(f"**Quality Index**")
        st.markdown(f'<p class="{quality_class}">{quality_status}</p>', unsafe_allow_html=True)
    
    with col3:
        # Thermal status
        thermal_incidents = metrics['thermal_incidents']
        if thermal_incidents == 0:
            thermal_status = "🟢 Optimal"
            thermal_class = "status-excellent"
        elif thermal_incidents <= 5:
            thermal_status = "🟡 Stable"
            thermal_class = "status-good"
        elif thermal_incidents <= 20:
            thermal_status = "🟠 Monitor"
            thermal_class = "status-warning"
        else:
            thermal_status = "🔴 Critical"
            thermal_class = "status-critical"
        
        st.markdown(f"**Thermal Management**")
        st.markdown(f'<p class="{thermal_class}">{thermal_status}</p>', unsafe_allow_html=True)
    
    # Quick insights
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance distribution
        fig1 = px.histogram(
            df, x='performance_score', nbins=25,
            title="Performance Score Distribution",
            color_discrete_sequence=['#3b82f6']
        )
        fig1.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=350
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Yield trends
        daily_yield = df.groupby(df['timestamp'].dt.date).agg({
            'test_result': lambda x: (x == 'PASS').mean() * 100
        }).reset_index()
        
        fig2 = px.line(
            daily_yield, x='timestamp', y='test_result',
            title="Daily Manufacturing Yield Trend",
            color_discrete_sequence=['#10b981']
        )
        fig2.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=350,
            yaxis_title="Yield Rate (%)"
        )
        st.plotly_chart(fig2, use_container_width=True)

def show_production_analytics(df):
    """Production analytics dashboard"""
    st.markdown('<h2 class="section-header">Production Analytics</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance by chip type
        fig1 = px.box(
            df, x='chip_type', y='performance_score',
            title="Performance Analysis by Technology",
            color='chip_type',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig1.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Efficiency analysis
        fig3 = px.scatter(
            df, x='power_consumption_watts', y='performance_score',
            color='chip_type', size='efficiency_score',
            title="Power Efficiency Analysis",
            hover_data=['manufacturer']
        )
        fig3.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # Manufacturing yield by type
        yield_by_type = df.groupby(['chip_type', 'test_result']).size().reset_index(name='count')
        fig2 = px.bar(
            yield_by_type, x='chip_type', y='count', color='test_result',
            title="Manufacturing Yield by Technology",
            color_discrete_map={'PASS': '#10b981', 'FAIL': '#ef4444'}
        )
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Temperature analysis
        fig4 = px.scatter(
            df, x='temperature_celsius', y='performance_score',
            color='thermal_throttling', size='power_consumption_watts',
            title="Thermal Performance Analysis",
            color_discrete_map={True: '#ef4444', False: '#10b981'}
        )
        fig4.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        st.plotly_chart(fig4, use_container_width=True)

def show_quality_intelligence(df):
    """Quality intelligence dashboard"""
    st.markdown('<h2 class="section-header">Quality Intelligence</h2>', unsafe_allow_html=True)
    
    try:
        from analysis.performance.kpi_calculator import AdvancedKPICalculator
        
        calculator = AdvancedKPICalculator()
        report = calculator.generate_comprehensive_report(df)
        
        # Quality metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Quality Score", f"{report['summary']['average_score']:.1f}/100")
        with col2:
            st.metric("Manufacturing Yield", f"{report['yield_metrics']['basic_yield']:.1f}%")
        with col3:
            st.metric("Premium Quality", f"{report['yield_metrics']['quality_yield']:.1f}%")
        with col4:
            st.metric("Process Stability", f"{100 - report['summary']['outlier_rate']:.1f}%")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Quality grade distribution
            grade_data = report['summary']['grade_distribution']
            if grade_data:
                colors = {
                    'A+': '#059669', 'A': '#10b981', 'A-': '#34d399',
                    'B+': '#3b82f6', 'B': '#60a5fa', 'B-': '#93c5fd',
                    'C+': '#f59e0b', 'C': '#fbbf24', 'C-': '#fcd34d',
                    'F': '#ef4444'
                }
                
                fig = px.bar(
                    x=list(grade_data.keys()), y=list(grade_data.values()),
                    title="Quality Grade Distribution",
                    color=list(grade_data.keys()),
                    color_discrete_map=colors
                )
                fig.update_layout(
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top performers
            st.markdown("**🏆 Excellence Recognition**")
            top_performers = pd.DataFrame(report['top_performers'])
            if not top_performers.empty:
                display_performers = top_performers[['chip_id', 'chip_type', 'overall_performance_score']].head(8)
                display_performers.columns = ['Unit ID', 'Technology', 'Score']
                display_performers['Score'] = display_performers['Score'].round(1)
                st.dataframe(display_performers, use_container_width=True, hide_index=True)
        
        # Actionable insights
        st.markdown("**💡 Quality Insights**")
        insights_displayed = False
        for rec in report['recommendations']:
            if "🔴" in rec:
                st.markdown(f'<div class="alert-danger">🔴 CRITICAL: {rec.replace("🔴", "").strip()}</div>', unsafe_allow_html=True)
                insights_displayed = True
            elif "⚠️" in rec:
                st.markdown(f'<div class="alert-warning">⚠️ ATTENTION: {rec.replace("⚠️", "").strip()}</div>', unsafe_allow_html=True)
                insights_displayed = True
        
        if not insights_displayed:
            st.markdown('<div class="alert-success">✅ All quality metrics within operational excellence standards</div>', unsafe_allow_html=True)
        
    except ImportError:
        st.warning("⚠️ Advanced quality analytics temporarily unavailable. Contact IT support.")

def show_predictive_insights(df):
    """Predictive insights dashboard"""
    st.markdown('<h2 class="section-header">Predictive Analytics</h2>', unsafe_allow_html=True)
    
    try:
        from analysis.anomaly.outlier_detector import AdvancedAnomalyDetector
        
        detector = AdvancedAnomalyDetector(contamination_rate=0.05)
        report = detector.generate_anomaly_report(df)
        
        # Risk indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            anomaly_rate = report['summary']['anomaly_rate']
            st.metric("Process Stability", f"{100 - anomaly_rate:.1f}%")
        
        with col2:
            total_anomalies = report['summary']['total_anomalies']
            st.metric("Anomalies Detected", f"{total_anomalies:,}")
        
        with col3:
            risk_level = "LOW" if anomaly_rate < 5 else "MEDIUM" if anomaly_rate < 10 else "HIGH"
            st.metric("Risk Assessment", risk_level)
        
        with col4:
            methods_used = len(report['summary']['detection_methods_used'])
            st.metric("Analysis Depth", f"{methods_used} Models")
        
        # Risk analysis
        if 'pattern_analysis' in report and report['pattern_analysis'].get('anomaly_count', 0) > 0:
            patterns = report['pattern_analysis']
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'chip_type_distribution' in patterns and patterns['chip_type_distribution']:
                    type_data = patterns['chip_type_distribution']
                    fig = px.pie(
                        values=list(type_data.values()), 
                        names=list(type_data.keys()),
                        title="Risk Distribution by Technology"
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'manufacturer_distribution' in patterns and patterns['manufacturer_distribution']:
                    mfg_data = patterns['manufacturer_distribution']
                    fig = px.bar(
                        x=list(mfg_data.keys()), 
                        y=list(mfg_data.values()),
                        title="Risk Events by Manufacturing Partner",
                        color_discrete_sequence=['#ef4444']
                    )
                    fig.update_layout(
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Predictive recommendations
        st.markdown("**🎯 Predictive Recommendations**")
        recommendations_shown = False
        for rec in report['recommendations']:
            if "🔴" in rec or "🚨" in rec:
                st.markdown(f'<div class="alert-danger">🚨 IMMEDIATE ACTION: {rec.replace("🔴", "").replace("🚨", "").strip()}</div>', unsafe_allow_html=True)
                recommendations_shown = True
            elif "⚠️" in rec:
                st.markdown(f'<div class="alert-warning">⚠️ PREVENTIVE ACTION: {rec.replace("⚠️", "").strip()}</div>', unsafe_allow_html=True)
                recommendations_shown = True
        
        if not recommendations_shown:
            st.markdown('<div class="alert-success">✅ No immediate risks detected. Continue monitoring protocols.</div>', unsafe_allow_html=True)
        
    except ImportError:
        st.warning("⚠️ Predictive analytics module temporarily unavailable. Contact IT support.")

def main():
    """Main enterprise dashboard application"""
    
    # Enterprise header
    create_enterprise_header()
    
    # Load data
    df = load_data()
    
    # Control panel
    date_range, chip_types, manufacturers, performance_threshold = create_control_panel(df)
    
    # Apply filters
    filtered_df = filter_data(df, date_range, chip_types, manufacturers, performance_threshold)
    
    if filtered_df.empty:
        st.warning("⚠️ No data matches current filter criteria. Please adjust analysis parameters.")
        st.stop()
    
    # Show active filters summary
    if len(filtered_df) < len(df):
        st.info(f"📊 Analysis Scope: {len(filtered_df):,} units ({len(filtered_df)/len(df)*100:.1f}% of total production)")
    
    # Main navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Executive Dashboard", 
        "🏭 Production Analytics", 
        "🎯 Quality Intelligence", 
        "🔮 Predictive Insights"
    ])
    
    with tab1:
        show_executive_dashboard(filtered_df)
    
    with tab2:
        show_production_analytics(filtered_df)
    
    with tab3:
        show_quality_intelligence(filtered_df)
    
    with tab4:
        show_predictive_insights(filtered_df)

if __name__ == "__main__":
    main()