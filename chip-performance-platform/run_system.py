# run_system.py - Complete System Startup Script
import os
import sys
import subprocess
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SemiconductorAnalyticsSystem:
    """Complete system management for semiconductor analytics platform"""
    
    def __init__(self):
        self.base_dir = Path.cwd()
        self.src_dir = self.base_dir / "src"
        self.data_dir = self.base_dir / "data"
        
    def check_environment(self):
        """Check if environment is properly set up"""
        logger.info("🔍 Checking system environment...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            logger.error("❌ Python 3.8+ required")
            return False
        logger.info(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check required directories
        required_dirs = [
            "src/api/routes",
            "src/models/ml_models",
            "src/analysis/anomaly",
            "src/analysis/performance", 
            "src/visualization/dashboards",
            "src/visualization/reports",
            "src/data/ingestion",
            "data/raw/chip_test_data",
            "data/models/trained_models",
            "data/outputs/reports",
            "data/outputs/exports",
            "data/streaming"
        ]
        
        for dir_path in required_dirs:
            full_path = self.base_dir / dir_path
            if not full_path.exists():
                logger.info(f"📁 Creating directory: {dir_path}")
                full_path.mkdir(parents=True, exist_ok=True)
            else:
                logger.info(f"✅ Directory exists: {dir_path}")
        
        return True
    
    def install_requirements(self):
        """Install required Python packages"""
        logger.info("📦 Installing Python requirements...")
        
        requirements = [
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
            "streamlit>=1.28.0",
            "pandas>=2.0.0",
            "numpy>=1.24.0",
            "scikit-learn>=1.3.0",
            "plotly>=5.17.0",
            "seaborn>=0.12.0",
            "matplotlib>=3.7.0",
            "pydantic>=2.0.0",
            "python-multipart>=0.0.6",
            "joblib>=1.3.0",
            "scipy>=1.11.0",
            "jinja2>=3.1.0",
            "python-dotenv>=1.0.0"
        ]
        
        for package in requirements:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                logger.info(f"✅ Installed: {package}")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to install {package}: {e}")
                return False
        
        return True
    
    def setup_data(self):
        """Set up initial data"""
        logger.info("🗃️ Setting up initial data...")
        
        # Check if we have data files
        data_files = [
            "data/raw/chip_test_data/secom_real_data.csv",
            "data/raw/chip_test_data/chip_performance_data.csv"
        ]
        
        data_exists = any(os.path.exists(f) for f in data_files)
        
        if not data_exists:
            logger.info("📊 No data found, generating synthetic data...")
            try:
                # Generate synthetic data
                self._generate_synthetic_data()
                logger.info("✅ Synthetic data generated")
            except Exception as e:
                logger.error(f"❌ Failed to generate synthetic data: {e}")
                return False
        else:
            logger.info("✅ Data files found")
        
        return True
    
    def _generate_synthetic_data(self):
        """Generate synthetic chip performance data"""
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        logger.info("🏭 Generating synthetic semiconductor data...")
        
        # Generate 10,000 chip records
        n_records = 10000
        
        # Basic chip info
        chip_types = ['CPU', 'GPU', 'ASIC', 'FPGA', 'SoC']
        manufacturers = ['Intel', 'AMD', 'NVIDIA', 'TSMC', 'Samsung']
        architectures = ['3nm', '5nm', '7nm', '10nm', '14nm']
        
        df = pd.DataFrame({
            'chip_id': [f'CHIP_{i:06d}' for i in range(n_records)],
            'timestamp': pd.date_range('2024-01-01', periods=n_records, freq='5min'),
            'chip_type': np.random.choice(chip_types, n_records, p=[0.35, 0.25, 0.15, 0.15, 0.10]),
            'manufacturer': np.random.choice(manufacturers, n_records),
            'architecture_nm': np.random.choice(architectures, n_records, p=[0.1, 0.2, 0.3, 0.25, 0.15])
        })
        
        # Performance characteristics
        df['clock_speed_ghz'] = np.random.normal(3.2, 0.8, n_records).clip(1.5, 5.5)
        df['power_consumption_watts'] = np.random.normal(150, 40, n_records).clip(50, 400)
        df['temperature_celsius'] = (
            df['power_consumption_watts'] * 0.25 + 
            np.random.normal(40, 8, n_records)
        ).clip(35, 95)
        df['voltage_v'] = np.random.normal(1.2, 0.15, n_records).clip(0.8, 1.8)
        df['current_a'] = df['power_consumption_watts'] / df['voltage_v']
        
        # Performance score based on chip type and specs
        type_base_performance = {
            'CPU': 8500, 'GPU': 12000, 'FPGA': 6500, 'ASIC': 15000, 'SoC': 7000
        }
        df['performance_score'] = df.apply(
            lambda row: type_base_performance[row['chip_type']] * 
                       (row['clock_speed_ghz'] / 3.0) * 
                       np.random.normal(1.0, 0.1), axis=1
        ).clip(3000, 20000)
        
        # Quality metrics
        df['error_rate'] = np.random.exponential(0.0005, n_records).clip(0, 0.01)
        df['cache_hit_ratio'] = np.random.beta(9, 1, n_records)
        df['thermal_throttling'] = df['temperature_celsius'] > 85
        df['efficiency_score'] = df['performance_score'] / df['power_consumption_watts']
        
        # Test results based on performance and temperature
        pass_probability = 0.95 - (df['temperature_celsius'] > 90) * 0.2 - (df['error_rate'] > 0.005) * 0.3
        df['test_result'] = np.where(np.random.random(n_records) < pass_probability, 'PASS', 'FAIL')
        
        # Round numerical values
        numeric_columns = ['clock_speed_ghz', 'power_consumption_watts', 'temperature_celsius', 
                          'voltage_v', 'current_a', 'efficiency_score', 'error_rate', 'cache_hit_ratio']
        df[numeric_columns] = df[numeric_columns].round(3)
        df['performance_score'] = df['performance_score'].round(0).astype(int)
        
        # Save synthetic data
        output_path = self.data_dir / "raw/chip_test_data/chip_performance_data.csv"
        df.to_csv(output_path, index=False)
        
        logger.info(f"✅ Generated {len(df)} synthetic chip records")
        return df
    
    def start_api_server(self, port=8000):
        """Start the FastAPI server"""
        logger.info(f"🚀 Starting FastAPI server on port {port}...")
        
        # Add src to Python path
        src_path = str(self.src_dir.absolute())
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        try:
            import uvicorn
            from api.routes.data_routes import app
            
            # Run the server
            uvicorn.run(
                app,
                host="0.0.0.0",
                port=port,
                log_level="info",
                reload=True
            )
        except ImportError as e:
            logger.error(f"❌ Failed to start API server: {e}")
            logger.error("Make sure all dependencies are installed")
            return False
        except Exception as e:
            logger.error(f"❌ API server error: {e}")
            return False
    
    def start_dashboard(self, port=8501):
        """Start the Streamlit dashboard"""
        logger.info(f"📊 Starting Streamlit dashboard on port {port}...")
        
        dashboard_path = self.src_dir / "visualization/dashboards/performance_dashboard.py"
        
        if not dashboard_path.exists():
            logger.error(f"❌ Dashboard file not found: {dashboard_path}")
            return False
        
        try:
            subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", 
                str(dashboard_path), "--server.port", str(port)
            ])
            logger.info(f"✅ Dashboard started at http://localhost:{port}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to start dashboard: {e}")
            return False
    
    def setup_complete_system(self):
        """Set up the complete system"""
        logger.info("🔧 Setting up complete semiconductor analytics system...")
        
        # Check environment
        if not self.check_environment():
            logger.error("❌ Environment check failed")
            return False
        
        # Install requirements
        logger.info("📦 Installing requirements...")
        try:
            self.install_requirements()
        except Exception as e:
            logger.warning(f"⚠️ Some packages may not have installed: {e}")
        
        # Setup data
        if not self.setup_data():
            logger.error("❌ Data setup failed")
            return False
        
        # Create configuration files
        self._create_config_files()
        
        logger.info("✅ System setup complete!")
        return True
    
    def _create_config_files(self):
        """Create necessary configuration files"""
        logger.info("📝 Creating configuration files...")
        
        # Create .env file
        env_content = """# Semiconductor Analytics Configuration
API_HOST=0.0.0.0
API_PORT=8000
DASHBOARD_PORT=8501
ML_MODEL_DIR=data/models/trained_models
DATA_DIR=data/raw/chip_test_data
REPORTS_DIR=data/outputs/reports

# Email configuration (optional)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USER=
EMAIL_PASSWORD=

# Database configuration (optional)
DATABASE_URL=sqlite:///data/chip_analytics.db
"""
        
        env_path = self.base_dir / ".env"
        with open(env_path, 'w') as f:
            f.write(env_content)
        
        logger.info("✅ Created .env configuration file")
    
    def show_system_status(self):
        """Show system status and available services"""
        logger.info("📋 System Status:")
        logger.info("=" * 50)
        
        # Check data
        data_files = [
            "data/raw/chip_test_data/secom_real_data.csv",
            "data/raw/chip_test_data/chip_performance_data.csv"
        ]
        
        data_available = any(os.path.exists(f) for f in data_files)
        logger.info(f"📊 Data Available: {'✅ Yes' if data_available else '❌ No'}")
        
        # Check key modules
        try:
            from api.routes.data_routes import app
            logger.info("🚀 FastAPI: ✅ Available")
        except ImportError:
            logger.info("🚀 FastAPI: ❌ Not available")
        
        dashboard_path = self.src_dir / "visualization/dashboards/performance_dashboard.py"
        logger.info(f"📊 Dashboard: {'✅ Available' if dashboard_path.exists() else '❌ Not available'}")
        
        logger.info("\n🎯 Available Services:")
        logger.info("1. FastAPI Server: http://localhost:8000")
        logger.info("2. API Documentation: http://localhost:8000/docs")
        logger.info("3. Streamlit Dashboard: http://localhost:8501")
        
        logger.info("\n🛠️ Quick Start Commands:")
        logger.info("python run_system.py api          # Start API server")
        logger.info("python run_system.py dashboard    # Start dashboard")
        logger.info("python run_system.py both         # Start both services")

def main():
    """Main function"""
    system = SemiconductorAnalyticsSystem()
    
    if len(sys.argv) < 2:
        print("🚀 Semiconductor Analytics Platform")
        print("=" * 40)
        print("Usage:")
        print("  python run_system.py setup        # Setup complete system")
        print("  python run_system.py api          # Start API server")
        print("  python run_system.py dashboard    # Start dashboard")
        print("  python run_system.py both         # Start both services")
        print("  python run_system.py status       # Show system status")
        return
    
    command = sys.argv[1].lower()
    
    if command == "setup":
        system.setup_complete_system()
        system.show_system_status()
    
    elif command == "api":
        system.start_api_server()
    
    elif command == "dashboard":
        system.start_dashboard()
    
    elif command == "both":
        # Start dashboard in background
        system.start_dashboard()
        time.sleep(3)  # Give dashboard time to start
        
        # Start API server (this will block)
        system.start_api_server()
    
    elif command == "status":
        system.show_system_status()
    
    else:
        print(f"❌ Unknown command: {command}")
        print("Use 'python run_system.py' for help")

if __name__ == "__main__":
    main()