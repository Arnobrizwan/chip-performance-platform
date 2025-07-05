#!/bin/bash
# Enhanced Semiconductor Data Integration - Complete Setup and Test Script

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
print_header() { echo -e "${PURPLE}🚀 $1${NC}"; }

echo ""
print_header "Enhanced Semiconductor Analytics Platform"
echo "=============================================="
echo ""

# Step 1: Environment Setup
setup_environment() {
    print_info "Setting up Python environment..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_status "Virtual environment created"
    fi
    
    # Activate virtual environment
    source venv/bin/activate || source venv/Scripts/activate
    print_status "Virtual environment activated"
    
    # Install/upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    if [ ! -f requirements.txt ]; then
        cat > requirements.txt << 'REQUIREMENTS'
# Core Data Processing
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0

# API and Web Framework
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
streamlit>=1.28.0
pydantic>=2.0.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.17.0

# Data Sources
kaggle>=1.5.16

# ML and Analytics
joblib>=1.3.0
scipy>=1.11.0

# Development
pytest>=7.4.0
black>=23.0.0
flake8>=6.0.0
jupyter>=1.0.0
REQUIREMENTS
    fi
    
    pip install -r requirements.txt
    print_status "Dependencies installed"
}

# Step 2: Directory Structure
create_directories() {
    print_info "Creating directory structure..."
    
    # Core directories
    mkdir -p data/{raw/{kaggle,chip_test_data},processed/{kaggle,metadata},streaming,outputs}
    mkdir -p src/{data/{ingestion,processing},api/routes,visualization/dashboards}
    mkdir -p {notebooks,tests,docs,logs}
    
    # Create __init__.py files
    find src -type d -exec touch {}/__init__.py \;
    
    print_status "Directory structure created"
}

# Step 3: Create demo data
create_demo_data() {
    print_info "Creating demo semiconductor data..."
    
    cat > create_demo.py << 'DEMO'
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Generate demo SECOM-like data
n_samples = 1000
n_features = 50

# Create feature data
np.random.seed(42)
feature_data = np.random.normal(0, 1, (n_samples, n_features))

# Create feature names
feature_names = [f'sensor_{i:03d}' for i in range(n_features)]

df = pd.DataFrame(feature_data, columns=feature_names)

# Add chip-specific columns
df['chip_id'] = [f'DEMO_CHIP_{i:06d}' for i in range(n_samples)]
df['timestamp'] = pd.date_range('2024-01-01', periods=n_samples, freq='1H')
df['chip_type'] = np.random.choice(['CPU', 'GPU', 'ASIC'], n_samples)
df['manufacturer'] = np.random.choice(['Intel', 'AMD', 'NVIDIA'], n_samples)

# Add performance metrics
df['temperature_celsius'] = np.random.normal(70, 10, n_samples).clip(40, 90)
df['power_consumption_watts'] = np.random.normal(150, 30, n_samples).clip(80, 250)
df['performance_score'] = np.random.normal(8000, 1500, n_samples).clip(5000, 12000)
df['efficiency_score'] = df['performance_score'] / df['power_consumption_watts']
df['voltage_v'] = np.random.normal(1.2, 0.15, n_samples).clip(0.8, 1.8)
df['current_a'] = df['power_consumption_watts'] / df['voltage_v']
df['error_rate'] = np.random.exponential(0.0005, n_samples).clip(0, 0.01)
df['cache_hit_ratio'] = np.random.beta(9, 1, n_samples)
df['thermal_throttling'] = df['temperature_celsius'] > 85

# Add test results (93% pass rate)
df['test_result'] = np.random.choice(['PASS', 'FAIL'], n_samples, p=[0.93, 0.07])

# Save demo data
os.makedirs('data/raw/chip_test_data', exist_ok=True)
demo_path = 'data/raw/chip_test_data/secom_real_data.csv'
df.to_csv(demo_path, index=False)

print(f"✅ Demo data created: {demo_path}")
print(f"   Shape: {df.shape}")
print(f"   Pass rate: {(df['test_result'] == 'PASS').mean()*100:.1f}%")
DEMO

    python create_demo.py
    rm create_demo.py
    print_status "Demo data created"
}

# Main function
main() {
    print_header "Starting Setup"
    
    setup_environment
    create_directories
    create_demo_data
    
    print_status "🎉 Setup completed!"
    print_info "Your data is ready at: data/raw/chip_test_data/secom_real_data.csv"
    print_info "Next: Use your existing FastAPI and Streamlit applications"
}

# Run main function
main
