#!/bin/bash
# Full Setup - Demo Data + Kaggle Integration

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

echo "🚀 Full Semiconductor Data Integration"
echo "======================================"

# 1. Setup environment
setup_env() {
    print_info "Setting up environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    
    source venv/bin/activate
    pip install --upgrade pip
    
    # Install packages
    pip install pandas numpy scikit-learn fastapi uvicorn streamlit plotly kaggle
    
    print_status "Environment ready"
}

# 2. Create directories
setup_dirs() {
    mkdir -p data/{raw/{kaggle,chip_test_data},processed/{kaggle,metadata},streaming,outputs}
    mkdir -p src/{data/{ingestion,processing},api/routes,visualization/dashboards}
    find src -type d -exec touch {}/__init__.py \;
    print_status "Directories created"
}

# 3. Create demo data
create_demo() {
    print_info "Creating demo data..."
    
python3 << 'PYTHON'
import pandas as pd
import numpy as np
import os

# Demo data generation
np.random.seed(42)
n = 1000

df = pd.DataFrame({
    'chip_id': [f'DEMO_{i:06d}' for i in range(n)],
    'timestamp': pd.date_range('2024-01-01', periods=n, freq='1H'),
    'chip_type': np.random.choice(['CPU', 'GPU', 'ASIC'], n),
    'manufacturer': np.random.choice(['Intel', 'AMD', 'NVIDIA'], n),
    'architecture_nm': np.random.choice(['7nm', '10nm', '14nm'], n),
    'temperature_celsius': np.random.normal(70, 10, n).clip(40, 90),
    'power_consumption_watts': np.random.normal(150, 30, n).clip(80, 250),
    'performance_score': np.random.normal(8000, 1500, n).clip(5000, 12000),
    'voltage_v': np.random.normal(1.2, 0.15, n).clip(0.8, 1.8),
    'test_result': np.random.choice(['PASS', 'FAIL'], n, p=[0.93, 0.07])
})

# Add derived metrics
df['efficiency_score'] = df['performance_score'] / df['power_consumption_watts']
df['current_a'] = df['power_consumption_watts'] / df['voltage_v']
df['error_rate'] = np.random.exponential(0.0005, n).clip(0, 0.01)
df['cache_hit_ratio'] = np.random.beta(9, 1, n)
df['thermal_throttling'] = df['temperature_celsius'] > 85

os.makedirs('data/raw/chip_test_data', exist_ok=True)
df.to_csv('data/raw/chip_test_data/secom_real_data.csv', index=False)

print(f"✅ Demo data: {len(df)} records created")
PYTHON

    print_status "Demo data created"
}

# 4. Setup Kaggle API
setup_kaggle() {
    print_info "Checking Kaggle API..."
    
    if [ ! -f ~/.kaggle/kaggle.json ]; then
        print_warning "Kaggle API not configured"
        echo "To add real Kaggle datasets:"
        echo "1. Go to https://www.kaggle.com/account"
        echo "2. Create API token"
        echo "3. Place kaggle.json in ~/.kaggle/"
        echo "4. Run: chmod 600 ~/.kaggle/kaggle.json"
        echo "5. Then run: ./full_setup.sh kaggle"
        return 1
    else
        chmod 600 ~/.kaggle/kaggle.json
        print_status "Kaggle API ready"
        return 0
    fi
}

# 5. Download Kaggle datasets
download_kaggle() {
    print_info "Downloading Kaggle semiconductor datasets..."
    
    # Key semiconductor datasets
    datasets=(
        "uciml/secom"
        "stephanmatzka/quality-prediction-in-a-mining-process"
        "shivamb/machine-predictive-maintenance-classification"
        "uciml/steel-plates-faults"
        "uciml/gas-sensor-arrays-in-open-sampling-settings"
    )
    
    for dataset in "${datasets[@]}"; do
        dataset_name=$(echo $dataset | sed 's/.*\///')
        output_dir="data/raw/kaggle/$dataset_name"
        
        print_info "Downloading: $dataset"
        
        if kaggle datasets download -d "$dataset" -p "$output_dir" --unzip; then
            print_status "Downloaded: $dataset_name"
        else
            print_warning "Failed: $dataset_name"
        fi
    done
    
    print_status "Kaggle downloads completed"
}

# 6. Generate summary
create_summary() {
    echo ""
    echo "📊 DATA SUMMARY"
    echo "==============="
    
    # Demo data
    if [ -f "data/raw/chip_test_data/secom_real_data.csv" ]; then
        demo_count=$(wc -l < data/raw/chip_test_data/secom_real_data.csv)
        echo "✅ Demo Data: $((demo_count-1)) records"
    fi
    
    # Kaggle data
    kaggle_files=$(find data/raw/kaggle -name "*.csv" 2>/dev/null | wc -l)
    if [ $kaggle_files -gt 0 ]; then
        echo "✅ Kaggle Datasets: $kaggle_files CSV files"
        echo "📁 Kaggle data location: data/raw/kaggle/"
    else
        echo "⚠️  Kaggle Datasets: None (API not configured)"
    fi
    
    echo ""
    echo "🚀 READY TO USE:"
    echo "- Demo data: data/raw/chip_test_data/secom_real_data.csv"
    echo "- Real data: data/raw/kaggle/ (if downloaded)"
    echo "- Your FastAPI will now see all datasets!"
}

# Main execution
case "${1:-all}" in
    "demo")
        setup_env && setup_dirs && create_demo && create_summary
        ;;
    "kaggle")
        if setup_kaggle; then
            download_kaggle && create_summary
        fi
        ;;
    "all")
        setup_env && setup_dirs && create_demo
        if setup_kaggle; then
            download_kaggle
        fi
        create_summary
        ;;
    *)
        echo "Usage: $0 {demo|kaggle|all}"
        echo ""
        echo "  demo   - Create demo data only"
        echo "  kaggle - Download Kaggle datasets (requires API)"
        echo "  all    - Create demo + try Kaggle (default)"
        ;;
esac
