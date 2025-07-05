# src/data/ingestion/enhanced_kaggle_semiconductor_datasets.py
import os
import pandas as pd
import numpy as np
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import json
import sqlite3

# Import the real-time streamer
from .real_time_streamer import RealTimeDataStreamer, ChipDataSimulator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedSemiconductorDatasetManager:
    """Enhanced version that integrates Kaggle datasets with real-time streaming"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.kaggle_datasets = {
            # Core semiconductor manufacturing datasets
            'secom': 'uciml/secom',
            'semiconductor_manufacturing': 'paresh2047/uci-semcom',
            'semiconductor_defects': 'humananalog/semiconductor-defect-detection',
            'chip_failure': 'shivamb/machine-predictive-maintenance-classification',
            
            # Process monitoring and quality control
            'process_monitoring': 'shasun/tool-wear-detection-in-cnc-machine',
            'quality_control': 'stephanmatzka/quality-prediction-in-a-mining-process',
            'manufacturing_defects': 'ravishah1/defect-detection-in-manufacturing',
            
            # IoT and sensor data
            'sensor_data': 'uciml/gas-sensor-arrays-in-open-sampling-settings',
            'industrial_iot': 'inIT-OWL/production-quality-dataset',
            'equipment_monitoring': 'stephanmatzka/predictive-maintenance-dataset-ai4i-2020',
            
            # Electronic components and testing
            'electronic_components': 'crawford/electronic-components',
            'component_failure': 'shivamb/machine-predictive-maintenance-classification',
            'pcb_defects': 'humananalog/pcb-defect-detection',
            
            # Additional manufacturing datasets
            'steel_defects': 'uciml/steel-plates-faults',
            'glass_manufacturing': 'uciml/glass',
            'wine_quality': 'uciml/red-wine-quality-cortez-et-al-2009',  # Process optimization
            
            # New semiconductor-specific datasets
            'wafer_map': 'qingyi7/semiconductor-wafer-fault-detection',
            'semiconductor_process': 'saurabhshahane/semiconductor-manufacturing-process-dataset',
            'chip_testing': 'prashant111/semiconductor-dataset',
            'failure_analysis': 'mahmoudima/semiconductor-failure-analysis'
        }
        
        self.api = KaggleApi()
        try:
            self.api.authenticate()
            logger.info("✅ Kaggle API authenticated successfully")
        except Exception as e:
            logger.error(f"❌ Kaggle API authentication failed: {e}")
            self.api = None
        
        # Initialize real-time streamer
        self.real_time_streamer = RealTimeDataStreamer()
        
    def search_and_discover_datasets(self, keywords: List[str] = None) -> pd.DataFrame:
        """Enhanced search with more semiconductor-specific keywords"""
        if keywords is None:
            keywords = [
                'semiconductor', 'chip manufacturing', 'wafer', 'defect detection',
                'quality control', 'process monitoring', 'manufacturing', 'IoT sensor',
                'predictive maintenance', 'failure detection', 'electronic components',
                'silicon wafer', 'fab process', 'yield optimization', 'test data',
                'integrated circuit', 'microchip', 'VLSI', 'SECOM'
            ]
        
        if not self.api:
            logger.error("Kaggle API not available")
            return pd.DataFrame()
        
        all_datasets = []
        
        for keyword in keywords:
            try:
                datasets = self.api.dataset_list(search=keyword, max_size=100)
                for dataset in datasets:
                    all_datasets.append({
                        'title': dataset.title,
                        'ref': dataset.ref,
                        'size': dataset.totalBytes,
                        'download_count': dataset.downloadCount,
                        'usability_rating': dataset.usabilityRating,
                        'keyword': keyword,
                        'url': f"https://www.kaggle.com/datasets/{dataset.ref}",
                        'subtitle': getattr(dataset, 'subtitle', ''),
                        'last_updated': getattr(dataset, 'lastUpdated', ''),
                        'license_name': getattr(dataset, 'licenseName', '')
                    })
                logger.info(f"Found {len(datasets)} datasets for keyword: {keyword}")
                
            except Exception as e:
                logger.warning(f"Error searching for {keyword}: {e}")
        
        df = pd.DataFrame(all_datasets)
        if not df.empty:
            df = df.drop_duplicates(subset=['ref'])
            df = df.sort_values('usability_rating', ascending=False)
            
            # Save search results
            search_results_path = self.data_dir / "processed" / "kaggle_search_results.csv"
            search_results_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(search_results_path, index=False)
            logger.info(f"Search results saved to: {search_results_path}")
        
        return df
    
    def download_dataset_with_validation(self, dataset_ref: str, extract_path: str = None) -> Dict[str, Any]:
        """Download dataset with comprehensive validation"""
        if not self.api:
            return {'success': False, 'error': 'Kaggle API not available'}
        
        try:
            if extract_path is None:
                extract_path = self.data_dir / "kaggle" / dataset_ref.replace('/', '_')
            
            extract_path = Path(extract_path)
            extract_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"📥 Downloading dataset: {dataset_ref}")
            
            # Get dataset metadata first
            try:
                dataset_info = self.api.dataset_metadata(dataset_ref)
                metadata = {
                    'title': dataset_info.title,
                    'description': dataset_info.description,
                    'size': dataset_info.totalBytes,
                    'files': [f.name for f in dataset_info.files],
                    'download_date': datetime.now().isoformat()
                }
            except:
                metadata = {'download_date': datetime.now().isoformat()}
            
            # Download the dataset
            self.api.dataset_download_files(dataset_ref, path=extract_path, unzip=True)
            
            # Validate downloaded files
            csv_files = list(extract_path.glob("*.csv"))
            validation_results = []
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file, nrows=100)  # Sample first 100 rows
                    file_validation = {
                        'file': csv_file.name,
                        'size_mb': csv_file.stat().st_size / (1024 * 1024),
                        'shape': [len(df), len(df.columns)],
                        'columns': list(df.columns),
                        'valid': True
                    }
                except Exception as e:
                    file_validation = {
                        'file': csv_file.name,
                        'valid': False,
                        'error': str(e)
                    }
                
                validation_results.append(file_validation)
            
            logger.info(f"✅ Dataset downloaded to: {extract_path}")
            return {
                'success': True,
                'path': str(extract_path),
                'metadata': metadata,
                'files': validation_results,
                'csv_count': len(csv_files)
            }
            
        except Exception as e:
            logger.error(f"❌ Error downloading dataset {dataset_ref}: {e}")
            return {'success': False, 'error': str(e)}
    
    def download_all_with_progress(self) -> Dict[str, Any]:
        """Download all datasets with detailed progress tracking"""
        results = {
            'successful': [],
            'failed': [],
            'total_downloaded': 0,
            'total_size_mb': 0,
            'download_summary': {}
        }
        
        total_datasets = len(self.kaggle_datasets)
        
        for i, (name, dataset_ref) in enumerate(self.kaggle_datasets.items(), 1):
            logger.info(f"📊 Processing dataset {i}/{total_datasets}: {name}")
            
            download_result = self.download_dataset_with_validation(dataset_ref)
            
            if download_result['success']:
                results['successful'].append(name)
                results['total_downloaded'] += 1
                
                # Calculate total size
                if 'files' in download_result:
                    dataset_size = sum(f.get('size_mb', 0) for f in download_result['files'])
                    results['total_size_mb'] += dataset_size
                
                # Process the dataset
                processed_data = self._process_and_standardize_dataset(name, dataset_ref, download_result)
                results['download_summary'][name] = {
                    'status': 'success',
                    'files': download_result.get('csv_count', 0),
                    'processed': processed_data is not None
                }
            else:
                results['failed'].append(name)
                results['download_summary'][name] = {
                    'status': 'failed',
                    'error': download_result.get('error', 'Unknown error')
                }
        
        # Generate comprehensive report
        self._generate_download_report(results)
        
        logger.info(f"🎉 Download completed: {results['total_downloaded']}/{total_datasets} successful")
        return results
    
    def _process_and_standardize_dataset(self, name: str, dataset_ref: str, download_result: Dict) -> pd.DataFrame:
        """Process and standardize downloaded dataset with semiconductor-specific logic"""
        try:
            dataset_path = Path(download_result['path'])
            csv_files = list(dataset_path.glob("*.csv"))
            
            if not csv_files:
                logger.warning(f"No CSV files found in {name} dataset")
                return None
            
            # Process the main CSV file
            main_csv = max(csv_files, key=lambda x: x.stat().st_size)
            logger.info(f"Processing main file: {main_csv}")
            
            # Load dataset with error handling
            try:
                df = pd.read_csv(main_csv)
            except UnicodeDecodeError:
                # Try different encodings
                for encoding in ['latin-1', 'iso-8859-1', 'cp1252']:
                    try:
                        df = pd.read_csv(main_csv, encoding=encoding)
                        break
                    except:
                        continue
                else:
                    logger.error(f"Could not read {main_csv} with any encoding")
                    return None
            
            # Semiconductor-specific processing
            df_processed = self._apply_semiconductor_transformations(df, name)
            
            # Save processed dataset
            processed_path = self.data_dir / "processed" / "kaggle" / f"{name}_processed.csv"
            processed_path.parent.mkdir(parents=True, exist_ok=True)
            df_processed.to_csv(processed_path, index=False)
            
            # Create metadata
            metadata = {
                'name': name,
                'source': dataset_ref,
                'original_file': str(main_csv),
                'processed_file': str(processed_path),
                'original_shape': df.shape,
                'processed_shape': df_processed.shape,
                'columns_original': list(df.columns),
                'columns_processed': list(df_processed.columns),
                'dtypes': df_processed.dtypes.to_dict(),
                'missing_values': df_processed.isnull().sum().to_dict(),
                'processing_date': datetime.now().isoformat(),
                'semiconductor_relevance': self._assess_semiconductor_relevance(df_processed),
                'recommended_uses': self._suggest_use_cases(df_processed, name)
            }
            
            # Save metadata
            metadata_path = self.data_dir / "processed" / "metadata" / f"{name}_metadata.json"
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"✅ Dataset {name} processed and saved")
            return df_processed
            
        except Exception as e:
            logger.error(f"❌ Error processing dataset {name}: {e}")
            return None
    
    def _apply_semiconductor_transformations(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Apply semiconductor-specific data transformations"""
        df_processed = df.copy()
        
        # Standardize column names
        df_processed.columns = df_processed.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
        
        # Add source information
        df_processed['source_dataset'] = dataset_name
        df_processed['data_source'] = 'kaggle'
        
        # Add chip ID if not present
        if 'chip_id' not in df_processed.columns:
            df_processed['chip_id'] = [f'{dataset_name.upper()}_{i:06d}' for i in range(len(df_processed))]
        
        # Add timestamp if not present
        if 'timestamp' not in df_processed.columns and 'time' not in df_processed.columns:
            df_processed['timestamp'] = pd.date_range('2024-01-01', periods=len(df_processed), freq='1H')
        
        # Standardize test results
        result_columns = [col for col in df_processed.columns if any(keyword in col.lower() for keyword in ['pass', 'fail', 'result', 'class', 'target'])]
        
        for col in result_columns:
            if df_processed[col].dtype in ['object', 'string']:
                # Map string values
                df_processed['test_result'] = df_processed[col].str.upper().map({
                    'PASS': 'PASS', 'FAIL': 'FAIL', 'P': 'PASS', 'F': 'FAIL',
                    'GOOD': 'PASS', 'BAD': 'FAIL', 'OK': 'PASS', 'NOK': 'FAIL'
                })
            else:
                # Map numeric values (common patterns)
                df_processed['test_result'] = df_processed[col].map({
                    1: 'FAIL', 0: 'PASS', -1: 'PASS',  # SECOM pattern
                    2: 'FAIL', 1: 'PASS',  # Alternative pattern
                })
            
            if 'test_result' in df_processed.columns:
                df_processed['test_result'] = df_processed['test_result'].fillna('UNKNOWN')
                break
        
        # Identify and standardize sensor columns
        sensor_columns = [col for col in df_processed.columns if col.startswith(('sensor', 'measurement', 'reading'))]
        
        # Calculate derived metrics if possible
        if len(sensor_columns) >= 3:
            # Create a composite performance score from sensor readings
            sensor_data = df_processed[sensor_columns].select_dtypes(include=[np.number])
            if not sensor_data.empty:
                df_processed['performance_score'] = sensor_data.mean(axis=1) * 1000  # Scale to typical range
        
        return df_processed
    
    def _assess_semiconductor_relevance(self, df: pd.DataFrame) -> str:
        """Assess how relevant the dataset is to semiconductor manufacturing"""
        columns_lower = [col.lower() for col in df.columns]
        
        # Semiconductor-specific keywords
        high_relevance_keywords = [
            'wafer', 'chip', 'semiconductor', 'fab', 'yield', 'defect', 'test_result'
        ]
        
        medium_relevance_keywords = [
            'temperature', 'pressure', 'flow', 'power', 'voltage', 'current',
            'sensor', 'measurement', 'process', 'quality', 'performance'
        ]
        
        low_relevance_keywords = [
            'manufacturing', 'production', 'failure', 'maintenance', 'monitoring'
        ]
        
        high_score = sum(1 for keyword in high_relevance_keywords 
                        if any(keyword in col for col in columns_lower))
        medium_score = sum(1 for keyword in medium_relevance_keywords 
                          if any(keyword in col for col in columns_lower))
        low_score = sum(1 for keyword in low_relevance_keywords 
                       if any(keyword in col for col in columns_lower))
        
        total_score = high_score * 3 + medium_score * 2 + low_score
        
        if total_score >= 10:
            return 'high'
        elif total_score >= 5:
            return 'medium'
        else:
            return 'low'
    
    def _suggest_use_cases(self, df: pd.DataFrame, dataset_name: str) -> List[str]:
        """Suggest potential use cases based on dataset characteristics"""
        use_cases = []
        columns_lower = [col.lower() for col in df.columns]
        
        # Classification use cases
        if 'test_result' in columns_lower or any('class' in col for col in columns_lower):
            use_cases.extend(['Quality Classification', 'Defect Detection', 'Pass/Fail Prediction'])
        
        # Sensor data analysis
        sensor_cols = [col for col in columns_lower if 'sensor' in col or 'measurement' in col]
        if len(sensor_cols) > 5:
            use_cases.extend(['Anomaly Detection', 'Process Monitoring', 'Multivariate Analysis'])
        
        # Time series analysis
        if 'timestamp' in columns_lower or 'time' in columns_lower:
            use_cases.extend(['Time Series Analysis', 'Trend Detection', 'Predictive Maintenance'])
        
        # Performance optimization
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 10:
            use_cases.extend(['Feature Engineering', 'Dimensionality Reduction', 'Parameter Optimization'])
        
        # Manufacturing specific
        if any(keyword in dataset_name.lower() for keyword in ['manufacturing', 'process', 'quality']):
            use_cases.extend(['Process Optimization', 'Quality Control', 'Yield Improvement'])
        
        return list(set(use_cases))  # Remove duplicates
    
    def integrate_with_realtime_stream(self, dataset_name: str = None) -> bool:
        """Integrate processed Kaggle data with real-time streaming"""
        try:
            # Load processed datasets
            if dataset_name:
                datasets_to_integrate = [dataset_name]
            else:
                processed_dir = self.data_dir / "processed" / "kaggle"
                datasets_to_integrate = [f.stem.replace('_processed', '') for f in processed_dir.glob("*_processed.csv")]
            
            logger.info(f"🔄 Integrating {len(datasets_to_integrate)} datasets with real-time stream")
            
            # Create enhanced chip simulator that uses Kaggle data patterns
            enhanced_simulator = self._create_enhanced_simulator(datasets_to_integrate)
            
            # Replace the simulator in the real-time streamer
            self.real_time_streamer.simulator = enhanced_simulator
            
            logger.info("✅ Real-time streamer enhanced with Kaggle data patterns")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error integrating with real-time stream: {e}")
            return False
    
    def _create_enhanced_simulator(self, dataset_names: List[str]) -> ChipDataSimulator:
        """Create enhanced simulator using patterns from Kaggle datasets"""
        class EnhancedChipDataSimulator(ChipDataSimulator):
            def __init__(self, kaggle_patterns):
                super().__init__()
                self.kaggle_patterns = kaggle_patterns
            
            def generate_chip_data(self) -> Dict:
                # Get base data from parent
                base_data = super().generate_chip_data()
                
                # Enhance with Kaggle patterns if available
                if self.kaggle_patterns:
                    # Add more realistic variations based on actual data distributions
                    pattern = np.random.choice(list(self.kaggle_patterns.keys()))
                    pattern_data = self.kaggle_patterns[pattern]
                    
                    # Adjust values based on learned patterns
                    if 'temperature_range' in pattern_data:
                        temp_min, temp_max = pattern_data['temperature_range']
                        base_data['temperature_celsius'] = np.random.uniform(temp_min, temp_max)
                    
                    if 'failure_rate' in pattern_data:
                        failure_prob = pattern_data['failure_rate']
                        base_data['test_result'] = 'FAIL' if np.random.random() < failure_prob else 'PASS'
                
                return base_data
        
        # Analyze patterns from processed datasets
        patterns = {}
        for dataset_name in dataset_names:
            try:
                processed_file = self.data_dir / "processed" / "kaggle" / f"{dataset_name}_processed.csv"
                if processed_file.exists():
                    df = pd.read_csv(processed_file)
                    
                    pattern = {}
                    if 'temperature_celsius' in df.columns:
                        pattern['temperature_range'] = (df['temperature_celsius'].min(), df['temperature_celsius'].max())
                    
                    if 'test_result' in df.columns:
                        pattern['failure_rate'] = (df['test_result'] == 'FAIL').mean()
                    
                    patterns[dataset_name] = pattern
            except Exception as e:
                logger.warning(f"Could not analyze patterns from {dataset_name}: {e}")
        
        return EnhancedChipDataSimulator(patterns)
    
    def _generate_download_report(self, results: Dict[str, Any]):
        """Generate comprehensive download and processing report"""
        report_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Semiconductor Datasets Integration Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); 
                          color: white; padding: 20px; border-radius: 10px; }}
                .summary {{ background: #f8fafc; padding: 20px; border-radius: 10px; margin: 20px 0; }}
                .dataset {{ border: 1px solid #e5e7eb; margin: 10px 0; padding: 15px; border-radius: 5px; }}
                .success {{ border-left: 5px solid #10b981; }}
                .failed {{ border-left: 5px solid #ef4444; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔬 Semiconductor Datasets Integration Report</h1>
                <p>Comprehensive download and processing results</p>
            </div>
            
            <div class="summary">
                <h2>📊 Summary</h2>
                <p><strong>Total Datasets:</strong> {len(self.kaggle_datasets)}</p>
                <p><strong>Successfully Downloaded:</strong> {results['total_downloaded']}</p>
                <p><strong>Failed Downloads:</strong> {len(results['failed'])}</p>
                <p><strong>Success Rate:</strong> {results['total_downloaded']/len(self.kaggle_datasets)*100:.1f}%</p>
                <p><strong>Total Data Size:</strong> {results['total_size_mb']:.1f} MB</p>
            </div>
            
            <h2>📋 Dataset Details</h2>
        """
        
        for dataset_name, summary in results['download_summary'].items():
            status_class = 'success' if summary['status'] == 'success' else 'failed'
            status_icon = '✅' if summary['status'] == 'success' else '❌'
            
            report_html += f"""
            <div class="dataset {status_class}">
                <h3>{status_icon} {dataset_name}</h3>
                <p><strong>Status:</strong> {summary['status']}</p>
            """
            
            if summary['status'] == 'success':
                report_html += f"""
                <p><strong>Files:</strong> {summary.get('files', 0)} CSV files</p>
                <p><strong>Processed:</strong> {'Yes' if summary.get('processed', False) else 'No'}</p>
                """
            else:
                report_html += f"<p><strong>Error:</strong> {summary.get('error', 'Unknown')}</p>"
            
            report_html += "</div>"
        
        report_html += """
            </body>
            </html>
        """
        
        # Save report
        report_path = self.data_dir / "processed" / "integration_report.html"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_html)
        
        logger.info(f"📄 Integration report saved: {report_path}")

def main():
    """Main function with enhanced features"""
    print("🚀 Enhanced Semiconductor Dataset Integration")
    print("=" * 60)
    
    # Initialize enhanced manager
    manager = EnhancedSemiconductorDatasetManager()
    
    # Search and discover datasets
    print("🔍 Searching for semiconductor datasets...")
    search_results = manager.search_and_discover_datasets()
    
    if not search_results.empty:
        print(f"Found {len(search_results)} potential datasets")
        print("\nTop 5 datasets by usability rating:")
        top_datasets = search_results.head(5)[['title', 'ref', 'usability_rating', 'download_count']]
        print(top_datasets.to_string(index=False))
    
    # Download all datasets with progress tracking
    print("\n📥 Downloading and processing datasets...")
    download_results = manager.download_all_with_progress()
    
    print(f"\n✅ Download Summary:")
    print(f"   Successfully downloaded: {download_results['total_downloaded']}")
    print(f"   Failed downloads: {len(download_results['failed'])}")
    print(f"   Total data size: {download_results['total_size_mb']:.1f} MB")
    
    # Integrate with real-time streaming
    print("\n🔄 Integrating with real-time streaming...")
    integration_success = manager.integrate_with_realtime_stream()
    
    if integration_success:
        print("✅ Real-time streaming enhanced with Kaggle data patterns")
    else:
        print("⚠️ Real-time integration had issues, but core functionality available")
    
    print("\n🎉 Enhanced integration completed!")
    print("📁 Check these locations:")
    print("   📊 Raw data: data/raw/kaggle/")
    print("   🔧 Processed data: data/processed/kaggle/")
    print("   📄 Integration report: data/processed/integration_report.html")
    print("   🎯 Metadata: data/processed/metadata/")
    
    return download_results

if __name__ == "__main__":
    main()