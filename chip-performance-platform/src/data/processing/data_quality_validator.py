# src/data/processing/data_quality_validator.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataQualityValidator:
    """Comprehensive data quality validation for semiconductor datasets"""
    
    def __init__(self, data_dir: str = "data/raw/kaggle"):
        self.data_dir = Path(data_dir)
        self.quality_report = {}
        
    def validate_all_datasets(self) -> Dict[str, Any]:
        """Validate all datasets in the data directory"""
        logger.info("🔍 Starting comprehensive data quality validation...")
        
        datasets_found = []
        
        # Find all CSV files
        for csv_file in self.data_dir.rglob("*.csv"):
            try:
                dataset_name = csv_file.parent.name
                logger.info(f"Validating {dataset_name}: {csv_file.name}")
                
                # Load dataset
                df = pd.read_csv(csv_file)
                
                # Validate dataset
                validation_result = self._validate_dataset(df, dataset_name, csv_file.name)
                
                datasets_found.append({
                    'name': dataset_name,
                    'file': csv_file.name,
                    'path': str(csv_file),
                    'validation': validation_result
                })
                
            except Exception as e:
                logger.error(f"Error validating {csv_file}: {e}")
                datasets_found.append({
                    'name': csv_file.parent.name,
                    'file': csv_file.name,
                    'path': str(csv_file),
                    'validation': {'error': str(e), 'valid': False}
                })
        
        # Generate comprehensive report
        self.quality_report = {
            'total_datasets': len(datasets_found),
            'valid_datasets': sum(1 for d in datasets_found if d['validation'].get('valid', False)),
            'datasets': datasets_found,
            'summary': self._generate_summary(datasets_found)
        }
        
        return self.quality_report
    
    def _validate_dataset(self, df: pd.DataFrame, dataset_name: str, file_name: str) -> Dict[str, Any]:
        """Validate a single dataset"""
        validation = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'statistics': {},
            'recommendations': []
        }
        
        try:
            # Basic statistics
            validation['statistics'] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'missing_values': df.isnull().sum().to_dict(),
                'duplicate_rows': df.duplicated().sum(),
                'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': len(df.select_dtypes(include=['object']).columns)
            }
            
            # Check for critical issues
            issues = self._check_critical_issues(df)
            validation['issues'].extend(issues)
            
            # Check for warnings
            warnings = self._check_warnings(df)
            validation['warnings'].extend(warnings)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(df, dataset_name)
            validation['recommendations'].extend(recommendations)
            
            # Determine if dataset is valid
            validation['valid'] = len(validation['issues']) == 0
            
            # Semiconductor-specific validation
            sem_validation = self._semiconductor_specific_validation(df, dataset_name)
            validation.update(sem_validation)
            
        except Exception as e:
            validation['valid'] = False
            validation['issues'].append(f"Validation error: {str(e)}")
        
        return validation
    
    def _check_critical_issues(self, df: pd.DataFrame) -> List[str]:
        """Check for critical data quality issues"""
        issues = []
        
        # Empty dataset
        if df.empty:
            issues.append("Dataset is empty")
        
        # All columns are null
        if df.isnull().all().any():
            null_cols = df.columns[df.isnull().all()].tolist()
            issues.append(f"Columns with all null values: {null_cols}")
        
        # Single row/column
        if df.shape[0] <= 1:
            issues.append("Dataset has only one or zero rows")
        
        if df.shape[1] <= 1:
            issues.append("Dataset has only one or zero columns")
        
        # Completely duplicated dataset
        if df.duplicated().all():
            issues.append("All rows are duplicates")
        
        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if np.isinf(df[col]).any():
                issues.append(f"Column '{col}' contains infinite values")
        
        return issues
    
    def _check_warnings(self, df: pd.DataFrame) -> List[str]:
        """Check for data quality warnings"""
        warnings = []
        
        # High percentage of missing values
        missing_pct = df.isnull().sum() / len(df) * 100
        high_missing = missing_pct[missing_pct > 50].index.tolist()
        if high_missing:
            warnings.append(f"Columns with >50% missing values: {high_missing}")
        
        # High number of duplicates
        dup_pct = df.duplicated().sum() / len(df) * 100
        if dup_pct > 10:
            warnings.append(f"High percentage of duplicate rows: {dup_pct:.1f}%")
        
        # Columns with single unique value
        single_value_cols = []
        for col in df.columns:
            if df[col].nunique() <= 1:
                single_value_cols.append(col)
        if single_value_cols:
            warnings.append(f"Columns with single unique value: {single_value_cols}")
        
        # Very high cardinality categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.9:
                warnings.append(f"Column '{col}' has very high cardinality ({unique_ratio:.1%})")
        
        # Potential encoding issues
        for col in categorical_cols:
            if df[col].astype(str).str.contains('\\x').any():
                warnings.append(f"Column '{col}' may have encoding issues")
        
        return warnings
    
    def _semiconductor_specific_validation(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Semiconductor industry-specific validation"""
        sem_validation = {
            'semiconductor_relevance': 'unknown',
            'data_type': 'unknown',
            'recommended_use_cases': []
        }
        
        columns_lower = [col.lower() for col in df.columns]
        
        # Identify semiconductor relevance
        semiconductor_keywords = [
            'wafer', 'chip', 'semiconductor', 'defect', 'yield', 'fab', 'process',
            'temperature', 'pressure', 'flow', 'power', 'voltage', 'current',
            'resistance', 'capacitance', 'thickness', 'contamination', 'particle'
        ]
        
        keyword_matches = sum(1 for keyword in semiconductor_keywords 
                             if any(keyword in col for col in columns_lower))
        
        if keyword_matches >= 3:
            sem_validation['semiconductor_relevance'] = 'high'
        elif keyword_matches >= 1:
            sem_validation['semiconductor_relevance'] = 'medium'
        else:
            sem_validation['semiconductor_relevance'] = 'low'
        
        # Identify data type
        if any('pass' in col or 'fail' in col or 'defect' in col for col in columns_lower):
            sem_validation['data_type'] = 'classification'
            sem_validation['recommended_use_cases'].append('Quality classification')
            sem_validation['recommended_use_cases'].append('Defect detection')
        
        if any('temperature' in col or 'pressure' in col or 'flow' in col for col in columns_lower):
            sem_validation['data_type'] = 'sensor_data'
            sem_validation['recommended_use_cases'].append('Process monitoring')
            sem_validation['recommended_use_cases'].append('Anomaly detection')
        
        if 'time' in columns_lower or 'timestamp' in columns_lower:
            sem_validation['recommended_use_cases'].append('Time series analysis')
            sem_validation['recommended_use_cases'].append('Trend analysis')
        
        # Check for sensor-like data patterns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 10:
            sem_validation['recommended_use_cases'].append('Multivariate analysis')
            sem_validation['recommended_use_cases'].append('Dimensionality reduction')
        
        return sem_validation
    
    def _generate_recommendations(self, df: pd.DataFrame, dataset_name: str) -> List[str]:
        """Generate recommendations for data improvement"""
        recommendations = []
        
        # Missing value handling
        missing_pct = df.isnull().sum() / len(df) * 100
        if missing_pct.max() > 0:
            recommendations.append("Consider imputation strategies for missing values")
        
        # Duplicate handling
        if df.duplicated().any():
            recommendations.append("Remove or investigate duplicate rows")
        
        # Feature engineering suggestions
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 5:
            recommendations.append("Consider feature selection or dimensionality reduction")
        
        # Scaling suggestions
        if len(numeric_cols) > 0:
            ranges = df[numeric_cols].max() - df[numeric_cols].min()
            if ranges.max() / ranges.min() > 100:
                recommendations.append("Consider feature scaling due to different value ranges")
        
        # Categorical encoding
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            recommendations.append("Consider encoding categorical variables for ML models")
        
        return recommendations
    
    def _generate_summary(self, datasets: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics across all datasets"""
        valid_datasets = [d for d in datasets if d['validation'].get('valid', False)]
        
        if not valid_datasets:
            return {'error': 'No valid datasets found'}
        
        total_rows = sum(d['validation']['statistics']['shape'][0] for d in valid_datasets)
        total_cols = sum(d['validation']['statistics']['shape'][1] for d in valid_datasets)
        
        semiconductor_relevant = sum(1 for d in valid_datasets 
                                   if d['validation'].get('semiconductor_relevance') in ['high', 'medium'])
        
        return {
            'total_rows': total_rows,
            'total_columns': total_cols,
            'avg_rows_per_dataset': total_rows / len(valid_datasets),
            'avg_cols_per_dataset': total_cols / len(valid_datasets),
            'semiconductor_relevant_datasets': semiconductor_relevant,
            'data_types_found': list(set(d['validation'].get('data_type', 'unknown') 
                                       for d in valid_datasets)),
            'common_use_cases': self._get_common_use_cases(valid_datasets)
        }
    
    def _get_common_use_cases(self, datasets: List[Dict]) -> List[str]:
        """Get most common recommended use cases"""
        use_case_counts = {}
        
        for dataset in datasets:
            use_cases = dataset['validation'].get('recommended_use_cases', [])
            for use_case in use_cases:
                use_case_counts[use_case] = use_case_counts.get(use_case, 0) + 1
        
        # Sort by frequency
        sorted_use_cases = sorted(use_case_counts.items(), key=lambda x: x[1], reverse=True)
        return [use_case for use_case, count in sorted_use_cases[:5]]
    
    def generate_quality_report(self, output_path: str = "data/processed/data_quality_report.html"):
        """Generate an HTML quality report"""
        if not self.quality_report:
            logger.error("No quality report available. Run validate_all_datasets() first.")
            return
        
        html_content = self._create_html_report()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Quality report generated: {output_path}")
    
    def _create_html_report(self) -> str:
        """Create HTML content for the quality report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Semiconductor Data Quality Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); 
                          color: white; padding: 20px; border-radius: 10px; }}
                .summary {{ background: #f8fafc; padding: 20px; border-radius: 10px; margin: 20px 0; }}
                .dataset {{ border: 1px solid #e5e7eb; margin: 20px 0; padding: 15px; border-radius: 5px; }}
                .valid {{ border-left: 5px solid #10b981; }}
                .invalid {{ border-left: 5px solid #ef4444; }}
                .statistics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
                .stat {{ background: #f1f5f9; padding: 10px; border-radius: 5px; }}
                .issues {{ color: #dc2626; }}
                .warnings {{ color: #f59e0b; }}
                .recommendations {{ color: #059669; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔬 Semiconductor Data Quality Report</h1>
                <p>Comprehensive validation of downloaded semiconductor datasets</p>
            </div>
            
            <div class="summary">
                <h2>📊 Summary</h2>
                <div class="statistics">
                    <div class="stat">
                        <strong>Total Datasets:</strong> {self.quality_report['total_datasets']}
                    </div>
                    <div class="stat">
                        <strong>Valid Datasets:</strong> {self.quality_report['valid_datasets']}
                    </div>
                    <div class="stat">
                        <strong>Success Rate:</strong> {self.quality_report['valid_datasets']/self.quality_report['total_datasets']*100:.1f}%
                    </div>
        """
        
        if 'summary' in self.quality_report:
            summary = self.quality_report['summary']
            html += f"""
                    <div class="stat">
                        <strong>Total Rows:</strong> {summary.get('total_rows', 'N/A'):,}
                    </div>
                    <div class="stat">
                        <strong>Semiconductor Relevant:</strong> {summary.get('semiconductor_relevant_datasets', 'N/A')}
                    </div>
            """
        
        html += """
                </div>
            </div>
            
            <h2>📋 Dataset Details</h2>
        """
        
        for dataset in self.quality_report['datasets']:
            validation = dataset['validation']
            is_valid = validation.get('valid', False)
            status_class = 'valid' if is_valid else 'invalid'
            status_icon = '✅' if is_valid else '❌'
            
            html += f"""
            <div class="dataset {status_class}">
                <h3>{status_icon} {dataset['name']} - {dataset['file']}</h3>
            """
            
            if 'statistics' in validation:
                stats = validation['statistics']
                html += f"""
                <div class="statistics">
                    <div class="stat"><strong>Shape:</strong> {stats['shape']}</div>
                    <div class="stat"><strong>Columns:</strong> {stats['numeric_columns']} numeric, {stats['categorical_columns']} categorical</div>
                    <div class="stat"><strong>Missing Values:</strong> {sum(stats['missing_values'].values())}</div>
                    <div class="stat"><strong>Duplicates:</strong> {stats['duplicate_rows']}</div>
                    <div class="stat"><strong>Size:</strong> {stats['memory_usage_mb']:.2f} MB</div>
                </div>
                """
            
            if validation.get('issues'):
                html += f"""
                <div class="issues">
                    <strong>🚨 Issues:</strong>
                    <ul>{''.join(f'<li>{issue}</li>' for issue in validation['issues'])}</ul>
                </div>
                """
            
            if validation.get('warnings'):
                html += f"""
                <div class="warnings">
                    <strong>⚠️ Warnings:</strong>
                    <ul>{''.join(f'<li>{warning}</li>' for warning in validation['warnings'])}</ul>
                </div>
                """
            
            if validation.get('recommendations'):
                html += f"""
                <div class="recommendations">
                    <strong>💡 Recommendations:</strong>
                    <ul>{''.join(f'<li>{rec}</li>' for rec in validation['recommendations'])}</ul>
                </div>
                """
            
            # Semiconductor-specific info
            sem_relevance = validation.get('semiconductor_relevance', 'unknown')
            data_type = validation.get('data_type', 'unknown')
            use_cases = validation.get('recommended_use_cases', [])
            
            if sem_relevance != 'unknown':
                html += f"""
                <div class="stat">
                    <strong>🔬 Semiconductor Relevance:</strong> {sem_relevance.title()}<br>
                    <strong>📊 Data Type:</strong> {data_type.replace('_', ' ').title()}<br>
                    <strong>🎯 Use Cases:</strong> {', '.join(use_cases) if use_cases else 'None identified'}
                </div>
                """
            
            html += "</div>"
        
        html += """
            </body>
            </html>
        """
        
        return html

def main():
    """Main function to validate all semiconductor datasets"""
    print("🔍 Starting Data Quality Validation")
    print("=" * 50)
    
    validator = DataQualityValidator()
    
    # Run validation
    report = validator.validate_all_datasets()
    
    # Generate HTML report
    validator.generate_quality_report()
    
    # Print summary
    print(f"\n📊 Validation Summary:")
    print(f"Total datasets: {report['total_datasets']}")
    print(f"Valid datasets: {report['valid_datasets']}")
    print(f"Success rate: {report['valid_datasets']/report['total_datasets']*100:.1f}%")
    
    if 'summary' in report:
        summary = report['summary']
        print(f"Total rows: {summary.get('total_rows', 'N/A'):,}")
        print(f"Semiconductor relevant: {summary.get('semiconductor_relevant_datasets', 'N/A')}")
        print(f"Common use cases: {', '.join(summary.get('common_use_cases', []))}")
    
    print(f"\n📄 Detailed report: data/processed/data_quality_report.html")
    
    return report

if __name__ == "__main__":
    main()