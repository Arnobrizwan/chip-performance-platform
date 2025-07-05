# src/analysis/anomaly/outlier_detector.py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AdvancedAnomalyDetector:
    """Advanced anomaly detection for semiconductor chip performance"""
    
    def __init__(self, contamination_rate: float = 0.1):
        """
        Initialize anomaly detector
        
        Args:
            contamination_rate: Expected proportion of anomalies (0.0 to 0.5)
        """
        self.contamination_rate = contamination_rate
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=contamination_rate,
            random_state=42,
            n_estimators=100
        )
        self.fitted = False
    
    def detect_statistical_anomalies(self, df: pd.DataFrame, 
                                   features: List[str] = None) -> Dict:
        """Detect anomalies using statistical methods (Z-score, IQR)"""
        
        if features is None:
            features = ['performance_score', 'temperature_celsius', 
                       'power_consumption_watts', 'clock_speed_ghz']
        
        # Filter available features
        available_features = [f for f in features if f in df.columns]
        
        anomalies = {}
        
        for feature in available_features:
            feature_anomalies = []
            
            # Z-score method (outliers beyond 3 standard deviations)
            z_scores = np.abs(stats.zscore(df[feature].dropna()))
            z_outliers = df[z_scores > 3].index.tolist()
            
            # IQR method
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = df[
                (df[feature] < lower_bound) | (df[feature] > upper_bound)
            ].index.tolist()
            
            # Modified Z-score (more robust)
            median = df[feature].median()
            mad = np.median(np.abs(df[feature] - median))
            modified_z_scores = 0.6745 * (df[feature] - median) / mad
            modified_z_outliers = df[np.abs(modified_z_scores) > 3.5].index.tolist()
            
            anomalies[feature] = {
                'z_score_outliers': z_outliers,
                'iqr_outliers': iqr_outliers,
                'modified_z_outliers': modified_z_outliers,
                'combined_outliers': list(set(z_outliers + iqr_outliers + modified_z_outliers))
            }
        
        return anomalies
    
    def detect_ml_anomalies(self, df: pd.DataFrame, 
                           features: List[str] = None) -> Dict:
        """Detect anomalies using machine learning methods"""
        
        if features is None:
            features = ['performance_score', 'temperature_celsius', 
                       'power_consumption_watts', 'clock_speed_ghz', 
                       'voltage_v', 'efficiency_score']
        
        # Filter available features
        available_features = [f for f in features if f in df.columns]
        
        if len(available_features) < 2:
            return {'error': 'Insufficient features for ML detection'}
        
        # Prepare data
        X = df[available_features].dropna()
        
        if len(X) < 10:
            return {'error': 'Insufficient data for ML detection'}
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Isolation Forest
        iso_forest_predictions = self.isolation_forest.fit_predict(X_scaled)
        iso_forest_anomalies = X[iso_forest_predictions == -1].index.tolist()
        
        # DBSCAN clustering (outliers are noise points)
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X_scaled)
        dbscan_anomalies = X[dbscan_labels == -1].index.tolist()
        
        # Local Outlier Factor would go here if available
        
        self.fitted = True
        
        return {
            'isolation_forest_anomalies': iso_forest_anomalies,
            'dbscan_anomalies': dbscan_anomalies,
            'combined_ml_anomalies': list(set(iso_forest_anomalies + dbscan_anomalies)),
            'anomaly_scores': self.isolation_forest.decision_function(X_scaled).tolist()
        }
    
    def detect_performance_anomalies(self, df: pd.DataFrame) -> Dict:
        """Detect performance-specific anomalies"""
        
        anomalies = {
            'thermal_anomalies': [],
            'power_anomalies': [],
            'performance_drops': [],
            'efficiency_anomalies': [],
            'failure_patterns': []
        }
        
        # Thermal anomalies
        if 'temperature_celsius' in df.columns:
            thermal_threshold = df['temperature_celsius'].quantile(0.95)
            anomalies['thermal_anomalies'] = df[
                df['temperature_celsius'] > thermal_threshold
            ].index.tolist()
        
        # Power consumption anomalies
        if 'power_consumption_watts' in df.columns:
            power_mean = df['power_consumption_watts'].mean()
            power_std = df['power_consumption_watts'].std()
            power_threshold = power_mean + 2 * power_std
            
            anomalies['power_anomalies'] = df[
                df['power_consumption_watts'] > power_threshold
            ].index.tolist()
        
        # Performance drops
        if 'performance_score' in df.columns:
            performance_threshold = df['performance_score'].quantile(0.1)
            anomalies['performance_drops'] = df[
                df['performance_score'] < performance_threshold
            ].index.tolist()
        
        # Efficiency anomalies
        if 'efficiency_score' in df.columns:
            efficiency_threshold = df['efficiency_score'].quantile(0.1)
            anomalies['efficiency_anomalies'] = df[
                df['efficiency_score'] < efficiency_threshold
            ].index.tolist()
        
        # Test failure patterns
        if 'test_result' in df.columns:
            anomalies['failure_patterns'] = df[
                df['test_result'] == 'FAIL'
            ].index.tolist()
        
        return anomalies
    
    def detect_time_series_anomalies(self, df: pd.DataFrame, 
                                   time_column: str = 'timestamp') -> Dict:
        """Detect anomalies in time series data"""
        
        if time_column not in df.columns:
            return {'error': f'Time column {time_column} not found'}
        
        # Ensure datetime
        df[time_column] = pd.to_datetime(df[time_column])
        df_sorted = df.sort_values(time_column)
        
        anomalies = {
            'sudden_changes': [],
            'trend_anomalies': [],
            'seasonal_anomalies': []
        }
        
        # Analyze performance score over time
        if 'performance_score' in df.columns:
            # Detect sudden changes (large differences between consecutive points)
            performance_diff = df_sorted['performance_score'].diff().abs()
            sudden_change_threshold = performance_diff.quantile(0.95)
            
            sudden_changes = df_sorted[
                performance_diff > sudden_change_threshold
            ].index.tolist()
            
            anomalies['sudden_changes'] = sudden_changes
        
        # Temperature trend anomalies
        if 'temperature_celsius' in df.columns:
            # Moving average to detect trend anomalies
            window_size = min(50, len(df_sorted) // 10)
            if window_size > 1:
                moving_avg = df_sorted['temperature_celsius'].rolling(
                    window=window_size, center=True
                ).mean()
                
                temp_deviation = np.abs(df_sorted['temperature_celsius'] - moving_avg)
                trend_threshold = temp_deviation.quantile(0.95)
                
                trend_anomalies = df_sorted[
                    temp_deviation > trend_threshold
                ].index.tolist()
                
                anomalies['trend_anomalies'] = trend_anomalies
        
        return anomalies
    
    def analyze_anomaly_patterns(self, df: pd.DataFrame, 
                                anomaly_indices: List[int]) -> Dict:
        """Analyze patterns in detected anomalies"""
        
        if not anomaly_indices:
            return {'message': 'No anomalies to analyze'}
        
        anomaly_data = df.loc[anomaly_indices]
        
        patterns = {
            'anomaly_count': len(anomaly_indices),
            'anomaly_rate': len(anomaly_indices) / len(df) * 100,
            'chip_type_distribution': {},
            'manufacturer_distribution': {},
            'common_characteristics': {},
            'severity_analysis': {}
        }
        
        # Chip type distribution
        if 'chip_type' in df.columns:
            patterns['chip_type_distribution'] = anomaly_data['chip_type'].value_counts().to_dict()
        
        # Manufacturer distribution
        if 'manufacturer' in df.columns:
            patterns['manufacturer_distribution'] = anomaly_data['manufacturer'].value_counts().to_dict()
        
        # Common characteristics analysis
        numeric_columns = anomaly_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in ['performance_score', 'temperature_celsius', 'power_consumption_watts']:
                patterns['common_characteristics'][col] = {
                    'mean': float(anomaly_data[col].mean()),
                    'std': float(anomaly_data[col].std()),
                    'min': float(anomaly_data[col].min()),
                    'max': float(anomaly_data[col].max())
                }
        
        # Severity analysis
        if 'performance_score' in anomaly_data.columns:
            performance_mean = df['performance_score'].mean()
            anomaly_performance_mean = anomaly_data['performance_score'].mean()
            
            patterns['severity_analysis'] = {
                'performance_impact': float(performance_mean - anomaly_performance_mean),
                'severe_anomalies': int((anomaly_data['performance_score'] < df['performance_score'].quantile(0.05)).sum()),
                'critical_failures': int((anomaly_data['test_result'] == 'FAIL').sum()) if 'test_result' in anomaly_data.columns else 0
            }
        
        return patterns
    
    def generate_anomaly_report(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive anomaly detection report"""
        
        print("🔍 Running comprehensive anomaly detection...")
        
        # Run all detection methods
        statistical_anomalies = self.detect_statistical_anomalies(df)
        ml_anomalies = self.detect_ml_anomalies(df)
        performance_anomalies = self.detect_performance_anomalies(df)
        time_series_anomalies = self.detect_time_series_anomalies(df)
        
        # Combine all anomalies
        all_anomaly_indices = set()
        
        # Add statistical anomalies
        for feature, methods in statistical_anomalies.items():
            all_anomaly_indices.update(methods['combined_outliers'])
        
        # Add ML anomalies
        if 'combined_ml_anomalies' in ml_anomalies:
            all_anomaly_indices.update(ml_anomalies['combined_ml_anomalies'])
        
        # Add performance anomalies
        for anomaly_type, indices in performance_anomalies.items():
            all_anomaly_indices.update(indices)
        
        # Add time series anomalies
        for anomaly_type, indices in time_series_anomalies.items():
            if isinstance(indices, list):
                all_anomaly_indices.update(indices)
        
        all_anomaly_indices = list(all_anomaly_indices)
        
        # Analyze patterns
        pattern_analysis = self.analyze_anomaly_patterns(df, all_anomaly_indices)
        
        # Generate recommendations
        recommendations = self._generate_anomaly_recommendations(
            df, all_anomaly_indices, pattern_analysis
        )
        
        return {
            'summary': {
                'total_chips': len(df),
                'total_anomalies': len(all_anomaly_indices),
                'anomaly_rate': len(all_anomaly_indices) / len(df) * 100,
                'detection_methods_used': ['Statistical', 'Machine Learning', 'Performance-based', 'Time Series']
            },
            'detailed_results': {
                'statistical_anomalies': statistical_anomalies,
                'ml_anomalies': ml_anomalies,
                'performance_anomalies': performance_anomalies,
                'time_series_anomalies': time_series_anomalies
            },
            'pattern_analysis': pattern_analysis,
            'anomaly_indices': all_anomaly_indices,
            'recommendations': recommendations
        }
    
    def _generate_anomaly_recommendations(self, df: pd.DataFrame, 
                                        anomaly_indices: List[int], 
                                        patterns: Dict) -> List[str]:
        """Generate actionable recommendations based on anomaly analysis"""
        
        recommendations = []
        
        anomaly_rate = len(anomaly_indices) / len(df) * 100
        
        # Overall anomaly rate recommendations
        if anomaly_rate > 15:
            recommendations.append(f"🔴 High anomaly rate ({anomaly_rate:.1f}%) - investigate manufacturing process")
        elif anomaly_rate > 10:
            recommendations.append(f"⚠️ Elevated anomaly rate ({anomaly_rate:.1f}%) - monitor closely")
        elif anomaly_rate < 2:
            recommendations.append("✅ Low anomaly rate - manufacturing process stable")
        
        # Chip type specific recommendations
        if 'chip_type_distribution' in patterns:
            for chip_type, count in patterns['chip_type_distribution'].items():
                type_rate = count / patterns['anomaly_count'] * 100
                if type_rate > 50:
                    recommendations.append(f"🔧 {chip_type} chips show high anomaly rate - review design")
        
        # Manufacturer specific recommendations
        if 'manufacturer_distribution' in patterns:
            for manufacturer, count in patterns['manufacturer_distribution'].items():
                mfg_rate = count / patterns['anomaly_count'] * 100
                if mfg_rate > 40:
                    recommendations.append(f"🏭 {manufacturer} shows elevated anomaly rate - review processes")
        
        # Performance specific recommendations
        if 'severity_analysis' in patterns:
            if patterns['severity_analysis'].get('critical_failures', 0) > 0:
                recommendations.append("🚨 Critical failures detected - immediate investigation required")
            
            if patterns['severity_analysis'].get('performance_impact', 0) > 1000:
                recommendations.append("📉 Significant performance impact from anomalies - optimize design")
        
        # Temperature recommendations
        if 'common_characteristics' in patterns and 'temperature_celsius' in patterns['common_characteristics']:
            avg_temp = patterns['common_characteristics']['temperature_celsius']['mean']
            if avg_temp > 85:
                recommendations.append(f"🌡️ High temperature anomalies (avg: {avg_temp:.1f}°C) - improve cooling")
        
        if not recommendations:
            recommendations.append("✅ Anomaly patterns within expected ranges")
        
        return recommendations

# Example usage
if __name__ == "__main__":
    # Test with real data
    try:
        df = pd.read_csv('data/raw/chip_test_data/secom_real_data.csv')
        
        detector = AdvancedAnomalyDetector(contamination_rate=0.05)
        report = detector.generate_anomaly_report(df)
        
        print("\n🎯 Anomaly Detection Results:")
        print("=" * 50)
        print(f"Total chips: {report['summary']['total_chips']:,}")
        print(f"Anomalies detected: {report['summary']['total_anomalies']:,}")
        print(f"Anomaly rate: {report['summary']['anomaly_rate']:.2f}%")
        
        print(f"\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"   {rec}")
            
    except FileNotFoundError:
        print("❌ No data file found. Please run data processing first.")