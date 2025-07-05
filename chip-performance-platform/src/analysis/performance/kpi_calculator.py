# src/analysis/performance/kpi_calculator.py - FIXED VERSION
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

class AdvancedKPICalculator:
    """Advanced KPI calculator for semiconductor chip performance analysis"""
    
    def __init__(self):
        # Performance weights for overall scoring
        self.performance_weights = {
            'speed': 0.25,
            'efficiency': 0.25,
            'thermal': 0.25,
            'reliability': 0.25
        }
        
        # Industry benchmarks
        self.benchmarks = {
            'cpu': {'min_performance': 8000, 'max_temp': 85, 'min_efficiency': 50},
            'gpu': {'min_performance': 12000, 'max_temp': 90, 'min_efficiency': 45},
            'asic': {'min_performance': 15000, 'max_temp': 80, 'min_efficiency': 60}
        }
    
    def calculate_overall_performance_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate comprehensive performance score (0-100)"""
        
        # Individual component scores
        speed_score = self._calculate_speed_score(df)
        efficiency_score = self._calculate_efficiency_score(df)
        thermal_score = self._calculate_thermal_score(df)
        reliability_score = self._calculate_reliability_score(df)
        
        # Weighted overall score
        overall_score = (
            speed_score * self.performance_weights['speed'] +
            efficiency_score * self.performance_weights['efficiency'] +
            thermal_score * self.performance_weights['thermal'] +
            reliability_score * self.performance_weights['reliability']
        )
        
        return overall_score.clip(0, 100)
    
    def _calculate_speed_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate speed performance score based on clock speed and performance metrics"""
        
        # Normalize clock speed (0-100)
        if 'clock_speed_ghz' in df.columns:
            max_clock = df['clock_speed_ghz'].quantile(0.95)
            min_clock = df['clock_speed_ghz'].quantile(0.05)
            if max_clock > min_clock:
                clock_score = ((df['clock_speed_ghz'] - min_clock) / (max_clock - min_clock) * 100).clip(0, 100)
            else:
                clock_score = pd.Series([50] * len(df))
        else:
            clock_score = pd.Series([50] * len(df))
        
        # Normalize performance score
        if 'performance_score' in df.columns:
            max_perf = df['performance_score'].quantile(0.95)
            min_perf = df['performance_score'].quantile(0.05)
            if max_perf > min_perf:
                perf_score = ((df['performance_score'] - min_perf) / (max_perf - min_perf) * 100).clip(0, 100)
            else:
                perf_score = pd.Series([50] * len(df))
        else:
            perf_score = pd.Series([50] * len(df))
        
        # Combined speed score
        return (clock_score * 0.4 + perf_score * 0.6)
    
    def _calculate_efficiency_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate power efficiency score"""
        
        if all(col in df.columns for col in ['performance_score', 'power_consumption_watts']):
            # Performance per watt
            # Avoid division by zero
            power_nonzero = df['power_consumption_watts'].replace(0, 1)
            efficiency = df['performance_score'] / power_nonzero
            max_eff = efficiency.quantile(0.95)
            min_eff = efficiency.quantile(0.05)
            
            if max_eff > min_eff:
                return ((efficiency - min_eff) / (max_eff - min_eff) * 100).clip(0, 100)
        
        return pd.Series([50] * len(df))
    
    def _calculate_thermal_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate thermal management score (lower temp = higher score)"""
        
        if 'temperature_celsius' in df.columns:
            # Inverse scoring - lower temperature is better
            max_temp = 100  # Maximum acceptable temperature
            min_temp = 30   # Minimum expected temperature
            
            # Normalize: higher temperature = lower score
            thermal_score = ((max_temp - df['temperature_celsius']) / (max_temp - min_temp) * 100).clip(0, 100)
            return thermal_score
        
        return pd.Series([50] * len(df))
    
    def _calculate_reliability_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate reliability score based on error rates and test results"""
        
        reliability_score = pd.Series([50] * len(df))  # Default
        
        # Factor in error rates
        if 'error_rate' in df.columns:
            # Lower error rate = higher score
            max_error = df['error_rate'].quantile(0.95)
            if max_error > 0:
                error_score = ((max_error - df['error_rate']) / max_error * 100).clip(0, 100)
                reliability_score = reliability_score * 0.5 + error_score * 0.5
        
        # Factor in test results
        if 'test_result' in df.columns:
            test_score = (df['test_result'] == 'PASS').astype(int) * 100
            reliability_score = reliability_score * 0.5 + test_score * 0.5
        
        return reliability_score
    
    def calculate_chip_grade(self, df: pd.DataFrame) -> pd.Series:
        """Assign letter grades to chips based on overall performance"""
        
        scores = self.calculate_overall_performance_score(df)
        
        def score_to_grade(score):
            if score >= 90: return 'A+'
            elif score >= 85: return 'A'
            elif score >= 80: return 'A-'
            elif score >= 75: return 'B+'
            elif score >= 70: return 'B'
            elif score >= 65: return 'B-'
            elif score >= 60: return 'C+'
            elif score >= 55: return 'C'
            elif score >= 50: return 'C-'
            else: return 'F'
        
        return scores.apply(score_to_grade)
    
    def identify_performance_outliers(self, df: pd.DataFrame) -> Dict:
        """Identify chips with exceptional performance (good or bad)"""
        
        scores = self.calculate_overall_performance_score(df)
        
        # Statistical outliers
        q1 = scores.quantile(0.25)
        q3 = scores.quantile(0.75)
        iqr = q3 - q1
        
        high_performers = df[scores > (q3 + 1.5 * iqr)]
        poor_performers = df[scores < (q1 - 1.5 * iqr)]
        
        return {
            'high_performers': high_performers,
            'poor_performers': poor_performers,
            'high_performer_count': len(high_performers),
            'poor_performer_count': len(poor_performers),
            'outlier_rate': (len(high_performers) + len(poor_performers)) / len(df) * 100 if len(df) > 0 else 0
        }
    
    def calculate_manufacturing_yield(self, df: pd.DataFrame) -> Dict:
        """Calculate various yield metrics"""
        
        total_chips = len(df)
        passed_chips = (df['test_result'] == 'PASS').sum() if 'test_result' in df.columns else 0
        
        # Basic yield
        basic_yield = (passed_chips / total_chips * 100) if total_chips > 0 else 0
        
        # Quality yield (passed + good performance)
        quality_chips = 0
        if total_chips > 0:
            scores = self.calculate_overall_performance_score(df)
            quality_threshold = 70  # Minimum acceptable performance
            if 'test_result' in df.columns:
                quality_chips = ((df['test_result'] == 'PASS') & (scores >= quality_threshold)).sum()
            else:
                quality_chips = (scores >= quality_threshold).sum()
            quality_yield = quality_chips / total_chips * 100
        else:
            quality_yield = 0
        
        # Thermal yield (no thermal throttling)
        if 'thermal_throttling' in df.columns and total_chips > 0:
            thermal_good = (~df['thermal_throttling']).sum()
            thermal_yield = thermal_good / total_chips * 100
        else:
            thermal_yield = 100
        
        return {
            'basic_yield': basic_yield,
            'quality_yield': quality_yield,
            'thermal_yield': thermal_yield,
            'total_chips': total_chips,
            'passed_chips': int(passed_chips),
            'quality_chips': int(quality_chips)
        }
    
    def benchmark_against_industry(self, df: pd.DataFrame) -> Dict:
        """Compare performance against industry benchmarks"""
        
        results = {}
        
        if 'chip_type' not in df.columns:
            return {'error': 'No chip_type column found'}
        
        for chip_type in df['chip_type'].unique():
            chip_data = df[df['chip_type'] == chip_type]
            chip_type_lower = str(chip_type).lower()
            
            if chip_type_lower in self.benchmarks:
                benchmark = self.benchmarks[chip_type_lower]
                
                # Performance benchmark
                if 'performance_score' in chip_data.columns:
                    avg_performance = chip_data['performance_score'].mean()
                    performance_vs_benchmark = (avg_performance / benchmark['min_performance'] * 100) if avg_performance > 0 else 0
                else:
                    performance_vs_benchmark = 0
                
                # Temperature benchmark
                if 'temperature_celsius' in chip_data.columns:
                    temp_compliance = (chip_data['temperature_celsius'] <= benchmark['max_temp']).mean() * 100
                else:
                    temp_compliance = 100
                
                # Efficiency benchmark
                if 'efficiency_score' in chip_data.columns:
                    avg_efficiency = chip_data['efficiency_score'].mean()
                    efficiency_vs_benchmark = (avg_efficiency / benchmark['min_efficiency'] * 100) if avg_efficiency > 0 else 0
                else:
                    efficiency_vs_benchmark = 100
                
                results[chip_type] = {
                    'performance_vs_benchmark': performance_vs_benchmark,
                    'temperature_compliance': temp_compliance,
                    'efficiency_vs_benchmark': efficiency_vs_benchmark,
                    'overall_compliance': (performance_vs_benchmark >= 100 and 
                                         temp_compliance >= 95 and 
                                         efficiency_vs_benchmark >= 100)
                }
        
        return results
    
    def generate_comprehensive_report(self, df: pd.DataFrame) -> Dict:
        """Generate a comprehensive KPI report"""
        
        # Calculate all metrics
        overall_scores = self.calculate_overall_performance_score(df)
        grades = self.calculate_chip_grade(df)
        outliers = self.identify_performance_outliers(df)
        yield_metrics = self.calculate_manufacturing_yield(df)
        benchmark_results = self.benchmark_against_industry(df)
        
        # Performance distribution
        grade_distribution = grades.value_counts().to_dict()
        
        # Top performers - FIXED: Add scores to dataframe first
        df_with_scores = df.copy()
        df_with_scores['overall_performance_score'] = overall_scores
        top_performers = df_with_scores.nlargest(10, 'overall_performance_score')
        
        return {
            'summary': {
                'total_chips': len(df),
                'average_score': float(overall_scores.mean()),
                'score_std': float(overall_scores.std()),
                'grade_distribution': grade_distribution,
                'outlier_rate': outliers['outlier_rate']
            },
            'yield_metrics': yield_metrics,
            'benchmark_comparison': benchmark_results,
            'outlier_analysis': {
                'high_performer_count': outliers['high_performer_count'],
                'poor_performer_count': outliers['poor_performer_count'],
                'outlier_rate': outliers['outlier_rate']
            },
            'top_performers': top_performers[['chip_id', 'chip_type', 'manufacturer', 'performance_score', 'overall_performance_score']].to_dict('records'),
            'recommendations': self._generate_recommendations(df, overall_scores, yield_metrics, benchmark_results)
        }
    
    def _generate_recommendations(self, df: pd.DataFrame, scores: pd.Series, 
                                yield_metrics: Dict, benchmark_results: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        
        recommendations = []
        
        # Yield recommendations
        if yield_metrics['basic_yield'] < 90:
            recommendations.append(f"🔴 Basic yield is {yield_metrics['basic_yield']:.1f}% - investigate manufacturing process issues")
        
        if yield_metrics['quality_yield'] < 80:
            recommendations.append(f"⚠️ Quality yield is {yield_metrics['quality_yield']:.1f}% - review design specifications")
        
        # Temperature recommendations
        if 'temperature_celsius' in df.columns:
            avg_temp = df['temperature_celsius'].mean()
            if avg_temp > 80:
                recommendations.append(f"🌡️ Average temperature is {avg_temp:.1f}°C - improve thermal management")
        
        # Performance recommendations
        if scores.mean() < 70:
            recommendations.append("📉 Overall performance below target - optimize chip design and manufacturing")
        
        # Efficiency recommendations
        if 'efficiency_score' in df.columns:
            avg_efficiency = df['efficiency_score'].mean()
            if avg_efficiency < 50:
                recommendations.append("⚡ Power efficiency is low - consider design optimizations")
        
        # Chip type specific recommendations
        for chip_type, metrics in benchmark_results.items():
            if not metrics.get('overall_compliance', True):
                recommendations.append(f"🔧 {chip_type} chips not meeting industry benchmarks - review design")
        
        if not recommendations:
            recommendations.append("✅ All metrics within acceptable ranges - maintain current processes")
        
        return recommendations

# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    import pandas as pd
    
    calculator = AdvancedKPICalculator()
    
    # Load real data if available
    try:
        df = pd.read_csv('data/raw/chip_test_data/secom_real_data.csv')
        print("📊 Running KPI analysis on real SECOM data...")
        
        # Generate comprehensive report
        report = calculator.generate_comprehensive_report(df)
        
        print("\n🎯 KPI Analysis Results:")
        print("=" * 50)
        print(f"Total chips analyzed: {report['summary']['total_chips']:,}")
        print(f"Average performance score: {report['summary']['average_score']:.1f}")
        print(f"Basic yield: {report['yield_metrics']['basic_yield']:.1f}%")
        print(f"Quality yield: {report['yield_metrics']['quality_yield']:.1f}%")
        
        print(f"\n📈 Grade Distribution:")
        for grade, count in report['summary']['grade_distribution'].items():
            print(f"   {grade}: {count:,} chips")
        
        print(f"\n🏆 Top 5 Performers:")
        for i, performer in enumerate(report['top_performers'][:5], 1):
            print(f"   {i}. {performer['chip_id']} ({performer['chip_type']}) - Score: {performer['overall_performance_score']:.1f}")
        
        print(f"\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"   {rec}")
            
    except FileNotFoundError:
        print("❌ No data file found. Please run data processing first.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()