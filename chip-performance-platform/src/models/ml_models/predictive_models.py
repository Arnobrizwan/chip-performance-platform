# src/models/ml_models/predictive_models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report, r2_score
import joblib
import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChipPerformancePredictor:
    """ML models for predicting chip performance and failures"""
    
    def __init__(self, model_dir: str = "data/models/trained_models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize models
        self.performance_model = None
        self.failure_model = None
        self.efficiency_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # Model configurations
        self.performance_features = [
            'clock_speed_ghz', 'power_consumption_watts', 'temperature_celsius',
            'voltage_v', 'current_a', 'cache_hit_ratio'
        ]
        
        self.failure_features = [
            'performance_score', 'temperature_celsius', 'power_consumption_watts',
            'voltage_v', 'error_rate', 'thermal_throttling', 'efficiency_score'
        ]
        
        self.efficiency_features = [
            'performance_score', 'power_consumption_watts', 'clock_speed_ghz',
            'temperature_celsius', 'voltage_v'
        ]
    
    def load_and_prepare_data(self, data_path: str = None) -> pd.DataFrame:
        """Load and prepare data for training"""
        if data_path is None:
            # Try to load from default locations
            possible_paths = [
                'data/raw/chip_test_data/secom_real_data.csv',
                'data/raw/chip_test_data/chip_performance_data.csv'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    data_path = path
                    break
            
            if data_path is None:
                raise FileNotFoundError("No data file found for training")
        
        df = pd.read_csv(data_path)
        
        # Data preprocessing
        df = self._preprocess_data(df)
        
        logger.info(f"Loaded and preprocessed {len(df)} records from {data_path}")
        return df
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for ML models"""
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        # Handle categorical variables
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col not in ['test_result', 'thermal_throttling']:  # Keep these as targets
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        # Create derived features
        if 'performance_score' in df.columns and 'power_consumption_watts' in df.columns:
            df['efficiency_score'] = df['performance_score'] / df['power_consumption_watts']
        
        # Encode categorical variables
        for col in ['chip_type', 'manufacturer', 'architecture_nm']:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    try:
                        df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
                    except ValueError:
                        # Handle unseen categories
                        df[f'{col}_encoded'] = 0
        
        return df
    
    def train_performance_predictor(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train model to predict chip performance score"""
        logger.info("Training performance prediction model...")
        
        # Prepare features and target
        available_features = [f for f in self.performance_features if f in df.columns]
        
        # Add encoded categorical features
        for col in ['chip_type', 'manufacturer', 'architecture_nm']:
            encoded_col = f'{col}_encoded'
            if encoded_col in df.columns:
                available_features.append(encoded_col)
        
        X = df[available_features].copy()
        y = df['performance_score']
        
        # Remove any remaining NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train multiple models and select best
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression()
        }
        
        best_model = None
        best_score = float('-inf')
        results = {}
        
        for name, model in models.items():
            if name == 'LinearRegression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            results[name] = {
                'r2_score': r2,
                'mae': mae,
                'rmse': rmse
            }
            
            if r2 > best_score:
                best_score = r2
                best_model = model
                self.performance_model = model
        
        # Save model
        model_path = os.path.join(self.model_dir, 'performance_predictor.joblib')
        joblib.dump({
            'model': self.performance_model,
            'scaler': self.scaler,
            'features': available_features,
            'label_encoders': self.label_encoders
        }, model_path)
        
        logger.info(f"Performance model trained and saved. Best R² score: {best_score:.3f}")
        return results
    
    def train_failure_predictor(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train model to predict chip test failures"""
        logger.info("Training failure prediction model...")
        
        # Prepare features and target
        available_features = [f for f in self.failure_features if f in df.columns]
        
        # Add encoded categorical features
        for col in ['chip_type', 'manufacturer', 'architecture_nm']:
            encoded_col = f'{col}_encoded'
            if encoded_col in df.columns:
                available_features.append(encoded_col)
        
        X = df[available_features].copy()
        y = (df['test_result'] == 'FAIL').astype(int)
        
        # Remove any remaining NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train models
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'LogisticRegression': LogisticRegression(random_state=42, class_weight='balanced')
        }
        
        best_model = None
        best_score = 0
        results = {}
        
        for name, model in models.items():
            if name == 'LogisticRegression':
                # Scale features for logistic regression
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'accuracy': accuracy,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            if accuracy > best_score:
                best_score = accuracy
                best_model = model
                self.failure_model = model
        
        # Save model
        model_path = os.path.join(self.model_dir, 'failure_predictor.joblib')
        joblib.dump({
            'model': self.failure_model,
            'features': available_features,
            'label_encoders': self.label_encoders
        }, model_path)
        
        logger.info(f"Failure model trained and saved. Best accuracy: {best_score:.3f}")
        return results
    
    def train_efficiency_predictor(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train model to predict power efficiency"""
        logger.info("Training efficiency prediction model...")
        
        # Prepare features and target
        available_features = [f for f in self.efficiency_features if f in df.columns]
        
        # Add encoded categorical features
        for col in ['chip_type', 'manufacturer', 'architecture_nm']:
            encoded_col = f'{col}_encoded'
            if encoded_col in df.columns:
                available_features.append(encoded_col)
        
        X = df[available_features].copy()
        y = df['efficiency_score']
        
        # Remove any remaining NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.efficiency_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.efficiency_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.efficiency_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results = {
            'r2_score': r2,
            'mae': mae,
            'rmse': rmse
        }
        
        # Save model
        model_path = os.path.join(self.model_dir, 'efficiency_predictor.joblib')
        joblib.dump({
            'model': self.efficiency_model,
            'features': available_features,
            'label_encoders': self.label_encoders
        }, model_path)
        
        logger.info(f"Efficiency model trained and saved. R² score: {r2:.3f}")
        return results
    
    def predict_performance(self, chip_data: Dict) -> float:
        """Predict performance score for a chip"""
        if self.performance_model is None:
            self._load_model('performance_predictor.joblib')
        
        # Prepare input data
        input_df = pd.DataFrame([chip_data])
        input_df = self._preprocess_data(input_df)
        
        # Get features used in training
        model_data = joblib.load(os.path.join(self.model_dir, 'performance_predictor.joblib'))
        features = model_data['features']
        
        X = input_df[features].fillna(0)
        
        # Scale if needed (for linear models)
        if isinstance(self.performance_model, LinearRegression):
            X = model_data['scaler'].transform(X)
        
        prediction = self.performance_model.predict(X)[0]
        return float(prediction)
    
    def predict_failure_probability(self, chip_data: Dict) -> float:
        """Predict probability of test failure"""
        if self.failure_model is None:
            self._load_model('failure_predictor.joblib')
        
        # Prepare input data
        input_df = pd.DataFrame([chip_data])
        input_df = self._preprocess_data(input_df)
        
        # Get features used in training
        model_data = joblib.load(os.path.join(self.model_dir, 'failure_predictor.joblib'))
        features = model_data['features']
        
        X = input_df[features].fillna(0)
        
        if hasattr(self.failure_model, 'predict_proba'):
            probability = self.failure_model.predict_proba(X)[0][1]  # Probability of failure
        else:
            probability = self.failure_model.predict(X)[0]
        
        return float(probability)
    
    def predict_efficiency(self, chip_data: Dict) -> float:
        """Predict power efficiency"""
        if self.efficiency_model is None:
            self._load_model('efficiency_predictor.joblib')
        
        # Prepare input data
        input_df = pd.DataFrame([chip_data])
        input_df = self._preprocess_data(input_df)
        
        # Get features used in training
        model_data = joblib.load(os.path.join(self.model_dir, 'efficiency_predictor.joblib'))
        features = model_data['features']
        
        X = input_df[features].fillna(0)
        
        prediction = self.efficiency_model.predict(X)[0]
        return float(prediction)
    
    def _load_model(self, model_filename: str):
        """Load a trained model"""
        model_path = os.path.join(self.model_dir, model_filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model_data = joblib.load(model_path)
        
        if 'performance' in model_filename:
            self.performance_model = model_data['model']
        elif 'failure' in model_filename:
            self.failure_model = model_data['model']
        elif 'efficiency' in model_filename:
            self.efficiency_model = model_data['model']
        
        # Update encoders if available
        if 'label_encoders' in model_data:
            self.label_encoders.update(model_data['label_encoders'])
    
    def get_feature_importance(self, model_type: str = 'performance') -> Dict[str, float]:
        """Get feature importance from trained models"""
        model = None
        features = []
        
        try:
            if model_type == 'performance' and self.performance_model is not None:
                model = self.performance_model
                model_data = joblib.load(os.path.join(self.model_dir, 'performance_predictor.joblib'))
                features = model_data['features']
            elif model_type == 'failure' and self.failure_model is not None:
                model = self.failure_model
                model_data = joblib.load(os.path.join(self.model_dir, 'failure_predictor.joblib'))
                features = model_data['features']
            elif model_type == 'efficiency' and self.efficiency_model is not None:
                model = self.efficiency_model
                model_data = joblib.load(os.path.join(self.model_dir, 'efficiency_predictor.joblib'))
                features = model_data['features']
        except FileNotFoundError:
            return {}
        
        if model is not None and hasattr(model, 'feature_importances_'):
            importance_dict = dict(zip(features, model.feature_importances_))
            # Sort by importance
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return {}
    
    def train_all_models(self, data_path: str = None) -> Dict[str, Any]:
        """Train all ML models"""
        logger.info("Starting comprehensive model training...")
        
        # Load and prepare data
        df = self.load_and_prepare_data(data_path)
        
        results = {}
        
        # Train performance predictor
        try:
            results['performance'] = self.train_performance_predictor(df)
        except Exception as e:
            logger.error(f"Error training performance model: {e}")
            results['performance'] = {'error': str(e)}
        
        # Train failure predictor
        try:
            results['failure'] = self.train_failure_predictor(df)
        except Exception as e:
            logger.error(f"Error training failure model: {e}")
            results['failure'] = {'error': str(e)}
        
        # Train efficiency predictor
        try:
            results['efficiency'] = self.train_efficiency_predictor(df)
        except Exception as e:
            logger.error(f"Error training efficiency model: {e}")
            results['efficiency'] = {'error': str(e)}
        
        # Save training metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'data_size': len(df),
            'features_used': {
                'performance': self.performance_features,
                'failure': self.failure_features,
                'efficiency': self.efficiency_features
            },
            'results': results
        }
        
        metadata_path = os.path.join(self.model_dir, 'training_metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Model training completed successfully")
        return results
    
    def generate_prediction_report(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive prediction report"""
        logger.info("Generating prediction report...")
        
        predictions = []
        
        for _, row in test_data.iterrows():
            chip_data = row.to_dict()
            
            try:
                pred_performance = self.predict_performance(chip_data)
                pred_failure_prob = self.predict_failure_probability(chip_data)
                pred_efficiency = self.predict_efficiency(chip_data)
                
                predictions.append({
                    'chip_id': chip_data.get('chip_id', 'Unknown'),
                    'predicted_performance': pred_performance,
                    'failure_probability': pred_failure_prob,
                    'predicted_efficiency': pred_efficiency,
                    'actual_performance': chip_data.get('performance_score', None),
                    'actual_result': chip_data.get('test_result', None),
                    'risk_level': 'HIGH' if pred_failure_prob > 0.3 else 'MEDIUM' if pred_failure_prob > 0.1 else 'LOW'
                })
            except Exception as e:
                logger.error(f"Error predicting for chip {chip_data.get('chip_id', 'Unknown')}: {e}")
        
        # Calculate summary statistics
        pred_df = pd.DataFrame(predictions)
        
        summary = {
            'total_predictions': len(predictions),
            'high_risk_chips': (pred_df['risk_level'] == 'HIGH').sum(),
            'medium_risk_chips': (pred_df['risk_level'] == 'MEDIUM').sum(),
            'low_risk_chips': (pred_df['risk_level'] == 'LOW').sum(),
            'avg_predicted_performance': pred_df['predicted_performance'].mean(),
            'avg_failure_probability': pred_df['failure_probability'].mean(),
            'avg_predicted_efficiency': pred_df['predicted_efficiency'].mean()
        }
        
        return {
            'predictions': predictions,
            'summary': summary,
            'feature_importance': {
                'performance': self.get_feature_importance('performance'),
                'failure': self.get_feature_importance('failure'),
                'efficiency': self.get_feature_importance('efficiency')
            }
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize predictor
    predictor = ChipPerformancePredictor()
    
    # Train all models
    print("🚀 Training ML models...")
    results = predictor.train_all_models()
    
    print("\n📊 Training Results:")
    for model_type, result in results.items():
        print(f"\n{model_type.upper()} Model:")
        if 'error' not in result:
            for metric, value in result.items():
                if isinstance(value, dict):
                    for sub_metric, sub_value in value.items():
                        if isinstance(sub_value, (int, float)):
                            print(f"  {sub_metric}: {sub_value:.3f}")
                elif isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.3f}")
        else:
            print(f"  Error: {result['error']}")
    
    # Test predictions
    print("\n🔮 Testing Predictions:")
    test_chip = {
        'chip_type': 'CPU',
        'manufacturer': 'Intel',
        'architecture_nm': '7nm',
        'clock_speed_ghz': 3.2,
        'power_consumption_watts': 125.0,
        'temperature_celsius': 68.5,
        'voltage_v': 1.35,
        'current_a': 92.6,
        'cache_hit_ratio': 0.94,
        'error_rate': 0.0001,
        'thermal_throttling': False
    }
    
    try:
        performance_pred = predictor.predict_performance(test_chip)
        failure_prob = predictor.predict_failure_probability(test_chip)
        efficiency_pred = predictor.predict_efficiency(test_chip)
        
        print(f"Predicted Performance: {performance_pred:.0f}")
        print(f"Failure Probability: {failure_prob:.3f}")
        print(f"Predicted Efficiency: {efficiency_pred:.2f}")
        
    except Exception as e:
        print(f"Error in predictions: {e}")
    
    print("\n✅ ML model training and testing completed!")