# src/api/routes/data_routes.py - COMPLETE VERSION
from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import logging
import asyncio
import json
import uvicorn

# Add src to path
sys.path.append('src')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title="Semiconductor Performance Analytics API",
    description="REST API for chip performance data and analytics",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for web dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class ChipData(BaseModel):
    chip_id: str
    chip_type: str
    manufacturer: str
    architecture_nm: Optional[str] = None
    clock_speed_ghz: float = Field(..., gt=0, le=10)
    power_consumption_watts: float = Field(..., gt=0, le=1000)
    temperature_celsius: float = Field(..., gt=-50, le=150)
    voltage_v: float = Field(..., gt=0, le=5)
    current_a: Optional[float] = None
    performance_score: Optional[float] = None
    error_rate: Optional[float] = Field(None, ge=0, le=1)
    test_result: Optional[str] = Field(None, pattern="^(PASS|FAIL)$") # <--- CHANGE THIS LINE
    thermal_throttling: Optional[bool] = None
    cache_hit_ratio: Optional[float] = Field(None, ge=0, le=1)

class PredictionRequest(BaseModel):
    chip_data: ChipData
    model_types: List[str] = Field(default=["performance", "failure", "efficiency"])

class PredictionResponse(BaseModel):
    chip_id: str
    predictions: Dict[str, float]
    risk_level: str
    recommendations: List[str]

class AnalyticsRequest(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    chip_types: Optional[List[str]] = None
    manufacturers: Optional[List[str]] = None
    include_charts: bool = False

class KPIResponse(BaseModel):
    total_units: int
    yield_rate: float
    avg_performance: float
    avg_temperature: float
    avg_efficiency: float
    thermal_incidents: int
    quality_score: float

# Import ML models and analytics modules
try:
    from models.ml_models.predictive_models import ChipPerformancePredictor
    ML_MODELS_AVAILABLE = True
    logger.info("✅ ML models loaded successfully")
except ImportError as e:
    ML_MODELS_AVAILABLE = False
    logger.warning(f"⚠️ ML models not available: {e}")

try:
    from analysis.anomaly.outlier_detector import AdvancedAnomalyDetector
    ANOMALY_DETECTION_AVAILABLE = True
    logger.info("✅ Anomaly detection loaded successfully")
except ImportError as e:
    ANOMALY_DETECTION_AVAILABLE = False
    logger.warning(f"⚠️ Anomaly detection not available: {e}")

try:
    from analysis.performance.kpi_calculator import AdvancedKPICalculator
    KPI_CALCULATOR_AVAILABLE = True
    logger.info("✅ KPI calculator loaded successfully")
except ImportError as e:
    KPI_CALCULATOR_AVAILABLE = False
    logger.warning(f"⚠️ KPI calculator not available: {e}")

try:
    from visualization.reports.automated_reports import ReportGenerator
    REPORTS_AVAILABLE = True
    logger.info("✅ Report generator loaded successfully")
except ImportError as e:
    REPORTS_AVAILABLE = False
    logger.warning(f"⚠️ Report generator not available: {e}")

try:
    from data.ingestion.real_time_streamer import RealTimeDataStreamer
    STREAMING_AVAILABLE = True
    logger.info("✅ Real-time streaming loaded successfully")
except ImportError as e:
    STREAMING_AVAILABLE = False
    logger.warning(f"⚠️ Real-time streaming not available: {e}")

# Global instances
ml_predictor = None
anomaly_detector = None
kpi_calculator = None
report_generator = None
data_streamer = None

# Initialize components
def initialize_components():
    """Initialize all system components"""
    global ml_predictor, anomaly_detector, kpi_calculator, report_generator, data_streamer
    
    try:
        if ML_MODELS_AVAILABLE:
            ml_predictor = ChipPerformancePredictor()
            logger.info("✅ ML predictor initialized")
        
        if ANOMALY_DETECTION_AVAILABLE:
            anomaly_detector = AdvancedAnomalyDetector(contamination_rate=0.05)
            logger.info("✅ Anomaly detector initialized")
        
        if KPI_CALCULATOR_AVAILABLE:
            kpi_calculator = AdvancedKPICalculator()
            logger.info("✅ KPI calculator initialized")
        
        if REPORTS_AVAILABLE:
            report_generator = ReportGenerator()
            logger.info("✅ Report generator initialized")
        
        if STREAMING_AVAILABLE:
            data_streamer = RealTimeDataStreamer()
            logger.info("✅ Data streamer initialized")
            
    except Exception as e:
        logger.error(f"❌ Error initializing components: {e}")

# Data access functions
def load_chip_data(limit: int = None, offset: int = 0) -> pd.DataFrame:
    """Load chip data from available sources"""
    possible_paths = [
        'data/raw/chip_test_data/secom_real_data.csv',
        'data/raw/chip_test_data/chip_performance_data.csv',
        'data/streaming/realtime_data.db'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                if path.endswith('.csv'):
                    df = pd.read_csv(path)
                elif path.endswith('.db'):
                    import sqlite3
                    conn = sqlite3.connect(path)
                    df = pd.read_sql_query("SELECT * FROM realtime_chip_data", conn)
                    conn.close()
                
                # Ensure timestamp column
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                else:
                    df['timestamp'] = pd.date_range('2024-01-01', periods=len(df), freq='1H')
                
                # Apply pagination
                if limit:
                    df = df.iloc[offset:offset+limit]
                
                logger.info(f"✅ Loaded {len(df)} records from {path}")
                return df
            except Exception as e:
                logger.warning(f"⚠️ Could not load {path}: {e}")
                continue
    
    raise HTTPException(status_code=404, detail="No chip data found")

def filter_data(df: pd.DataFrame, start_date: str = None, end_date: str = None,
                chip_types: List[str] = None, manufacturers: List[str] = None) -> pd.DataFrame:
    """Apply filters to the dataframe"""
    filtered_df = df.copy()
    
    if start_date:
        start_dt = pd.to_datetime(start_date)
        filtered_df = filtered_df[filtered_df['timestamp'] >= start_dt]
    
    if end_date:
        end_dt = pd.to_datetime(end_date)
        filtered_df = filtered_df[filtered_df['timestamp'] <= end_dt]
    
    if chip_types and 'chip_type' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['chip_type'].isin(chip_types)]
    
    if manufacturers and 'manufacturer' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['manufacturer'].isin(manufacturers)]
    
    return filtered_df

# API Routes

@app.get("/", tags=["General"])
async def root():
    """API root endpoint"""
    return {
        "message": "Semiconductor Performance Analytics API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "operational",
        "modules": {
            "ml_models": ML_MODELS_AVAILABLE,
            "anomaly_detection": ANOMALY_DETECTION_AVAILABLE,
            "kpi_calculator": KPI_CALCULATOR_AVAILABLE,
            "reports": REPORTS_AVAILABLE,
            "streaming": STREAMING_AVAILABLE
        }
    }

@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint"""
    try:
        df = load_chip_data(limit=1)
        data_available = len(df) > 0
    except:
        data_available = False
    
    return {
        "status": "healthy" if data_available else "degraded",
        "timestamp": datetime.now().isoformat(),
        "data_available": data_available,
        "modules_loaded": {
            "ml_models": ML_MODELS_AVAILABLE,
            "anomaly_detection": ANOMALY_DETECTION_AVAILABLE,
            "kpi_calculator": KPI_CALCULATOR_AVAILABLE,
            "reports": REPORTS_AVAILABLE,
            "streaming": STREAMING_AVAILABLE
        }
    }

@app.get("/data/chips", tags=["Data Access"])
async def get_chips(
    limit: int = Query(100, le=10000, description="Maximum number of records"),
    offset: int = Query(0, ge=0, description="Number of records to skip"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    chip_type: Optional[str] = Query(None, description="Filter by chip type"),
    manufacturer: Optional[str] = Query(None, description="Filter by manufacturer")
):
    """Get chip data with optional filtering and pagination"""
    try:
        df = load_chip_data()
        
        # Apply filters
        chip_types = [chip_type] if chip_type else None
        manufacturers = [manufacturer] if manufacturer else None
        filtered_df = filter_data(df, start_date, end_date, chip_types, manufacturers)
        
        # Apply pagination
        paginated_df = filtered_df.iloc[offset:offset+limit]
        
        # Convert to JSON-serializable format
        result = paginated_df.replace({np.nan: None}).to_dict('records')
        
        return {
            "data": result,
            "total_count": len(filtered_df),
            "returned_count": len(result),
            "offset": offset,
            "limit": limit,
            "filters_applied": {
                "start_date": start_date,
                "end_date": end_date,
                "chip_type": chip_type,
                "manufacturer": manufacturer
            }
        }
    
    except Exception as e:
        logger.error(f"Error in get_chips: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/chips/{chip_id}", tags=["Data Access"])
async def get_chip_by_id(chip_id: str):
    """Get specific chip data by ID"""
    try:
        df = load_chip_data()
        chip_data = df[df['chip_id'] == chip_id]
        
        if chip_data.empty:
            raise HTTPException(status_code=404, detail=f"Chip {chip_id} not found")
        
        result = chip_data.replace({np.nan: None}).iloc[0].to_dict()
        return {"chip": result}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_chip_by_id: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/kpis", response_model=KPIResponse, tags=["Analytics"])
async def get_kpis(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    chip_types: Optional[str] = Query(None, description="Comma-separated chip types"),
    manufacturers: Optional[str] = Query(None, description="Comma-separated manufacturers")
):
    """Get key performance indicators"""
    try:
        df = load_chip_data()
        
        # Parse comma-separated filters
        chip_type_list = chip_types.split(',') if chip_types else None
        manufacturer_list = manufacturers.split(',') if manufacturers else None
        
        # Apply filters
        filtered_df = filter_data(df, start_date, end_date, chip_type_list, manufacturer_list)
        
        if filtered_df.empty:
            raise HTTPException(status_code=404, detail="No data found for specified filters")
        
        # Calculate KPIs
        total_units = len(filtered_df)
        
        # Basic metrics with proper null handling
        if 'test_result' in filtered_df.columns:
            yield_rate = (filtered_df['test_result'] == 'PASS').mean() * 100
        else:
            yield_rate = 95.0  # Default
        
        avg_performance = filtered_df.get('performance_score', pd.Series([5000] * total_units)).mean()
        avg_temperature = filtered_df.get('temperature_celsius', pd.Series([70] * total_units)).mean()
        
        # Calculate efficiency
        if 'efficiency_score' in filtered_df.columns:
            avg_efficiency = filtered_df['efficiency_score'].mean()
        elif 'performance_score' in filtered_df.columns and 'power_consumption_watts' in filtered_df.columns:
            avg_efficiency = (filtered_df['performance_score'] / filtered_df['power_consumption_watts']).mean()
        else:
            avg_efficiency = 45.0  # Default
        
        thermal_incidents = filtered_df.get('thermal_throttling', pd.Series([False] * total_units)).sum()
        
        # Calculate quality score
        quality_factors = [
            min(yield_rate / 95, 1.0),  # Target 95% yield
            max(0, 1 - (avg_temperature - 70) / 30),  # Penalty for high temp
            1 - min(thermal_incidents / total_units * 10, 0.5)  # Penalty for thermal issues
        ]
        quality_score = np.mean(quality_factors) * 100
        
        return KPIResponse(
            total_units=total_units,
            yield_rate=round(yield_rate, 2),
            avg_performance=round(avg_performance, 1),
            avg_temperature=round(avg_temperature, 1),
            avg_efficiency=round(avg_efficiency, 2),
            thermal_incidents=int(thermal_incidents),
            quality_score=round(quality_score, 1)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_kpis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analytics/advanced", tags=["Analytics"])
async def get_advanced_analytics(request: AnalyticsRequest):
    """Get advanced analytics with optional chart data"""
    try:
        df = load_chip_data()
        
        # Apply filters
        filtered_df = filter_data(
            df, request.start_date, request.end_date, 
            request.chip_types, request.manufacturers
        )
        
        if filtered_df.empty:
            raise HTTPException(status_code=404, detail="No data found for specified filters")
        
        # Use KPI calculator if available
        if KPI_CALCULATOR_AVAILABLE and kpi_calculator:
            try:
                kpi_report = kpi_calculator.generate_comprehensive_report(filtered_df)
                
                analytics = {
                    "summary": kpi_report.get('summary', {}),
                    "performance_metrics": {
                        "avg_performance": float(filtered_df.get('performance_score', pd.Series([5000])).mean()),
                        "performance_std": float(filtered_df.get('performance_score', pd.Series([5000])).std()),
                        "grade_distribution": kpi_report.get('summary', {}).get('grade_distribution', {}),
                        "top_performers": kpi_report.get('top_performers', [])[:5]
                    },
                    "quality_metrics": kpi_report.get('yield_metrics', {}),
                    "recommendations": kpi_report.get('recommendations', [])
                }
            except Exception as e:
                logger.warning(f"KPI calculator error: {e}")
                analytics = self._generate_basic_analytics(filtered_df)
        else:
            analytics = self._generate_basic_analytics(filtered_df)
        
        # Add chart data if requested
        if request.include_charts:
            analytics["charts"] = self._generate_chart_data(filtered_df)
        
        return analytics
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_advanced_analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _generate_basic_analytics(df: pd.DataFrame) -> Dict:
    """Generate basic analytics when KPI calculator is not available"""
    total_chips = len(df)
    
    # Performance metrics
    perf_col = 'performance_score' if 'performance_score' in df.columns else df.select_dtypes(include=[np.number]).columns[0]
    avg_performance = df[perf_col].mean()
    performance_std = df[perf_col].std()
    
    # Quality metrics
    if 'test_result' in df.columns:
        yield_rate = (df['test_result'] == 'PASS').mean() * 100
        defect_rate = 100 - yield_rate
    else:
        yield_rate = 95.0
        defect_rate = 5.0
    
    return {
        "summary": {
            "total_chips": total_chips,
            "date_range": {
                "start": df['timestamp'].min().isoformat() if 'timestamp' in df.columns else None,
                "end": df['timestamp'].max().isoformat() if 'timestamp' in df.columns else None
            }
        },
        "performance_metrics": {
            "avg_performance": float(avg_performance),
            "performance_std": float(performance_std),
            "min_performance": float(df[perf_col].min()),
            "max_performance": float(df[perf_col].max())
        },
        "quality_metrics": {
            "yield_rate": yield_rate,
            "defect_rate": defect_rate
        },
        "recommendations": ["Basic analytics mode - install full KPI calculator for detailed insights"]
    }

def _generate_chart_data(df: pd.DataFrame) -> Dict:
    """Generate chart data"""
    charts = {}
    
    # Performance distribution
    perf_col = 'performance_score' if 'performance_score' in df.columns else df.select_dtypes(include=[np.number]).columns[0]
    perf_hist, perf_bins = np.histogram(df[perf_col].dropna(), bins=20)
    charts["performance_distribution"] = {
        "counts": perf_hist.tolist(),
        "bins": perf_bins.tolist()
    }
    
    # Daily trends if timestamp exists
    if 'timestamp' in df.columns:
        try:
            daily_stats = df.groupby(df['timestamp'].dt.date).agg({
                perf_col: 'mean',
                'test_result': lambda x: (x == 'PASS').mean() * 100 if 'test_result' in df.columns else 95,
                'temperature_celsius': 'mean' if 'temperature_celsius' in df.columns else lambda x: 70
            }).reset_index()
            
            charts["daily_trends"] = {
                "dates": daily_stats['timestamp'].astype(str).tolist(),
                "avg_performance": daily_stats[perf_col].tolist(),
                "yield_rate": daily_stats['test_result'].tolist() if 'test_result' in df.columns else [95] * len(daily_stats),
                "avg_temperature": daily_stats['temperature_celsius'].tolist() if 'temperature_celsius' in df.columns else [70] * len(daily_stats)
            }
        except Exception as e:
            logger.warning(f"Error generating daily trends: {e}")
    
    return charts

@app.post("/predict", response_model=PredictionResponse, tags=["Machine Learning"])
async def predict_chip_performance(request: PredictionRequest):
    """Predict chip performance, failure probability, and efficiency"""
    try:
        chip_data = request.chip_data.dict()
        predictions = {}
        recommendations = []
        
        if ML_MODELS_AVAILABLE and ml_predictor:
            # Use trained ML models
            try:
                if "performance" in request.model_types:
                    pred_performance = ml_predictor.predict_performance(chip_data)
                    predictions["performance_score"] = float(pred_performance)
                    
                    if pred_performance < 6000:
                        recommendations.append("Performance below average - review design parameters")
                
                if "failure" in request.model_types:
                    failure_prob = ml_predictor.predict_failure_probability(chip_data)
                    predictions["failure_probability"] = float(failure_prob)
                    
                    if failure_prob > 0.3:
                        recommendations.append("High failure risk - implement additional quality checks")
                    elif failure_prob > 0.1:
                        recommendations.append("Moderate failure risk - monitor closely")
                
                if "efficiency" in request.model_types:
                    pred_efficiency = ml_predictor.predict_efficiency(chip_data)
                    predictions["efficiency_score"] = float(pred_efficiency)
                    
                    if pred_efficiency < 40:
                        recommendations.append("Low efficiency - optimize power consumption")
                        
            except Exception as e:
                logger.warning(f"ML prediction error: {e}")
                # Fall back to heuristic predictions
                predictions.update(_generate_heuristic_predictions(chip_data, request.model_types))
        else:
            # Use heuristic predictions
            predictions.update(_generate_heuristic_predictions(chip_data, request.model_types))
        
        # Determine risk level
        failure_prob = predictions.get("failure_probability", 0.05)
        if failure_prob > 0.3:
            risk_level = "HIGH"
        elif failure_prob > 0.1:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Add thermal recommendations
        if chip_data.get("temperature_celsius", 0) > 85:
            recommendations.append("High temperature detected - improve thermal management")
        
        if not recommendations:
            recommendations.append("All metrics within acceptable ranges")
        
        return PredictionResponse(
            chip_id=request.chip_data.chip_id,
            predictions=predictions,
            risk_level=risk_level,
            recommendations=recommendations
        )
    
    except Exception as e:
        logger.error(f"Error in predict_chip_performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _generate_heuristic_predictions(chip_data: Dict, model_types: List[str]) -> Dict:
    """Generate heuristic predictions when ML models are not available"""
    predictions = {}
    
    # Basic heuristic calculations
    temp = chip_data.get("temperature_celsius", 65)
    power = chip_data.get("power_consumption_watts", 150)
    clock_speed = chip_data.get("clock_speed_ghz", 3.0)
    
    if "performance" in model_types:
        # Simple performance heuristic
        base_perf = 5000
        temp_factor = max(0.8, 1.2 - (temp - 65) / 50)
        clock_factor = clock_speed / 3.0
        predictions["performance_score"] = base_perf * temp_factor * clock_factor
    
    if "failure" in model_types:
        # Simple failure probability heuristic
        temp_risk = max(0, (temp - 80) / 50)
        power_risk = max(0, (power - 200) / 200)
        base_risk = 0.02
        predictions["failure_probability"] = min(0.95, base_risk + temp_risk + power_risk)
    
    if "efficiency" in model_types:
        # Simple efficiency heuristic
        if power > 0:
            predictions["efficiency_score"] = predictions.get("performance_score", 5000) / power
        else:
            predictions["efficiency_score"] = 40.0
    
    return predictions

@app.get("/analytics/anomalies", tags=["Analytics"])
async def detect_anomalies(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    threshold: float = Query(0.05, description="Anomaly detection threshold")
):
    """Detect anomalies in chip performance data"""
    try:
        df = load_chip_data()
        
        # Apply date filters
        if start_date or end_date:
            filtered_df = filter_data(df, start_date, end_date)
        else:
            filtered_df = df
        
        if filtered_df.empty:
            raise HTTPException(status_code=404, detail="No data found for specified period")
        
        if ANOMALY_DETECTION_AVAILABLE and anomaly_detector:
            # Use advanced anomaly detection
            try:
                detector = AdvancedAnomalyDetector(contamination_rate=threshold)
                report = detector.generate_anomaly_report(filtered_df)
                
                return {
                    "anomaly_summary": report.get('summary', {}),
                    "pattern_analysis": report.get('pattern_analysis', {}),
                    "recommendations": report.get('recommendations', []),
                    "detection_timestamp": datetime.now().isoformat(),
                    "method": "advanced_ml"
                }
            except Exception as e:
                logger.warning(f"Advanced anomaly detection error: {e}")
                return _generate_basic_anomaly_report(filtered_df, threshold)
        else:
            return _generate_basic_anomaly_report(filtered_df, threshold)
    
    except Exception as e:
        logger.error(f"Error in detect_anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _generate_basic_anomaly_report(df: pd.DataFrame, threshold: float) -> Dict:
    """Generate basic anomaly report using statistical methods"""
    # Simple statistical anomaly detection
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    total_anomalies = 0
    
    for col in numeric_cols:
        # Z-score method
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        anomalies = (z_scores > 3).sum()
        total_anomalies += anomalies
    
    anomaly_rate = (total_anomalies / len(df)) * 100 if len(df) > 0 else 0
    
    return {
        "anomaly_summary": {
            "total_records": len(df),
            "total_anomalies": int(total_anomalies),
            "anomaly_rate": round(anomaly_rate, 2)
        },
        "pattern_analysis": {
            "message": "Basic statistical analysis - upgrade to advanced detection for detailed patterns"
        },
        "recommendations": [
            f"Anomaly rate: {anomaly_rate:.1f}%",
            "Enable advanced anomaly detection for detailed analysis"
        ],
        "detection_timestamp": datetime.now().isoformat(),
        "method": "basic_statistical"
    }

@app.post("/data/chips", tags=["Data Management"])
async def add_chip_data(chip_data: ChipData):
    """Add new chip test data"""
    try:
        # Convert to dictionary
        chip_dict = chip_data.dict()
        chip_dict['timestamp'] = datetime.now().isoformat()
        chip_dict['created_via_api'] = True
        
        # Basic validation and enhancement
        if chip_data.temperature_celsius > 100:
            chip_dict['thermal_warning'] = True
        
        # Calculate derived metrics
        if chip_data.performance_score and chip_data.power_consumption_watts:
            chip_dict['efficiency_score'] = chip_data.performance_score / chip_data.power_consumption_watts
        
        if chip_data.power_consumption_watts and chip_data.voltage_v:
            chip_dict['current_a'] = chip_data.power_consumption_watts / chip_data.voltage_v
        
        # In a real implementation, save to database
        # For now, log the data
        logger.info(f"Received chip data: {chip_dict['chip_id']}")
        
        return {
            "message": "Chip data received successfully",
            "chip_data": chip_dict,
            "status": "validated",
            "timestamp": chip_dict['timestamp']
        }
    
    except Exception as e:
        logger.error(f"Error in add_chip_data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reports/generate", tags=["Reporting"])
async def generate_report(
    background_tasks: BackgroundTasks,
    report_type: str = Query(..., regex="^(daily|weekly)$"),
    date: Optional[str] = Query(None, description="Report date (YYYY-MM-DD)")
):
    """Generate automated reports"""
    try:
        if not REPORTS_AVAILABLE:
            raise HTTPException(status_code=503, detail="Report generation service not available")
        
        def generate_report_task(report_type: str, date_str: str = None):
            """Background task to generate report"""
            try:
                generator = ReportGenerator()
                
                if date_str:
                    report_date = datetime.strptime(date_str, '%Y-%m-%d')
                else:
                    report_date = datetime.now()
                
                if report_type == "daily":
                    report_path = generator.generate_daily_report(report_date)
                else:
                    report_path = generator.generate_weekly_report(report_date)
                
                logger.info(f"Report generated: {report_path}")
                
            except Exception as e:
                logger.error(f"Background report generation failed: {e}")
        
        # Start background task
        background_tasks.add_task(generate_report_task, report_type, date)
        
        return {
            "message": f"{report_type.title()} report generation started",
            "report_type": report_type,
            "date": date or datetime.now().strftime('%Y-%m-%d'),
            "status": "processing",
            "estimated_completion": "2-5 minutes"
        }
    
    except Exception as e:
        logger.error(f"Error in generate_report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/export", tags=["Data Management"])
async def export_data(
    format: str = Query("csv", regex="^(csv|json|excel)$"),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    limit: int = Query(10000, le=50000)
):
    """Export chip data in various formats"""
    try:
        df = load_chip_data(limit=limit)
        
        # Apply date filters
        if start_date or end_date:
            df = filter_data(df, start_date, end_date)
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found for export")
        
        # Create export directory
        export_dir = "data/outputs/exports"
        os.makedirs(export_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == "csv":
            filename = f"chip_data_export_{timestamp}.csv"
            filepath = os.path.join(export_dir, filename)
            df.to_csv(filepath, index=False)
        elif format == "json":
            filename = f"chip_data_export_{timestamp}.json"
            filepath = os.path.join(export_dir, filename)
            df.to_json(filepath, orient='records', date_format='iso')
        elif format == "excel":
            filename = f"chip_data_export_{timestamp}.xlsx"
            filepath = os.path.join(export_dir, filename)
            df.to_excel(filepath, index=False)
        
        return FileResponse(
            filepath,
            filename=filename,
            media_type='application/octet-stream'
        )
    
    except Exception as e:
        logger.error(f"Error in export_data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/streaming/status", tags=["Real-time Data"])
async def get_streaming_status():
    """Get real-time data streaming status"""
    try:
        if not STREAMING_AVAILABLE or not data_streamer:
            return {
                "streaming_available": False,
                "message": "Real-time streaming not available"
            }
        
        stats = data_streamer.get_streaming_stats()
        return {
            "streaming_available": True,
            "status": stats,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error getting streaming status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/streaming/start", tags=["Real-time Data"])
async def start_streaming(interval: float = Query(2.0, description="Streaming interval in seconds")):
    """Start real-time data streaming"""
    try:
        if not STREAMING_AVAILABLE or not data_streamer:
            raise HTTPException(status_code=503, detail="Real-time streaming not available")
        
        data_streamer.start_streaming(interval=interval)
        
        return {
            "message": "Real-time streaming started",
            "interval": interval,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error starting streaming: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/streaming/stop", tags=["Real-time Data"])
async def stop_streaming():
    """Stop real-time data streaming"""
    try:
        if not STREAMING_AVAILABLE or not data_streamer:
            raise HTTPException(status_code=503, detail="Real-time streaming not available")
        
        data_streamer.stop_streaming()
        
        return {
            "message": "Real-time streaming stopped",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error stopping streaming: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ml/train", tags=["Machine Learning"])
async def train_models(background_tasks: BackgroundTasks):
    """Train ML models in background"""
    try:
        if not ML_MODELS_AVAILABLE:
            raise HTTPException(status_code=503, detail="ML models not available")
        
        def train_models_task():
            """Background task to train ML models"""
            try:
                global ml_predictor
                ml_predictor = ChipPerformancePredictor()
                results = ml_predictor.train_all_models()
                logger.info(f"Model training completed: {results}")
            except Exception as e:
                logger.error(f"Model training failed: {e}")
        
        background_tasks.add_task(train_models_task)
        
        return {
            "message": "ML model training started",
            "status": "processing",
            "estimated_completion": "5-15 minutes",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error starting model training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Resource not found", 
            "detail": str(exc.detail),
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error", 
            "detail": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Semiconductor Performance Analytics API...")
    
    # Initialize all components
    initialize_components()
    
    logger.info("✅ API started successfully")
    logger.info("📚 API documentation available at /docs")
    logger.info("🔍 Health check available at /health")

if __name__ == "__main__":
    # Run the FastAPI application
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        log_level="info",
        reload=True  # Enable auto-reload for development
    )