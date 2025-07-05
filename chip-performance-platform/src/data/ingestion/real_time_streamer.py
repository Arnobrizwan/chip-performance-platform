# src/data/ingestion/real_time_streamer.py
import asyncio
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import logging
import time
import threading
from queue import Queue, Empty
import sqlite3
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChipDataSimulator:
    """Simulates real-time chip testing data"""
    
    def __init__(self):
        self.chip_counter = 0
        self.manufacturers = ['Intel', 'AMD', 'NVIDIA', 'TSMC', 'Samsung']
        self.chip_types = ['CPU', 'GPU', 'ASIC', 'FPGA', 'SoC']
        self.architectures = ['3nm', '5nm', '7nm', '10nm', '14nm']
        
    def generate_chip_data(self) -> Dict:
        """Generate realistic chip testing data"""
        self.chip_counter += 1
        
        # Base metrics with realistic correlations
        chip_type = np.random.choice(self.chip_types, p=[0.35, 0.25, 0.15, 0.15, 0.10])
        manufacturer = np.random.choice(self.manufacturers)
        architecture = np.random.choice(self.architectures, p=[0.1, 0.2, 0.3, 0.25, 0.15])
        
        # Performance varies by chip type
        type_performance_base = {
            'CPU': 8500, 'GPU': 12000, 'FPGA': 6500, 'ASIC': 15000, 'SoC': 7000
        }
        
        base_performance = type_performance_base[chip_type]
        clock_speed = np.random.normal(3.2, 0.8)
        clock_speed = np.clip(clock_speed, 1.5, 5.5)
        
        # Power consumption correlates with performance and architecture
        arch_factor = {'3nm': 0.7, '5nm': 0.8, '7nm': 1.0, '10nm': 1.2, '14nm': 1.4}
        power_consumption = clock_speed * 40 * arch_factor[architecture] + np.random.normal(0, 20)
        power_consumption = np.clip(power_consumption, 50, 400)
        
        # Temperature correlates with power
        temperature = power_consumption * 0.25 + np.random.normal(40, 8)
        temperature = np.clip(temperature, 35, 95)
        
        # Performance affected by thermal throttling
        performance_score = base_performance * (clock_speed / 3.0) * np.random.normal(1.0, 0.1)
        if temperature > 85:
            performance_score *= 0.9  # Thermal throttling
        performance_score = np.clip(performance_score, 3000, 20000)
        
        # Voltage and current
        voltage = np.random.normal(1.2, 0.15)
        voltage = np.clip(voltage, 0.8, 1.8)
        current = power_consumption / voltage
        
        # Quality metrics
        error_rate = np.random.exponential(0.0005)
        error_rate = np.clip(error_rate, 0, 0.01)
        
        # Test result based on performance and temperature
        pass_probability = 0.95
        if performance_score < base_performance * 0.7:
            pass_probability = 0.6
        if temperature > 90:
            pass_probability *= 0.8
        if error_rate > 0.005:
            pass_probability *= 0.7
            
        test_result = 'PASS' if np.random.random() < pass_probability else 'FAIL'
        
        return {
            'chip_id': f'RT_CHIP_{self.chip_counter:06d}',
            'timestamp': datetime.now().isoformat(),
            'chip_type': chip_type,
            'manufacturer': manufacturer,
            'architecture_nm': architecture,
            'clock_speed_ghz': round(clock_speed, 2),
            'power_consumption_watts': round(power_consumption, 1),
            'temperature_celsius': round(temperature, 1),
            'voltage_v': round(voltage, 2),
            'current_a': round(current, 2),
            'performance_score': round(performance_score, 0),
            'error_rate': round(error_rate, 6),
            'test_result': test_result,
            'thermal_throttling': temperature > 85,
            'efficiency_score': round(performance_score / power_consumption, 2),
            'cache_hit_ratio': round(np.random.beta(9, 1), 3)
        }

class RealTimeDataStreamer:
    """Real-time data streaming system for chip performance data"""
    
    def __init__(self, buffer_size: int = 1000, batch_size: int = 10):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.data_queue = Queue(maxsize=buffer_size)
        self.subscribers = []
        self.is_streaming = False
        self.simulator = ChipDataSimulator()
        self.db_path = 'data/streaming/realtime_data.db'
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for real-time data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS realtime_chip_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chip_id TEXT,
                timestamp TEXT,
                chip_type TEXT,
                manufacturer TEXT,
                architecture_nm TEXT,
                clock_speed_ghz REAL,
                power_consumption_watts REAL,
                temperature_celsius REAL,
                voltage_v REAL,
                current_a REAL,
                performance_score REAL,
                error_rate REAL,
                test_result TEXT,
                thermal_throttling BOOLEAN,
                efficiency_score REAL,
                cache_hit_ratio REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def subscribe(self, callback: Callable[[Dict], None]):
        """Subscribe to real-time data updates"""
        self.subscribers.append(callback)
        logger.info(f"New subscriber added. Total subscribers: {len(self.subscribers)}")
        
    def unsubscribe(self, callback: Callable[[Dict], None]):
        """Unsubscribe from real-time data updates"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
            logger.info(f"Subscriber removed. Total subscribers: {len(self.subscribers)}")
    
    def _notify_subscribers(self, data: Dict):
        """Notify all subscribers of new data"""
        for callback in self.subscribers:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")
    
    def _store_data(self, data: Dict):
        """Store data in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO realtime_chip_data (
                    chip_id, timestamp, chip_type, manufacturer, architecture_nm,
                    clock_speed_ghz, power_consumption_watts, temperature_celsius,
                    voltage_v, current_a, performance_score, error_rate,
                    test_result, thermal_throttling, efficiency_score, cache_hit_ratio
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['chip_id'], data['timestamp'], data['chip_type'],
                data['manufacturer'], data['architecture_nm'], data['clock_speed_ghz'],
                data['power_consumption_watts'], data['temperature_celsius'],
                data['voltage_v'], data['current_a'], data['performance_score'],
                data['error_rate'], data['test_result'], data['thermal_throttling'],
                data['efficiency_score'], data['cache_hit_ratio']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing data: {e}")
    
    def start_streaming(self, interval: float = 2.0):
        """Start real-time data streaming"""
        if self.is_streaming:
            logger.warning("Streaming already active")
            return
            
        self.is_streaming = True
        logger.info(f"Starting real-time data streaming (interval: {interval}s)")
        
        def stream_worker():
            while self.is_streaming:
                try:
                    # Generate new chip data
                    chip_data = self.simulator.generate_chip_data()
                    
                    # Add to queue
                    if not self.data_queue.full():
                        self.data_queue.put(chip_data)
                    
                    # Store in database
                    self._store_data(chip_data)
                    
                    # Notify subscribers
                    self._notify_subscribers(chip_data)
                    
                    logger.info(f"Streamed data for {chip_data['chip_id']}")
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    logger.error(f"Error in streaming worker: {e}")
                    time.sleep(1)
        
        # Start streaming in background thread
        self.stream_thread = threading.Thread(target=stream_worker, daemon=True)
        self.stream_thread.start()
    
    def stop_streaming(self):
        """Stop real-time data streaming"""
        self.is_streaming = False
        logger.info("Stopped real-time data streaming")
    
    def get_latest_data(self, count: int = 10) -> List[Dict]:
        """Get latest data from queue"""
        data = []
        for _ in range(min(count, self.data_queue.qsize())):
            try:
                data.append(self.data_queue.get_nowait())
            except Empty:
                break
        return data
    
    def get_historical_data(self, hours: int = 24) -> pd.DataFrame:
        """Get historical data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT * FROM realtime_chip_data 
                WHERE created_at >= datetime('now', '-{} hours')
                ORDER BY created_at DESC
            '''.format(hours)
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving historical data: {e}")
            return pd.DataFrame()
    
    def get_streaming_stats(self) -> Dict:
        """Get streaming statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total records
            cursor.execute("SELECT COUNT(*) FROM realtime_chip_data")
            total_records = cursor.fetchone()[0]
            
            # Records in last hour
            cursor.execute('''
                SELECT COUNT(*) FROM realtime_chip_data 
                WHERE created_at >= datetime('now', '-1 hour')
            ''')
            recent_records = cursor.fetchone()[0]
            
            # Latest timestamp
            cursor.execute('''
                SELECT MAX(created_at) FROM realtime_chip_data
            ''')
            latest_timestamp = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'total_records': total_records,
                'recent_records': recent_records,
                'latest_timestamp': latest_timestamp,
                'queue_size': self.data_queue.qsize(),
                'is_streaming': self.is_streaming,
                'subscribers': len(self.subscribers)
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}

class DataQualityMonitor:
    """Monitor data quality in real-time"""
    
    def __init__(self, streamer: RealTimeDataStreamer):
        self.streamer = streamer
        self.quality_metrics = {
            'total_processed': 0,
            'anomalies_detected': 0,
            'quality_score': 100.0,
            'last_check': datetime.now()
        }
        
        # Subscribe to data stream
        self.streamer.subscribe(self._check_data_quality)
    
    def _check_data_quality(self, data: Dict):
        """Check quality of incoming data"""
        self.quality_metrics['total_processed'] += 1
        
        # Check for anomalies
        anomaly_detected = False
        
        # Temperature anomaly
        if data['temperature_celsius'] > 95 or data['temperature_celsius'] < 20:
            anomaly_detected = True
            
        # Performance anomaly
        if data['performance_score'] < 1000 or data['performance_score'] > 25000:
            anomaly_detected = True
            
        # Power anomaly
        if data['power_consumption_watts'] < 10 or data['power_consumption_watts'] > 500:
            anomaly_detected = True
        
        if anomaly_detected:
            self.quality_metrics['anomalies_detected'] += 1
            logger.warning(f"Data quality anomaly detected in {data['chip_id']}")
        
        # Update quality score
        anomaly_rate = self.quality_metrics['anomalies_detected'] / self.quality_metrics['total_processed']
        self.quality_metrics['quality_score'] = max(0, 100 - (anomaly_rate * 100))
        self.quality_metrics['last_check'] = datetime.now()
    
    def get_quality_report(self) -> Dict:
        """Get data quality report"""
        return self.quality_metrics.copy()

# Example usage and testing
if __name__ == "__main__":
    # Create and start real-time streamer
    streamer = RealTimeDataStreamer()
    
    # Add quality monitor
    quality_monitor = DataQualityMonitor(streamer)
    
    # Example subscriber
    def data_logger(data):
        print(f"Received: {data['chip_id']} - Performance: {data['performance_score']}")
    
    streamer.subscribe(data_logger)
    
    # Start streaming
    streamer.start_streaming(interval=1.0)  # 1 second intervals
    
    try:
        # Run for demo
        print("🚀 Starting real-time chip data streaming...")
        print("Press Ctrl+C to stop")
        
        while True:
            time.sleep(5)
            
            # Show stats every 5 seconds
            stats = streamer.get_streaming_stats()
            quality = quality_monitor.get_quality_report()
            
            print(f"\n📊 Streaming Stats:")
            print(f"   Total Records: {stats['total_records']}")
            print(f"   Recent Records: {stats['recent_records']}")
            print(f"   Queue Size: {stats['queue_size']}")
            print(f"   Data Quality: {quality['quality_score']:.1f}%")
            print(f"   Anomalies: {quality['anomalies_detected']}/{quality['total_processed']}")
            
    except KeyboardInterrupt:
        print("\n⏹️ Stopping real-time streaming...")
        streamer.stop_streaming()
        print("✅ Streaming stopped")