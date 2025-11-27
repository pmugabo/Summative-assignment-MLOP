import random
import time
import json
import os
from locust import HttpUser, task, between
from datetime import datetime


class AudioClassifierUser(HttpUser):
    """
    Simulates a user interacting with the UrbanSound8K API.
    """
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Called when a simulated user starts."""
        # Create a test audio file if it doesn't exist
        self.test_audio_path = self.create_test_audio()
        
        # Log the start time
        self.start_time = time.time()
        print(f"User {self.__class__.__name__} started at {datetime.now()}")
    
    def create_test_audio(self):
        """
        Create a test audio file for upload.
        In a real scenario, you would use actual audio files.
        """
        # For testing, we'll create a small dummy WAV file
        # In production, you should use real audio files
        test_audio_path = "test_audio.wav"
        
        if not os.path.exists(test_audio_path):
            # Create a simple sine wave as test audio
            import numpy as np
            import wave
            
            sample_rate = 22050
            duration = 2  # seconds
            frequency = 440  # Hz (A4 note)
            
            # Generate sine wave
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio_data = np.sin(2 * np.pi * frequency * t)
            
            # Convert to 16-bit integers
            audio_data = (audio_data * 32767).astype(np.int16)
            
            # Write WAV file
            with wave.open(test_audio_path, 'w') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
        
        return test_audio_path
    
    @task(70)  # 70% of users will do predictions
    def predict(self):
        """
        Simulate an audio prediction request.
        """
        try:
            # List of potential test files (use real files if available)
            test_files = [
                "data/raw/UrbanSound8K/audio/fold1/101415-3-0-2.wav",
                "data/raw/UrbanSound8K/audio/fold2/102305-6-0-0.wav",
                self.test_audio_path  # Fallback to generated test file
            ]
            
            # Choose an existing test file
            available_files = [f for f in test_files if os.path.exists(f)]
            if not available_files:
                test_file = self.test_audio_path
            else:
                test_file = random.choice(available_files)
            
            with open(test_file, 'rb') as audio_file:
                files = {'file': (os.path.basename(test_file), audio_file, 'audio/wav')}
                
                with self.client.post(
                    "/predict/",
                    files=files,
                    catch_response=True
                ) as response:
                    if response.status_code == 200:
                        result = response.json()
                        # Validate response structure
                        if 'prediction' in result and 'confidence' in result:
                            response.success()
                        else:
                            response.failure("Invalid response format")
                    else:
                        response.failure(f"HTTP {response.status_code}: {response.text}")
                        
        except Exception as e:
            print(f"Error in predict: {str(e)}")
    
    @task(20)  # 20% of users will check training status
    def get_training_status(self):
        """
        Check the training status endpoint.
        """
        with self.client.get("/training-status/", catch_response=True) as response:
            if response.status_code == 200:
                status = response.json()
                if 'status' in status:
                    response.success()
                else:
                    response.failure("Invalid status response")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(5)  # 5% of users will check health
    def get_health(self):
        """
        Check the health endpoint.
        """
        with self.client.get("/health/", catch_response=True) as response:
            if response.status_code == 200:
                health = response.json()
                if health.get('status') == 'healthy':
                    response.success()
                else:
                    response.failure("Service not healthy")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(3)  # 3% of users will check metrics
    def get_metrics(self):
        """
        Check the Prometheus metrics endpoint.
        """
        with self.client.get("/metrics/", catch_response=True) as response:
            if response.status_code == 200:
                # Check if it's Prometheus format
                if 'http_request_duration_seconds' in response.text:
                    response.success()
                else:
                    response.failure("Invalid metrics format")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(2)  # 2% of users will attempt retraining (rare operation)
    def retrain_model(self):
        """
        Simulate a model retraining request.
        This is a heavy operation, so it's less frequent.
        """
        try:
            # Create multiple test files for retraining
            files = []
            for i in range(3):  # Upload 3 files
                with open(self.test_audio_path, 'rb') as audio_file:
                    files.append(('files', (f'test_audio_{i}.wav', audio_file.read(), 'audio/wav')))
            
            with self.client.post(
                "/retrain/",
                files=files,
                catch_response=True
            ) as response:
                if response.status_code == 200:
                    result = response.json()
                    if result.get('status') == 'training_started':
                        response.success()
                    else:
                        response.failure("Retraining not started")
                else:
                    response.failure(f"HTTP {response.status_code}: {response.text}")
                    
        except Exception as e:
            print(f"Error in retrain_model: {str(e)}")
    
    def on_stop(self):
        """Called when a simulated user stops."""
        duration = time.time() - self.start_time
        print(f"User {self.__class__.__name__} stopped after {duration:.2f} seconds")


class HeavyUser(AudioClassifierUser):
    """
    Simulates a heavy user making frequent requests.
    """
    wait_time = between(0.5, 1.5)  # Shorter wait time
    
    @task(90)  # 90% predictions
    def predict(self):
        """Override with higher frequency."""
        super().predict()
    
    @task(10)  # 10% status checks
    def get_training_status(self):
        """Override with higher frequency."""
        super().get_training_status()


class AdminUser(AudioClassifierUser):
    """
    Simulates an admin user monitoring and managing the system.
    """
    wait_time = between(2, 5)
    
    @task(40)
    def get_training_status(self):
        """Admin checks status more frequently."""
        super().get_training_status()
    
    @task(30)
    def get_health(self):
        """Admin monitors health."""
        super().get_health()
    
    @task(20)
    def get_metrics(self):
        """Admin monitors metrics."""
        super().get_metrics()
    
    @task(10)
    def predict(self):
        """Admin occasionally tests predictions."""
        super().predict()


# Test scenarios for different load patterns
class StressTestUser(AudioClassifierUser):
    """
    User for stress testing with minimal wait times.
    """
    wait_time = between(0.1, 0.5)
    
    @task(95)
    def predict(self):
        """Almost all requests are predictions."""
        super().predict()
    
    @task(5)
    def get_health(self):
        """Occasional health checks."""
        super().get_health()


if __name__ == "__main__":
    """
    Instructions for running Locust tests:
    
    1. Start the API server:
       uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
    
    2. Run Locust with different scenarios:
    
    # Basic load test (default users):
    locust -f locust/locustfile.py --host http://localhost:8000
    
    # Stress test:
    locust -f locust/locustfile.py --host http://localhost:8000 -u 100 -r 10 --run-time 300s
    
    # Admin-heavy test:
    locust -f locust/locustfile.py --host http://localhost:8000 --user-class AdminUser
    
    3. Access Locust web interface at http://localhost:8089
    
    4. Configure test parameters:
       - Number of users: 1-1000
       - Spawn rate: 1-100 users/second
       - Test duration: 60-600 seconds
    
    5. Monitor results:
       - Average response time
       - Requests per second
       - Success rate
       - Percentiles (50%, 95%, 99%)
    """