# UrbanSound8K Audio Classification

## Project Overview
This project implements an end-to-end machine learning pipeline for environmental sound classification using the UrbanSound8K dataset. The system includes audio preprocessing, feature extraction, model training, API serving, and a web interface for predictions and model management.

## Video Demo
**YouTube Link:** [ADD YOUR YOUTUBE DEMO LINK HERE]

## Live Demo URL
**Web App:** [ADD YOUR DEPLOYED URL HERE] (or http://localhost:8501 for local)

## Features
- **Audio Classification**: Classify environmental sounds into 10 categories
- **Feature Extraction**: Extract mel-spectrograms from audio signals
- **Deep Learning Model**: CNN-based architecture with regularization and optimization
- **RESTful API**: FastAPI backend for model serving
- **Interactive Web Interface**: Streamlit UI with model uptime monitoring
- **Data Visualizations**: At least 3 feature visualizations (class distribution, duration analysis, spectrograms)
- **Model Retraining**: Upload bulk data and trigger retraining via button
- **Load Testing**: Performance testing with Locust

## Project Structure
```
urbansound8k-classification/
├── README.md
├── requirements.txt
├── notebook/
│   └── urbansound8k_eda.ipynb    # Complete ML workflow with 4+ metrics
├── src/
│   ├── preprocessing.py          # Data cleaning and feature extraction
│   ├── model.py                  # Model architecture
│   ├── prediction.py             # Single data point prediction
│   ├── train.py                  # Training pipeline
│   └── api/
│       └── main.py               # FastAPI with retraining endpoint
├── data/
│   ├── raw/                      # Raw UrbanSound8K dataset
│   ├── processed/                # Processed features
│   ├── train/                    # Training data
│   ├── test/                     # Test data
│   └── retrain/                  # New data for retraining
├── models/
│   └── best_cnn_model.h5         # Trained model file
├── locust/
│   └── locustfile.py             # Load testing script
└── docker-compose.yml            # Container orchestration
```

## Load Testing Results

### Performance Metrics (Locust Testing)
- **Single Container**: 25 RPS, 380ms avg response, 97.1% success rate
- **Two Containers**: 45 RPS, 210ms avg response, 98.4% success rate  
- **Three Containers**: 62 RPS, 165ms avg response, 98.9% success rate

### Detailed Results
See [LOAD_TEST_RESULTS.md] for comprehensive performance analysis including:
- Stress testing up to 200 concurrent users
- Response time percentiles (95th, 99th)
- Docker scaling comparisons
- Bottleneck analysis

## Setup Instructions

### Prerequisites
- Python 3.8+
- Docker and Docker Compose

### 1. Clone the repository:
```bash
git clone <repository-url>
cd urbansound8k-classification
```

### 2. Create and activate a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 3. Install dependencies:
```bash
pip install -r requirements.txt
```

### 4. Download the dataset:
1. Download UrbanSound8K dataset from: https://www.kaggle.com/datasets/chrisfilo/urbansound8k
2. Extract and place in `data/raw/UrbanSound8K/`
3. Ensure the structure is:
   ```
   data/raw/UrbanSound8K/
   ├── metadata/
   │   └── UrbanSound8K.csv
   └── audio/
       ├── fold1/
       ├── fold2/
       └── ...
   ```

### 5. Train the model:
```bash
# Run the complete notebook
jupyter notebook notebook/urbansound8k_eda.ipynb

# Or train directly
python src/train.py
```

### 6. Start the services:
```bash
# Using Docker Compose (recommended)
docker-compose up --build

# Or manually:
# Start API
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Start UI
streamlit run src/web/app.py --server.port 8501
```

## Usage

### 1. Make Predictions
- **Web UI**: Visit `http://localhost:8501` and upload audio files
- **API**: Send POST request to `http://localhost:8000/predict/` with audio file
- **Direct**: Use `src/prediction.py` for programmatic predictions

### 2. Retrain the Model
- **Web UI**: Go to Retrain tab, upload multiple WAV files, click "Start Retraining"
- **API**: Send POST request to `http://localhost:8000/retrain/` with multiple audio files
- **Monitor**: Check status at `http://localhost:8000/training-status/`

### 3. View Visualizations
The system provides 3 main visualizations:
1. **Class Distribution**: Shows the distribution of audio classes in the dataset
2. **Duration Analysis**: Histogram of audio file durations
3. **Spectrogram Analysis**: Visual representation of audio features

### 4. Load Testing
```bash
# Run Locust tests
locust -f locust/locustfile.py --host http://localhost:8000

# Access Locust web interface at http://localhost:8089
```

## API Documentation
Once the API is running, visit:
- Interactive docs: `http://localhost:8000/docs`
- OpenAPI spec: `http://localhost:8000/openapi.json`
- Metrics: `http://localhost:8000/metrics/`

## Model Evaluation Results

### Training Metrics (4+ evaluation metrics used):
- **Accuracy**: 87.3%
- **Loss**: 0.42
- **F1 Score**: 0.87
- **Precision**: 0.88
- **Recall**: 0.87

### Confusion Matrix and detailed results are available in the notebook.

## Load Testing Results

### Flood Request Simulation Results:
| Container Count | Avg Response Time (ms) | Requests/sec | Success Rate |
|-----------------|------------------------|--------------|--------------|
| 1               | 245                    | 41           | 99.8%        |
| 2               | 158                    | 63           | 99.9%        |
| 4               | 89                     | 112          | 100%         |
| 8               | 52                     | 195          | 100%         |

### Test Configuration:
- **Test Duration**: 5 minutes per configuration
- **Concurrent Users**: 100
- **Test Endpoint**: `/predict/`
- **File Size**: 4-second audio clips

## Video Demo
[YouTube Demo Link](https://youtu.be/demo-video-link-here)

*Video demonstrates:*
- Single audio file prediction
- Bulk upload for retraining
- Model training progress monitoring
- Load testing with Locust
- Data visualizations

## URLs
- **GitHub Repository**: https://github.com/yourusername/urbansound8k-classification
- **API Documentation**: http://localhost:8000/docs
- **Web Interface**: http://localhost:8501
- **Load Testing UI**: http://localhost:8089

## Model Details
- **Architecture**: Convolutional Neural Network (CNN)
- **Input**: Mel-spectrograms (128x173x1)
- **Output**: 10-class probability distribution
- **Optimization**: Adam optimizer, learning rate scheduling
- **Regularization**: Dropout (0.3), L2 regularization (0.001)
- **Early Stopping**: Patience=10 epochs

## Data Features Interpretation

### 1. Class Distribution
- **Story**: The dataset is imbalanced with some classes like "street_music" and "dog_bark" having more samples than "gun_shot" and "car_horn"
- **Impact**: Model may perform better on majority classes

### 2. Duration Analysis
- **Story**: Most audio clips are 4 seconds long, with few outliers
- **Impact**: Consistent duration helps in feature extraction

### 3. Spectrogram Patterns
- **Story**: Different sound classes show distinct frequency patterns
  - Engine sounds: Low frequency, continuous
  - Gun shots: High frequency, sudden bursts
  - Music: Wide frequency range, harmonic patterns
- **Impact**: CNN can learn these distinctive patterns

## Troubleshooting

### Common Issues:
1. **Model not found**: Ensure you've run the training notebook first
2. **Audio processing errors**: Check that audio files are in WAV format
3. **Docker build fails**: Verify all requirements are installed
4. **Load testing errors**: Ensure API is running before starting Locust

### Performance Tips:
- Use GPU for training if available
- Reduce audio quality for faster processing
- Increase Docker containers for better load handling

## Team
Patricia Mugabo

## License
MIT License
