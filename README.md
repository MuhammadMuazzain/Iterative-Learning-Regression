# AI Text Detection System

A production-level machine learning system that detects whether text is human-written, AI-generated, or human-edited AI content. The system uses fine-tuned RoBERTa models with a continuous feedback loop for model improvement.

## 🚀 Features

- **Text Classification**: Classify text as Human-written, AI-generated, or Human-edited AI
- **Regression Scoring**: Continuous output (0.0-1.0) for more nuanced predictions
- **Feedback Collection**: Collect user feedback through a web interface
- **Automated Retraining**: Process feedback data with quality gates for model improvement
- **Model Versioning**: Multiple model versions (v1-v4) with MLflow tracking
- **RESTful API**: FastAPI backend with prediction and feedback endpoints
- **Web Interface**: Simple HTML frontend for testing and feedback submission

## 📋 Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [API Documentation](#api-documentation)
- [Feedback & Retraining](#feedback--retraining)
- [Development](#development)

## 🔧 Installation

### Prerequisites

- Python 3.10+
- PostgreSQL database
- CUDA (optional, for GPU acceleration)
- Git

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Production-Level
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # On Windows PowerShell:
   Copy-Item .env.example .env
   ```

4. **Configure your `.env` file:**
   Edit `.env` and set your actual credentials:
   ```env
   DB_NAME=ai_text_feedback
   DB_USER=postgres
   DB_PASSWORD=your_actual_password
   DB_HOST=localhost
   DB_PORT=5432
   
   MLFLOW_TRACKING_URI=sqlite:///./mlflow.db
   MODEL_PATH=runs:/bf6fa44badde4e9b80412dce11457a0c/roberta_large_lora-sigmoid
   MODEL_NAME=roberta-large
   ```

5. **Set up the database:**
   
   Create a PostgreSQL database and run the schema script:
   ```bash
   # Create database (as PostgreSQL superuser)
   createdb ai_text_feedback
   
   # Run the schema script
   psql -d ai_text_feedback -f db/schema.sql
   ```
   
   Or manually create the database and run the SQL commands from `db/schema.sql`.
   
   The schema includes:
   - `feedback_events` table to store user feedback
   - Automatic `target_score` calculation via database trigger
   - Indexes for optimal query performance
   
   **Alternative:** If you prefer to set up manually, see the SQL commands in `db/schema.sql`.

6. **Test database connection:**
   ```bash
   python db/test_postgre.py
   ```

##  Usage

### Starting the API Server

```bash
# From the project root
cd backend
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# Or using Python directly
python backend/api.py
```

The API will be available at `http://localhost:8000`

### Accessing the Web Interface

Open your browser and navigate to:
```
http://localhost:8000/
```

The web interface allows you to:
- Submit text for prediction
- View model scores and classifications
- Submit feedback on predictions

### Making API Calls

**Get Prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here"}'
```

**Submit Feedback:**
```bash
curl -X POST "http://localhost:8000/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your text here",
    "model_score": 0.75,
    "user_feedback": "ai",
    "model_version": "roberta_large_lora-sigmoid"
  }'
```

## 📁 Project Structure

```
Production-Level/
├── backend/                 # FastAPI application
│   ├── api.py             # Main API with prediction and feedback endpoints
│   └── api_v2.py          # Alternative API implementation
├── frontend/               # Web interface
│   └── index.html         # Simple HTML frontend
├── trainer/                # Model training scripts
│   ├── v1/                # Initial RoBERTa-base model
│   ├── v2/                # RoBERTa-large with LoRA
│   ├── v3/                # Enhanced version 3
│   └── v4/                # Latest model version
├── retrainer/              # Feedback processing and retraining
│   └── db_fetch_qualgates.py  # Data quality gates for feedback
├── db/                     # Database utilities
│   ├── schema.sql         # Database schema with triggers
│   └── test_postgre.py    # Database connection test
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── .gitignore             # Git ignore rules
├── README.md              # This file
└── SECURITY_MIGRATION_SUMMARY.md  # Security migration details (optional reference)
```

## 🧠 Model Training

### Training a New Model

Navigate to the desired version directory and run the training script:

```bash
cd trainer/v4
python v4_final.py
```

### Model Versions

- **v1**: RoBERTa-base with full fine-tuning
- **v2**: RoBERTa-large with LoRA (Low-Rank Adaptation)
- **v3**: Enhanced v2 with additional features
- **v4**: Latest version with improved architecture

### Training Configuration

Models use the following default settings:
- **Base Model**: `roberta-large`
- **Max Length**: 512 tokens
- **Batch Size**: 4
- **Learning Rate**: 2e-4
- **Epochs**: 3
- **Loss Function**: MSE (Mean Squared Error)
- **Framework**: PyTorch Lightning with LoRA

### MLflow Tracking

All experiments are tracked with MLflow. To view results:

```bash
# Start MLflow UI
mlflow ui --backend-store-uri sqlite:///./mlflow.db
```

Then open `http://localhost:5000` in your browser.

## 📡 API Documentation

### Endpoints

#### `POST /predict`
Get a prediction for text classification.

**Request Body:**
```json
{
  "text": "Your text to classify"
}
```

**Response:**
```json
{
  "text": "Your text to classify",
  "score": 0.75,
  "label": "AI-generated"
}
```

**Score Interpretation:**
- `score < 0.35`: Human-written
- `0.35 ≤ score ≤ 0.65`: Human-edited AI
- `score > 0.65`: AI-generated

#### `POST /feedback`
Submit user feedback on a prediction.

**Request Body:**
```json
{
  "text": "The original text",
  "model_score": 0.75,
  "user_feedback": "ai",
  "model_version": "roberta_large_lora-sigmoid"
}
```

**Valid feedback values:** `human`, `ai`, `human_edited_ai`

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda"
}
```

## 🔄 Feedback & Retraining

### Processing Feedback Data

The system includes automated data quality gates for feedback processing:

```bash
cd retrainer
python db_fetch_qualgates.py
```

This script:
1. Fetches feedback from the database
2. Applies deduplication (keeps latest feedback per text)
3. Filters by language (English only)
4. Filters by text length (5-1000 characters)
5. Removes inconsistent feedback (contradictory labels)
6. Exports clean data to `clean_feedback.csv`

### Retraining Workflow

1. Collect feedback through the web interface
2. Run `db_fetch_qualgates.py` to process feedback
3. Use the cleaned CSV for model retraining
4. Log new model to MLflow
5. Update `MODEL_PATH` in `.env` to use the new model

## 🛠️ Development

### Running Tests

```bash
# Test database connection
python db/test_postgre.py

# Check CUDA availability
python cuda_check.py
```

### Code Style

This project follows PEP 8 guidelines. Consider using:
- `black` for code formatting
- `flake8` or `pylint` for linting
- `mypy` for type checking

### Adding New Features

1. Create a new branch: `git checkout -b feature/new-feature`
2. Make your changes
3. Test thoroughly
4. Submit a pull request

### Environment Variables

All sensitive configuration is stored in `.env` file. See `.env.example` for available variables.

**Important:** Never commit the `.env` file to version control!

## 🔒 Security

- All sensitive data (passwords, API keys) is stored in environment variables
- Database credentials are never hardcoded
- `.env` file is excluded from version control via `.gitignore`
- Use different `.env` files for development, staging, and production

## 📊 Model Performance

Models are evaluated using:
- Mean Squared Error (MSE)
- Confusion matrices at different thresholds
- Calibration curves
- Residual distributions

Results are automatically logged to MLflow for comparison.

## 🐛 Troubleshooting

### Common Issues

**Database Connection Error:**
- Verify PostgreSQL is running
- Check `.env` file has correct credentials
- Test connection with `python db/test_postgre.py`

**Model Loading Error:**
- Verify `MODEL_PATH` in `.env` points to a valid MLflow run
- Check MLflow tracking URI is correct
- Ensure model artifacts exist in MLflow

**CUDA Out of Memory:**
- Reduce batch size in training scripts
- Use smaller models or LoRA for memory efficiency
- Enable gradient checkpointing

**Import Errors:**
- Install all dependencies: `pip install -r requirements.txt`
- Ensure Python version is 3.10+

## 📝 License

[Add your license here]

## 👥 Contributors

[Add contributor information here]

## 🙏 Acknowledgments

- Hugging Face Transformers for pre-trained models
- PyTorch Lightning for training framework
- MLflow for experiment tracking
- FastAPI for the REST API framework

## 📞 Support

For issues and questions, please open an issue on the repository.

---

**Note:** This is a production-level system. Ensure proper testing and validation before deploying to production environments.

