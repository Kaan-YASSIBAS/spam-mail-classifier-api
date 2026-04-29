# Spam Mail Classifier API

![CI](https://github.com/Kaan-YASSIBAS/spam-mail-classifier-api/actions/workflows/ci.yml/badge.svg)

A containerized NLP classification web application built with **FastAPI**, **scikit-learn**, **TF-IDF**, **Logistic Regression**, Docker, and a simple frontend.

The application classifies text messages as either:

- `spam`
- `ham`

## Project Overview

This project demonstrates an end-to-end NLP and Machine Learning workflow:

1. Use the Kaggle SMS Spam Collection dataset
2. Preprocess text data with TF-IDF vectorization
3. Train a Logistic Regression classifier
4. Evaluate the model with classification metrics
5. Save the trained pipeline as a `.pkl` file with Joblib
6. Serve the model through a FastAPI backend
7. Validate incoming requests with Pydantic
8. Test the API with Pytest
9. Containerize the backend with Docker
10. Add a simple HTML/CSS/JavaScript frontend
11. Run the full application with Docker Compose
12. Run tests automatically with GitHub Actions CI

## Tech Stack

- Python
- FastAPI
- Pydantic
- scikit-learn
- pandas
- joblib
- Uvicorn
- Pytest
- Docker
- Docker Compose
- Nginx
- HTML / CSS / JavaScript
- GitHub Actions

## Project Structure

```text
spam-mail-classifier-api/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   └── model/
│   │       └── spam_classifier_model.pkl
│   │
│   ├── data/
│   │   └── spam.csv
│   │
│   ├── scripts/
│   │   └── train_model.py
│   │
│   ├── tests/
│   │   └── test_api.py
│   │
│   ├── Dockerfile
│   ├── .dockerignore
│   ├── requirements.txt
│   └── pytest.ini
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   ├── Dockerfile
│   └── .dockerignore
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── docker-compose.yml
├── README.md
└── .gitignore
```

## Features

- Trains a spam classification model
- Uses TF-IDF for text vectorization
- Uses Logistic Regression for classification
- Saves the full ML pipeline as a `.pkl` file
- Serves predictions through FastAPI
- Provides a `/health` endpoint
- Provides a `/predict` endpoint
- Validates request data with Pydantic
- Includes Swagger UI for API testing
- Includes Pytest-based API tests
- Includes a simple frontend UI
- Supports Docker and Docker Compose
- Includes GitHub Actions CI
- Includes Docker image build checks in CI

## Dataset

This project uses the **SMS Spam Collection Dataset** from Kaggle.

The dataset contains SMS messages labeled as:

| Label | Meaning |
| --- | --- |
| `ham` | Normal, legitimate message |
| `spam` | Unwanted or suspicious message |

The dataset is stored at:

```text
backend/data/spam.csv
```

The original Kaggle dataset commonly uses these columns:

| Original Column | Meaning |
| --- | --- |
| `v1` | Label: `ham` or `spam` |
| `v2` | Message text |

In the training script, these columns are renamed to:

| Project Column | Meaning |
| --- | --- |
| `label` | Message class |
| `message` | SMS text |

## How the Model Works

The model is built as a scikit-learn pipeline:

```text
Raw text message
↓
TF-IDF Vectorizer
↓
Logistic Regression Classifier
↓
spam / ham prediction
```

### TF-IDF

TF-IDF stands for:

```text
Term Frequency - Inverse Document Frequency
```

It converts text into numerical vectors by giving higher importance to words that are meaningful in a specific message but not too common across all messages.

Example spam-related words may include:

```text
free
prize
winner
click
urgent
claim
```

### Logistic Regression

Logistic Regression is a supervised Machine Learning classification model.

In this project, it learns patterns from TF-IDF features and predicts whether a message is:

```text
spam
```

or:

```text
ham
```

## Model Training

Go to the backend folder:

```bash
cd backend
```

Run the training script:

```bash
python scripts/train_model.py
```

The script will:

1. Load the dataset
2. Clean and rename columns
3. Split the data into train and test sets
4. Train the TF-IDF + Logistic Regression pipeline
5. Print classification metrics
6. Save the model to:

```text
backend/app/model/spam_classifier_model.pkl
```

## Model Evaluation

The training script prints metrics such as:

```text
Accuracy
Precision
Recall
F1 Score
Confusion Matrix
Classification Report
```

Example output:

```text
Accuracy: 0.9704
Precision: 1.0
Recall: 0.7785
F1 Score: 0.8754
```

## What Is a `.pkl` File?

A `.pkl` file is a serialized Python object file.

In this project, the `.pkl` file stores the complete trained scikit-learn pipeline:

```text
TfidfVectorizer + LogisticRegression
```

This means the API can load the model and make predictions without retraining it every time.

## Run the Backend Locally

Go to the backend folder:

```bash
cd backend
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn app.main:app --reload
```

The backend will run at:

```text
http://127.0.0.1:8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

## API Endpoints

### Health Check

```http
GET /health
```

Response:

```json
{
  "message": "Spam Mail Classifier API is running"
}
```

### Predict Message

```http
POST /predict
```

Request body:

```json
{
  "message": "Congratulations! You won a free prize. Click now!"
}
```

Response:

```json
{
  "prediction": "spam"
}
```

Normal message example:

```json
{
  "message": "Can you send me the homework file?"
}
```

Response:

```json
{
  "prediction": "ham"
}
```

### Validation Error Example

Request body:

```json
{
  "message": ""
}
```

Response status:

```text
422 Unprocessable Entity
```

## Frontend

The frontend is a simple HTML/CSS/JavaScript interface.

It allows the user to:

1. Enter a message
2. Click the classify button
3. See whether the message is `spam` or `ham`

When running with Docker Compose, the frontend is available at:

```text
http://127.0.0.1:8080
```

The frontend sends requests to:

```text
http://127.0.0.1:8000/predict
```

## Run with Docker Compose

From the project root:

```bash
docker compose up --build
```

Run in detached mode:

```bash
docker compose up --build -d
```

Check running containers:

```bash
docker compose ps
```

Stop the application:

```bash
docker compose down
```

Application URLs:

| Service | URL |
| --- | --- |
| Frontend | `http://127.0.0.1:8080` |
| Backend API | `http://127.0.0.1:8000` |
| Swagger UI | `http://127.0.0.1:8000/docs` |
| Health Check | `http://127.0.0.1:8000/health` |

## Running Tests

Go to the backend folder:

```bash
cd backend
```

Run tests:

```bash
python -m pytest -v
```

The tests cover:

- Health check endpoint
- Spam message prediction
- Ham message prediction
- Empty message validation
- Missing message field validation

## Continuous Integration

This project uses GitHub Actions for CI.

The workflow runs automatically on:

- Pushes to the `main` branch
- Pull requests targeting the `main` branch

The CI workflow:

1. Checks out the repository
2. Sets up Python 3.12
3. Installs backend dependencies
4. Runs backend tests
5. Builds the backend Docker image
6. Builds the frontend Docker image

Workflow file:

```text
.github/workflows/ci.yml
```

## Example cURL Request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Congratulations! You won a free prize. Click now!"
  }'
```

## Notes

This project is built for learning purposes.

The main goal is to understand the workflow of:

```text
Text Data → TF-IDF → Classification Model → Model Saving → FastAPI Serving → Frontend → Docker → Testing → CI
```

## Future Improvements

- Add prediction confidence scores
- Add more advanced preprocessing
- Compare Logistic Regression with Naive Bayes and LinearSVC
- Add Kubernetes manifests
- Add logging and monitoring
- Deploy the application to a cloud platform
