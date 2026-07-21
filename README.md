# 🚢 Titanic Survival Prediction MLOps Project

An end-to-end Machine Learning Operations (MLOps) project that predicts whether a passenger would survive the Titanic disaster using a **Random Forest Classifier**. The project demonstrates the complete ML lifecycle—from data preprocessing and model training to serving predictions through a FastAPI REST API, containerizing the application with Docker, and publishing the project on GitHub and Docker Hub.

---

## 📌 Project Overview

This project covers the complete workflow of building and deploying a machine learning model in a production-ready manner.

### Features

* Data preprocessing
* Missing value handling
* Feature engineering
* Random Forest Classification
* Model serialization using Joblib
* FastAPI REST API
* Interactive Swagger UI
* Docker containerization
* Git version control
* GitHub repository
* Docker Hub image

---

## 🛠️ Tech Stack

* Python 3.12
* Pandas
* NumPy
* Scikit-learn
* FastAPI
* Uvicorn
* Joblib
* Docker
* Git
* GitHub

---

## 📂 Project Structure

```text
titanic-survival-mlops/
│
├── app/
│   └── app.py
│
├── data/
│
├── model/
│   ├── train.py
│   └── titanic_model.pkl
│
├── notebooks/
│
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Siri-Trivya/titanic-survival-mlops.git

cd titanic-survival-mlops
```

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate it:

Linux/macOS

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🧠 Train the Model

```bash
python model/train.py
```

The trained model will be saved as:

```text
model/titanic_model.pkl
```

---

## 🚀 Run the API

```bash
uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload
```

API URL:

```
http://localhost:8000
```

Swagger Documentation:

```
http://localhost:8000/docs
```

---

## 📡 API Endpoints

### Home

```
GET /
```

Response

```json
{
  "message": "Titanic Survival Prediction API is running",
  "model": "Random Forest",
  "version": "1.0.0"
}
```

---

### Health Check

```
GET /health
```

Response

```json
{
  "status": "Healthy"
}
```

---

### Predict Survival

```
POST /predict
```

Sample Request

```json
{
  "pclass": 3,
  "sex": "male",
  "age": 22,
  "sibsp": 1,
  "parch": 0,
  "fare": 7.25,
  "embarked": "S"
}
```

Sample Response

```json
{
  "prediction": "Did Not Survive"
}
```

---

## 🐳 Docker

### Build Image

```bash
docker build -t titanic-survival-mlops:v1 .
```

### Run Container

```bash
docker run -d -p 8000:8000 --name titanic-api titanic-survival-mlops:v1
```

---

## ☁️ Docker Hub

```text
docker pull siri987/titanic-survival-mlops:v1
```

---

## 📈 Machine Learning Workflow

1. Load Titanic Dataset
2. Data Cleaning
3. Feature Selection
4. Data Preprocessing
5. Train/Test Split
6. Random Forest Model Training
7. Model Evaluation
8. Save Model
9. FastAPI Prediction Service
10. Docker Containerization
11. GitHub Version Control
12. Docker Hub Deployment

---

## 🔮 Future Enhancements

* CI/CD using Jenkins or GitHub Actions
* Kubernetes deployment
* Model monitoring
* MLflow experiment tracking
* Automated model retraining
* Unit and integration testing
* Cloud deployment on AWS

---

## 👩‍💻 Author

**Sireesha Buddha**

GitHub: https://github.com/Siri-Trivya

Docker Hub: https://hub.docker.com/u/siri987

---

## ⭐ If you found this project useful, consider giving it a star on GitHub!
