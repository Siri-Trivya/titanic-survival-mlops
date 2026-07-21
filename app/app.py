from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load trained model
model = joblib.load("model/titanic_model.pkl")

app = FastAPI(
    title="Titanic Survival Prediction API",
    version="1.0.0",
    description="Predicts whether a passenger would survive the Titanic disaster."
)


class Passenger(BaseModel):
    pclass: int
    sex: str
    age: float
    sibsp: int
    parch: int
    fare: float
    embarked: str


@app.get("/")
def home():
    return {
        "message": "Titanic Survival Prediction API is running",
        "model": "Random Forest",
        "version": "1.0.0"
    }


@app.get("/health")
def health():
    return {"status": "Healthy"}


@app.post("/predict")
def predict(data: Passenger):
    sample = [[
        data.age,
        data.fare,
        data.sibsp,
        data.parch,
        data.pclass,
        data.sex,
        data.embarked
    ]]

    prediction = model.predict(sample)[0]

    return {
        "prediction": "Survived" if prediction == 1 else "Did Not Survive"
    }
