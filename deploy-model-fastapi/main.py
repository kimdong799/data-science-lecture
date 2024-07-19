from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

#load model
model = joblib.load('./model/my_diabetes_predict_model.pkl')

class DiabetesPredictApplication(BaseModel):
  age: float
  sex: float
  bmi: float
  bp: float
  s1: float
  s2: float
  s3: float
  s4: float
  s5: float
  s6: float
  
class DiabetesPredictOutput(BaseModel):
  predicted_disease_progression : float

@app.get("/")
async def root():
  return {"message": "Hello World!"}

@app.post("/predict/diabetes", response_model=DiabetesPredictOutput)
def predict_diabetes(application: DiabetesPredictApplication):
  print("****************************************************")
  print(application)
  input_arr = np.array([[application.age, application.sex, application.bmi, application.bp, application.s1, application.s2, application.s3, application.s4, application.s5, application.s6]])

  prediction = model.predict(input_arr)
  print("****************************************************")
  print(prediction)
  return {"predicted_disease_progression": prediction[0]}