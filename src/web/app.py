import uvicorn
from fastapi import FastAPI

from src.train.predict import Predictor
from src.web.schemes import Category, Item
from src.web.service import TitleService

app = FastAPI()

predictor = Predictor()  # Initialize your predictor here

service = TitleService(predictor)

@app.post("/predict")
def predict(item: Item) -> Category:
    response = Category(category=service.predict(item.description or ""))
    return response

def serve():
    uvicorn.run("src.web.app:app", host="0.0.0.0", port=8000)