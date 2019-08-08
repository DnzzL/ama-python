from allennlp.predictors.predictor import Predictor
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Request(BaseModel):
    passage: str
    question: str


class Answer(BaseModel):
    content: str


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/qa")
async def ask_question(request: Request):
    predictor = Predictor.from_path(
        "https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz")
    prediction = predictor.predict(
        passage=request.passage,
        question=request.question
    )
    answer = prediction.get("best_span_str")
    return answer
