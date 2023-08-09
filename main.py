import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from fastapi import Request, FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI()

origins = [
    "http://localhost:3000",
    "localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "API is working"}

def remove_special_characters(text):
    # Remove emojis (assuming they are in unicode format)
    text = re.sub(r'[^\u0000-\uFFFF]', '', text)
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text

#  Tokenization and removing stopwords
nltk.download('punkt')
nltk.download('stopwords')

def remove_stopwords_and_tokenize(text):
    stop_words = set(stopwords.words('nepali'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)

class AnalyzePayload(BaseModel):
    text: str

@app.post("/analyze")
async def predict(payload: Request):
    try:
        data = await payload.json()
        model = pickle.load(open("final_model.pkl", "rb"))

        text = data["text"]
        text = remove_special_characters(text)
        text = remove_stopwords_and_tokenize(text)

        tokenizer = Tokenizer()
        tokens = tokenizer.texts_to_sequences([text])
        padded_tokens = pad_sequences(tokens, maxlen=163, padding='pre', truncating='pre')
        predictions = model.predict(padded_tokens)
        predicted_class = predictions.argmax(axis=-1)[0]
        return {"sentiment": str(predicted_class), "confidence": str(predictions[0][predicted_class]), "status": 20}
    except Exception as e:
        print(e)
        return {"error": str(e)}

## uvicorn main:app --reload