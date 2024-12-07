from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import gdown
import httpx

url = "https://drive.google.com/drive/folders/1yXytE9ozUThCTdGawYve8h2XGLD_C9Wh"

output_dir = "./model"
gdown.download_folder(url, output=output_dir, quiet=False, use_cookies=False)

# Inicializa o modelo e o tokenizer
path_model = "./model"
model = AutoModelForSequenceClassification.from_pretrained(path_model)
tokenizer = AutoTokenizer.from_pretrained(path_model)

# Labels de emoções
emotion_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust",
    "embarrassment", "excitement", "fear", "gratitude", "grief", "joy",
    "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]

# Inicializa o FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

class TextRequest(BaseModel):
    texts: list[str]  

class YoutubeRequest(BaseModel):
    api_key: str  
    video_id: str 

# Função para prever emoções de uma lista de textos
def predict_emotions_batch(texts):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    probs = torch.sigmoid(logits).cpu().numpy()
    threshold = 0.5

    results = []
    for text, prob_array in zip(texts, probs):
        predicted_emotions = [i for i, prob in enumerate(prob_array) if prob >= threshold]
        emotion_names = [emotion_labels[idx] for idx in predicted_emotions]
        results.append({"texto": text, "emocao_prevista": emotion_names})
    
    return results

# Endpoint para prever emoções
@app.post("/predict_emotions")
async def predict_emotions_endpoint(request: TextRequest):
    texts = request.texts
    results = predict_emotions_batch(texts)
    return results

# Função para obter comentários de um video no youtube
async def get_comments(api_key, video_id):
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": 25,
        "key": api_key
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
    
    comments = [
        item['snippet']['topLevelComment']['snippet']['textDisplay']
        for item in data.get('items', [])
    ]
    
    return comments

# Endpoint para obter comentários do video youtube
@app.post("/get_comments")
async def get_comments_endpoint(request: YoutubeRequest):
    try:
        comments = get_comments(request.api_key, request.video_id)
        return comments
    except Exception as e:
        print(f"Erro no endpoint /get_comments: {e}")
        raise HTTPException(status_code=500, detail="Erro ao processar a requisição")
