from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
import gdown

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

# Classe de modelo para receber os textos na requisição
class TextRequest(BaseModel):
    texts: list[str]  # Lista de textos para análise

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
def get_comments(api_key, video_id):
    url = f"https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": 100,  # Número máximo de comentários por requisição
        "key": api_key
    }

    comments = []
    while True:
        try:
            # Faz a requisição para a API
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            # Extrai os comentários
            for item in data['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)

            # Verifica se há mais páginas de resultados
            next_page_token = data.get('nextPageToken')
            if next_page_token:
                # Se houver, faz a próxima requisição com o token da próxima página
                params['pageToken'] = next_page_token
            else:
                # Se não houver mais páginas, encerra o loop
                break

        except requests.exceptions.RequestException as e:
            print(f"Erro na requisição: {e}")
            break

    return comments

# Endpoint para obter comentários do video youtube
@app.post("/get_comments")
async def get_comments_endpoint(request: YoutubeRequest):
    comments = get_comments(request.api_key, request.video_id)
    return comments
