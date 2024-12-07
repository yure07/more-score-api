from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import gdown
import httpx
import gc

url = "https://drive.google.com/drive/folders/1yXytE9ozUThCTdGawYve8h2XGLD_C9Wh"

output_dir = "./model"
gdown.download_folder(url, output=output_dir, quiet=False, use_cookies=False)

# Inicializa o modelo e o tokenizer
path_model = "./model"
model = AutoModelForSequenceClassification.from_pretrained(path_model)
tokenizer = AutoTokenizer.from_pretrained(path_model)
model.eval()

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

def batch_texts(texts, batch_size=10):
    for i in range(0, len(texts), batch_size):
        yield texts[i:i + batch_size]

# Função para prever emoções de uma lista de textos
def predict_emotions_batch(texts):
    for batch in batch_texts(texts):
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True)
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
    
    del inputs
    del outputs
    gc.collect()
    torch.cuda.empty_cache()
    
    return results

# Função para obter comentários de um video no youtube
async def get_comments(api_key, video_id):
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": 50,
        "key": api_key
    }

    try:
        # Cria um cliente assíncrono
        async with httpx.AsyncClient() as client:
            # Faz a requisição GET
            response = await client.get(url, params=params)
            response.raise_for_status()  # Gera uma exceção em caso de erro HTTP
            data = response.json()  # Decodifica o JSON da resposta

        # Extrai os comentários
        comments = [
            item['snippet']['topLevelComment']['snippet']['textDisplay']
            for item in data.get('items', [])
        ]

        return comments

    except httpx.HTTPStatusError as e:
        print(f"Erro HTTP: {e.response.status_code}, {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail="Erro na API do YouTube")
    except Exception as e:
        print(f"Erro inesperado: {e}")
        raise HTTPException(status_code=500, detail="Erro ao processar a requisição")
    
# Nova rota combinada para obter comentários e prever emoções
@app.post("/process_video")
async def process_video(request: YoutubeRequest):
    try:
        # Obter comentários do vídeo
        comments = await get_comments(request.api_key, request.video_id)
        
        if not comments:
            raise HTTPException(status_code=404, detail="Nenhum comentário encontrado para este vídeo.")
        
        # Prever emoções dos comentários
        results = predict_emotions_batch(comments)
        
        # Retornar a resposta combinada
        return {"comentarios": comments, "emocao_analise": results}
    
    except Exception as e:
        print(f"Erro no endpoint /process_video: {e}")
        raise HTTPException(status_code=500, detail="Erro ao processar a requisição")
