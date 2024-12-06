from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import instaloader
import gdown
import os

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

class InstagramRequest(BaseModel):
    username: str  # Nome de usuário do Instagram
    password: str  # Senha do Instagram
    post_shortcode: str  # Shortcode do post no Instagram

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

# Função para obter comentários de um post no Instagram
def get_comments_instagram(username, password, post_shortcode):
    L = instaloader.Instaloader()
    L.context.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    session_file = f"./session-{username}"

    def login_and_save_session():
        try:
            L.login(username, password)
            L.save_session_to_file(filename=session_file)
        except Exception as e:
            raise HTTPException(status_code=401, detail=f"Erro ao fazer login: {e}")

    try:
        if os.path.exists(session_file):
            L.load_session_from_file(username, filename=session_file)
        else:
            login_and_save_session()
    except instaloader.exceptions.ConnectionException as e:
        login_and_save_session()

    # Validar sessão
    try:
        profile = instaloader.Profile.from_username(L.context, username)
        if not profile.is_followed_by:
            raise HTTPException(status_code=403, detail="Sessão inválida ou sem permissão.")
    except Exception as e:
        raise HTTPException(status_code=403, detail=f"Erro ao validar sessão: {e}")

    # Coletar post
    try:
        post = instaloader.Post.from_shortcode(L.context, post_shortcode)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Post não encontrado: {e}")

    # Coletar comentários
    comments = []
    try:
        for comment in post.get_comments():
            comments.append(comment.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao coletar comentários: {e}")

    return comments

# Endpoint para obter comentários do Instagram
@app.post("/get_comments")
async def get_comments_endpoint(request: InstagramRequest):
    comments = get_comments_instagram(request.username, request.password, request.post_shortcode)
    return comments
