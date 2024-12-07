# Usar a imagem base do Python
FROM python:3.10-slim

# Definir o diretório de trabalho
WORKDIR /app

# Copiar os arquivos necessários
COPY . .

# Instalar as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Expor a porta da aplicação
EXPOSE 8000

# Comando para iniciar o servidor
CMD python -m uvicorn api:app --host 0.0.0.0 --port 8000 --limit-max-requests 100
