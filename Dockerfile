FROM python:3.11-slim

# Install system dependencies & Ollama
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

# Copy application source code
COPY . /app

# Set port for Hugging Face Spaces (default 7860)
ENV PORT=7860
EXPOSE 7860

# Start Ollama service, pull models, and launch My-LLM server
CMD ["sh", "-c", "ollama serve & sleep 4 && ollama pull nomic-embed-text && ollama pull llama3.2:1b && python main.py"]
