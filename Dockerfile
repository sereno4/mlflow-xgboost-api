# Multi-stage build para imagem otimizada
FROM python:3.10-slim as builder

WORKDIR /app

# Instalar dependências de build
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage final
FROM python:3.10-slim

WORKDIR /app

# Copiar dependências do builder
COPY --from=builder /root/.local /root/.local

# Garantir que os scripts estão no PATH
ENV PATH=/root/.local/bin:$PATH

# Copiar código da API
COPY api/ ./api/
COPY mlflow.db ./mlflow.db 2>/dev/null || true

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_TRACKING_URI=http://localhost:5000 \
    MODEL_RUN_ID=3a0452332cc74f2d978ed336782764ef

# Expor porta
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Comando de inicialização
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
