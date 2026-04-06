# Stage 1: Builder (instala dependências)
FROM python:3.10-slim as builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime (imagem final leve)
FROM python:3.10-slim
WORKDIR /app
ENV PATH=/root/.local/bin:$PATH
COPY --from=builder /root/.local /root/.local
COPY api/ ./api/
EXPOSE $PORT
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "10000"]
