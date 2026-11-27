# ---- React console -------------------------------------------------------
FROM oven/bun:1 AS web
WORKDIR /web
COPY web/package.json ./
RUN bun install
COPY web ./
RUN bun run build

# ---- API ----------------------------------------------------------------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
COPY src ./src
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install --no-deps -e .

COPY --from=web /web/dist ./web/dist

# Generate the demo dataset, train the models and seed the database at build
# time. Doing it here instead of at container start means a cold start only has
# to load the model: hosting tiers with a fraction of a CPU (Render's free
# 0.1 CPU instance, for example) would otherwise spend minutes training and
# scoring SHAP explanations before the port ever opens.
RUN python -m fraud_platform.cli setup --rows 2500

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=5 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8000}/health" || exit 1

# Honour $PORT so platform hosters (Render, Railway, Cloud Run) can pick the
# port; Docker Compose and `docker run -p 8000:8000` fall back to 8000.
CMD ["sh", "-c", "exec uvicorn fraud_platform.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]


