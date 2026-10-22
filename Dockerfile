# ---------- Image Classification app (Flask + TensorFlow CPU) ----------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_NUM_INTRAOP_THREADS=2 \
    TF_NUM_INTEROP_THREADS=1 \
    OMP_NUM_THREADS=2 \
    PORT=8080

WORKDIR /app

# Install dependencies first so this layer is cached between code changes
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Application code
COPY src/ src/
COPY templates/ templates/
COPY app.py params.yaml ./

# Trained model pulled from S3 by `dvc pull` in the CI job before `docker build`
COPY artifacts/model/ artifacts/model/

# Run as an unprivileged user
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

ARG APP_VERSION=dev
ENV APP_VERSION=${APP_VERSION}

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8080/health').status==200 else 1)"

# A single worker avoids loading a separate model in each process.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "4", "--timeout", "120", "app:app"]
