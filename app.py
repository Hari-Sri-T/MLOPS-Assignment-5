"""Flask web app: upload an image, get a cat/dog prediction.

Routes
  GET  /          upload page
  POST /predict   multipart form field "file" -> JSON prediction
  GET  /health    liveness/readiness probe used by Docker and CI/CD
"""
import os
import socket

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

from src.predictor import Predictor

ALLOWED_EXT = {"jpg", "jpeg", "png", "bmp", "gif", "webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload limit
predictor = Predictor()

APP_VERSION = os.getenv("APP_VERSION", "dev")


def allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@app.get("/")
def index():
    return render_template("index.html", version=APP_VERSION, host=socket.gethostname())


@app.get("/health")
def health():
    try:
        ready = predictor.ready
    except Exception:
        app.logger.exception("Model could not be loaded")
        ready = False
    return jsonify(status="ok" if ready else "unavailable", model_ready=ready,
                   version=APP_VERSION), 200 if ready else 503


@app.post("/predict")
def predict():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify(error="No file uploaded (form field 'file')."), 400
    if not allowed(file.filename):
        return jsonify(error=f"Unsupported file type. Use one of: {sorted(ALLOWED_EXT)}"), 400
    try:
        ready = predictor.ready
    except Exception:
        app.logger.exception("Model could not be loaded")
        ready = False
    if not ready:
        return jsonify(error="Model is unavailable. Please try again later."), 503
    try:
        result = predictor.predict(file.read())
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError):
        return jsonify(error="Could not process image. Upload a valid image."), 422
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify(error="Prediction is unavailable. Please try again later."), 503
    return jsonify(result)


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="File too large (max 8 MB)."), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)), debug=False)
