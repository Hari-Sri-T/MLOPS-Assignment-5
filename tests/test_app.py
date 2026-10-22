"""Fast tests that run in CI without the trained model or TensorFlow inference."""
import io

import pytest
from PIL import Image

import app as app_module


@pytest.fixture
def client(monkeypatch, tmp_path):
    from src.predictor import Predictor

    monkeypatch.setattr(app_module, "predictor", Predictor(
        model_path=str(tmp_path / "missing.keras"),
        classes_path=str(tmp_path / "missing.json"),
    ))
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


def _png_bytes(color=(200, 120, 40)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, format="PNG")
    return buf.getvalue()


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"Cats vs Dogs Classifier" in r.data


def test_health_without_model(client):
    r = client.get("/health")
    assert r.status_code == 503
    assert r.get_json()["model_ready"] is False


def test_health_with_loaded_model(client, monkeypatch):
    monkeypatch.setattr(app_module.predictor, "_load", lambda: object())
    monkeypatch.setattr("src.predictor.os.path.isfile", lambda _: True)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["model_ready"] is True


def test_health_rejects_broken_model(client, monkeypatch):
    def fail():
        raise ValueError("broken model")

    monkeypatch.setattr(app_module.predictor, "_load", fail)
    monkeypatch.setattr("src.predictor.os.path.isfile", lambda _: True)
    assert client.get("/health").status_code == 503


def test_predict_requires_file(client):
    r = client.post("/predict", data={})
    assert r.status_code == 400


def test_predict_rejects_bad_extension(client):
    data = {"file": (io.BytesIO(b"hello"), "notes.txt")}
    r = client.post("/predict", data=data, content_type="multipart/form-data")
    assert r.status_code == 400


def test_predict_without_model(client):
    data = {"file": (io.BytesIO(_png_bytes()), "pet.png")}
    assert client.post("/predict", data=data).status_code == 503


def test_predict_rejects_corrupt_image(client, monkeypatch):
    from src.predictor import Predictor

    class LoadedPredictor(Predictor):
        @property
        def ready(self):
            return True

        def _load(self):
            return object()

    monkeypatch.setattr(app_module, "predictor", LoadedPredictor())
    r = client.post("/predict", data={"file": (io.BytesIO(b"not an image"), "bad.png")})
    assert r.status_code == 422


def test_predict_rejects_oversized_upload(client):
    r = client.post("/predict", data={"file": (io.BytesIO(b"x" * (8 * 1024 * 1024)), "big.png")})
    assert r.status_code == 413


def test_ready_requires_class_metadata(tmp_path):
    from src.predictor import Predictor

    model = tmp_path / "model.keras"
    model.touch()
    assert Predictor(str(model), str(tmp_path / "missing.json")).ready is False


def test_predict_with_stub_model(client, monkeypatch):
    class StubPredictor:
        ready = True

        def predict(self, _bytes):
            return {"label": "dogs", "confidence": 0.93,
                    "probabilities": {"cats": 0.07, "dogs": 0.93}}

    monkeypatch.setattr(app_module, "predictor", StubPredictor())
    data = {"file": (io.BytesIO(_png_bytes()), "pet.png")}
    r = client.post("/predict", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    body = r.get_json()
    assert body["label"] == "dogs"
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-6


def test_preprocess_shape():
    from src.predictor import Predictor

    p = Predictor(model_path="missing.keras")
    x = p.preprocess(_png_bytes())
    assert x.shape == (1, 160, 160, 3)


def test_predictor_probabilities(monkeypatch):
    import numpy as np
    from src.predictor import Predictor

    p = Predictor(model_path="missing.keras")

    def model(x, training):
        assert training is False
        assert x.shape == (1, 160, 160, 3)
        return np.array([[0.8]])

    monkeypatch.setattr(p, "_load", lambda: model)
    result = p.predict(_png_bytes())
    assert result == {"label": "dogs", "confidence": 0.8,
                      "probabilities": {"cats": 0.2, "dogs": 0.8}}
