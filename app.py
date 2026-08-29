"""Flask web app: type a seed phrase, get an LSTM-generated continuation.
Also exposes /train to kick off the full training pipeline.

Run locally:
    python app.py
Then open http://localhost:5000 in your browser.
"""


import os

from flask import Flask, render_template, request

from src.nextWordPredictor.config.configuration import ConfigurationManager
from src.nextWordPredictor.logging import logger

app = Flask(__name__, template_folder=".", static_folder="static")

_prediction_pipeline = None
_params = ConfigurationManager().params.InferenceParams


def get_prediction_pipeline():
    """Lazy-load the PredictionPipeline (and the trained model) once,
    on first request, and cache it."""
    global _prediction_pipeline
    if _prediction_pipeline is None:
        from src.nextWordPredictor.pipeline.prediction import PredictionPipeline
        _prediction_pipeline = PredictionPipeline()
    return _prediction_pipeline


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        seed_text=None,
        generated=None,
        error=None,
        default_num_words=_params.default_num_words,
        max_num_words=_params.max_num_words,
    )


@app.route("/train", methods=["GET"])
def train():
    """Trigger the full training pipeline (ingestion through evaluation)
    by running main.py."""
    os.system("python main.py")
    return "Training pipeline completed successfully!"


@app.route("/predict", methods=["POST"])
def predict():
    seed_text = request.form.get("seed_text", "").strip()
    num_words = request.form.get("num_words", _params.default_num_words)
    creative = request.form.get("creative") == "on"

    try:
        num_words = int(num_words)
    except ValueError:
        num_words = _params.default_num_words
    num_words = max(1, min(num_words, _params.max_num_words))

    if not seed_text:
        return render_template(
            "index.html",
            seed_text=None,
            generated=None,
            error="Please enter a starting phrase first.",
            default_num_words=_params.default_num_words,
            max_num_words=_params.max_num_words,
        )

    try:
        pipeline = get_prediction_pipeline()
    except Exception as e:
        logger.exception(e)
        return render_template(
            "index.html",
            seed_text=seed_text,
            generated=None,
            error=(
                "No trained model found. Train it first by visiting /train "
                "or running: python main.py"
            ),
            default_num_words=_params.default_num_words,
            max_num_words=_params.max_num_words,
        )

    temperature = 0.7 if creative else 0.0
    generated = pipeline.generate(seed_text, num_words=num_words, temperature=temperature)
    continuation = generated[len(seed_text):]

    return render_template(
        "index.html",
        seed_text=seed_text,
        generated=generated,
        continuation=continuation,
        error=None,
        default_num_words=num_words,
        max_num_words=_params.max_num_words,
    )


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
