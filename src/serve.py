"""FastAPI service for CIFAR-10 predictions."""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

from src.dataset import CIFAR10_CLASSES, get_transforms
from src.model import get_model

MODEL: torch.nn.Module | None = None
CLASS_NAMES: list[str] = CIFAR10_CLASSES
LOADED_MODEL_PATH: Path | None = None
INFERENCE_TRANSFORM = get_transforms(train=False)


def resolve_model_path() -> Path:
    """Locate the mounted, environment-provided, or local checkpoint."""

    configured = os.getenv("MODEL_PATH")
    candidates = [
        Path(configured) if configured else None,
        Path("/app/checkpoints/classifier_v1.pt"),
        Path("checkpoints/classifier_v1.pt"),
    ]

    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Model checkpoint not found. Train the model first or set MODEL_PATH."
    )


def load_model(path: Path) -> tuple[torch.nn.Module, list[str]]:
    """Load a trusted local model checkpoint."""

    checkpoint: dict[str, Any] = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )
    model = get_model(
        str(checkpoint["architecture"]),
        int(checkpoint["num_classes"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    class_names = list(checkpoint.get("class_names", CIFAR10_CLASSES))
    return model, class_names


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the model once when the application starts."""

    global MODEL, CLASS_NAMES, LOADED_MODEL_PATH
    LOADED_MODEL_PATH = resolve_model_path()
    MODEL, CLASS_NAMES = load_model(LOADED_MODEL_PATH)
    yield
    MODEL = None


app = FastAPI(
    title="CIFAR-10 Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    """Return 200 only when the model is loaded."""

    if MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded.",
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_path": str(LOADED_MODEL_PATH),
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict[str, Any]:
    """Return the predicted class and all class probabilities."""

    if MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded.",
        )

    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Upload an image file.",
        )

    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is not a valid image.",
        ) from None

    input_tensor = INFERENCE_TRANSFORM(pil_image).unsqueeze(0)

    with torch.inference_mode():
        logits = MODEL(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_index = int(probabilities.argmax().item())
    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "predicted_index": predicted_index,
        "confidence": round(float(probabilities[predicted_index]), 6),
        "probabilities": {
            class_name: round(float(probability), 6)
            for class_name, probability in zip(CLASS_NAMES, probabilities)
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.serve:app", host="0.0.0.0", port=8080)
