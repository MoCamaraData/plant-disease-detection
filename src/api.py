# api.py

import os
import io
import sys
import traceback
from typing import Optional, List

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# -------------------------------------------
# Trick so torch.load(model) from notebook works
# (notebook pickled __main__.ResNet9 / ImageClassificationBase)
# -------------------------------------------
sys.modules["__main__"] = sys.modules[__name__]

# -------------------------------------------
# Model definitions (copied from your notebook)
# -------------------------------------------

# base class for the model
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                      # Generate predictions
        loss = F.cross_entropy(out, labels)     # Calculate loss
        _, preds = torch.max(out, dim=1)        # Get the predictions
        acc = torch.tensor(torch.sum(preds == labels).item() / len(preds))
        return {"val_loss": loss.detach(), "val_accuracy": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_accuracy = [x["val_accuracy"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()       # Combine loss
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy}

    def epoch_end(self, epoch, result):
        # not needed for inference, just here for pickle compatibility
        pass


# convolution block with BatchNormalization
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        # 4 in your notebook; keep same to match saved model
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


# ResNet9 architecture (exact same structure as notebook)
class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(
            ConvBlock(128, 128),
            ConvBlock(128, 128),
        )
        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(
            ConvBlock(512, 512),
            ConvBlock(512, 512),
        )

        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, num_diseases),
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


# -------------------------------------------
# Config
# -------------------------------------------

# 👉 This is the BEST MODEL you saved in the notebook for ResNet9
MODEL_PATH = os.getenv("MODEL_PATH", "models\plant-disease-model.pth")

DEVICE = torch.device("cpu")  # keep it simple

# If you want readable labels later, put train.classes here
CLASS_NAMES: List[str] = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]  


# Match your test_tf in the notebook (Resize → ToTensor)
TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# -------------------------------------------
# FastAPI app
# -------------------------------------------

app = FastAPI(
    title="Plant Disease Prediction API (ResNet9)",
    description="API pour prédire la maladie d'une plante à partir d'une image de feuille (meilleur modèle ResNet9).",
    version="1.0.0",
)

model: Optional[nn.Module] = None


def load_model(path: str) -> nn.Module:
    """
    Charge un ResNet9 sauvegardé avec:
        torch.save(model.state_dict(), PATH)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier modèle introuvable: {path}")

    # Charger les poids (state_dict)
    state_dict = torch.load(path, map_location=DEVICE)

    # nombre de classes -> adapte si nécessaire (ex: 38 classes)
    num_classes = len(CLASS_NAMES) if CLASS_NAMES else 38

    # reconstruire l'architecture EXACTE utilisée au training
    net = ResNet9(in_channels=3, num_diseases=num_classes)
    net.load_state_dict(state_dict)

    net.to(DEVICE)
    net.eval()
    return net



@app.on_event("startup")
def startup_event():
    global model
    try:
        model = load_model(MODEL_PATH)
        print(f"✅ Modèle ResNet9 chargé depuis {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {e}")
        traceback.print_exc()
        model = None


@app.get("/health")
def health():
    """Vérifier le statut de l'API et le chargement du modèle."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH if model is not None else "N/A",
        "device": str(DEVICE),
    }


def preprocess_image(file_bytes: bytes) -> torch.Tensor:
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Fichier image invalide.")

    img_t = TRANSFORM(img)          # (C, H, W)
    img_t = img_t.unsqueeze(0)      # (1, C, H, W)
    return img_t.to(DEVICE)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Prédire la maladie à partir d'une image de feuille.

    - Content-Type: multipart/form-data
    - Champ: file (UploadFile, image)
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé. Vérifiez MODEL_PATH et les logs du serveur."
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image.")

    try:
        file_bytes = await file.read()
        img_t = preprocess_image(file_bytes)

        with torch.no_grad():
            outputs = model(img_t)
            probs = F.softmax(outputs, dim=1)[0]
            pred_index = int(torch.argmax(probs).item())
            pred_prob = float(probs[pred_index].item())

            # top-3 for debug
            top_probs, top_indices = torch.topk(probs, k=min(3, probs.shape[0]))
            top_probs = top_probs.cpu().numpy().tolist()
            top_indices = top_indices.cpu().numpy().tolist()

        if CLASS_NAMES:
            pred_class = CLASS_NAMES[pred_index]
            top_classes = [CLASS_NAMES[i] for i in top_indices]
        else:
            pred_class = pred_index
            top_classes = top_indices

        return JSONResponse({
            "status": "success",
            "prediction": {
                "class": pred_class,
                "index": pred_index,
                "probability": pred_prob,
            },
            "top3": [
                {"class": str(c), "index": int(i), "probability": float(p)}
                for c, i, p in zip(top_classes, top_indices, top_probs)
            ]
        })

    except HTTPException:
        raise
    except Exception as e:
        print(f"Erreur lors de la prédiction : {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Erreur serveur : {str(e)}"
        )


@app.get("/")
def root():
    """Endpoint racine avec liens utiles."""
    return {
        "message": "API ResNet9 pour la détection de maladies des plantes",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict (multipart/form-data, champ: file)",
    }
