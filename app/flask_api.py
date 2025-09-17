import os
import io
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Limit upload size to 6 MB
app.config['MAX_CONTENT_LENGTH'] = 6 * 1024 * 1024

# Locate model and class names (adjust paths if your repo layout differs)
POSSIBLE_MODEL_PATHS = [
    "model/emotion_model.pt",
    "emotion_model.pt",
]
POSSIBLE_CLASS_PATHS = [
    "model/class_names.txt",
    "class_names.txt",
]

def find_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

MODEL_PATH = find_existing(POSSIBLE_MODEL_PATHS)
CLASS_PATH = find_existing(POSSIBLE_CLASS_PATHS)

if MODEL_PATH is None:
    raise FileNotFoundError("Model file not found. Expected one of: " + ", ".join(POSSIBLE_MODEL_PATHS))
if CLASS_PATH is None:
    raise FileNotFoundError("class_names.txt not found. Expected one of: " + ", ".join(POSSIBLE_CLASS_PATHS))

with open(CLASS_PATH, "r") as f:
    CLASS_NAMES = [line.strip() for line in f.readlines() if line.strip()]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build the same architecture used during training
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Same transforms used during training
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_image_bytes(img_bytes):
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        top_idx = int(probs.argmax())
        result = {
            "label": CLASS_NAMES[top_idx],
            "confidence": float(probs[top_idx]),
            "all": { CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES)) }
        }
    return result

@app.route("/", methods=["GET"])
def home():
    return "hi", 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok", "model_path": os.path.basename(MODEL_PATH)}), 200

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error":"No image file part in the request. Use multipart/form-data with key 'image'."}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error":"Empty filename"}), 400
    try:
        img_bytes = file.read()
        res = predict_image_bytes(img_bytes)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error":"prediction failed", "details": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
