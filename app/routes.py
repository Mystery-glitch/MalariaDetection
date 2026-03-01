import torch
from torchvision import transforms
from PIL import Image
from flask import Blueprint, render_template, request
import os

from src.model_builder import get_model

main = Blueprint("main", __name__)

MODEL_PATH = "best_model.pth"
CLASS_NAMES = ["Parasitized", "Uninfected"]

# Load model once
model = get_model("resnet50")  # Change if your best model is different
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])


@main.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    image_path = None

    if request.method == "POST":

        file = request.files["image"]

        upload_folder = os.path.join("app", "static", "uploads")
        filepath = os.path.join(upload_folder, file.filename)
        file.save(filepath)

        image = Image.open(filepath).convert("RGB")
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image)
            _, pred = torch.max(outputs, 1)

        prediction = CLASS_NAMES[pred.item()]
        image_path = "static/uploads/" + file.filename

    return render_template("index.html",
                           prediction=prediction,
                           image_path=image_path)