from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
model = load_model("model/pneumonia_model.keras")

def preprocess_image(image_bytes):
     image = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # 🟢 彩色图像
     image = image.resize((150, 150))
     image = np.array(image) / 255.0
     image = image.reshape(1, 150, 150, 3)                   
     return image

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    image_bytes = file.read()
    processed = preprocess_image(image_bytes)

    prediction = model.predict(processed)
    proba = float(prediction[0][0])
    return jsonify({
        "is_pneumonia": "Yes" if proba > 0.5 else "No",
        "confidence": f"{round(proba * 100, 1)}%"     
    })

if __name__ == "__main__":
    app.run(debug=True)
