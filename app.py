import os
import torch
import cv2
import logging
import boto3

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from signLanguage.pipeline.training_pipeline import TrainPipeline
from signLanguage.utils.main_utils import decodeImage, encodeImageIntoBase64

app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f"Device being used: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU only'}")

S3_BUCKET_NAME = "sign-lang-detection-basant"
S3_MODEL_KEY = "best.pt"
LOCAL_MODEL_PATH = "artifacts/training/best.pt"

def download_model_from_s3():
    if not os.path.exists(LOCAL_MODEL_PATH):
        logging.info("Model not found locally. Downloading from S3...")
        os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)
        s3 = boto3.client("s3")
        try:
            s3.download_file(S3_BUCKET_NAME, S3_MODEL_KEY, LOCAL_MODEL_PATH)
            logging.info(f"Model downloaded from S3 to {LOCAL_MODEL_PATH}")
        except Exception as e:
            logging.error(f"Failed to download model from S3: {e}")
            raise

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template("index.html")

@app.route("/train", methods=["GET"])
@cross_origin()
def trainRoute():
    try:
        logging.info("Training started...")
        pipeline = TrainPipeline()
        pipeline.run_pipeline()
        logging.info("Training completed successfully.")
        return jsonify({"message": "Training completed successfully!"}), 200
    except Exception as e:
        logging.error(f"Training error: {str(e)}")
        return jsonify({"message": f"Training failed: {str(e)}"}), 500

@app.route("/predict", methods=["POST"])
@cross_origin()
def predictRoute():
    try:
        # Step 1: Decode and save image
        image = request.json['image']
        decodeImage(image, "inputImage.jpg")

        # Step 2: Download model if not present
        download_model_from_s3()

        # Step 3: Load model
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=LOCAL_MODEL_PATH, force_reload=True)
        model.to(device)

        # Step 4: Predict
        img = cv2.imread("inputImage.jpg")
        results = model(img)
        results.render()

        # Step 5: Save and encode result
        output_path = os.path.join("static", "predicted_image.jpg")
        cv2.imwrite(output_path, results.ims[0])
        encoded_img = encodeImageIntoBase64(output_path)

        return jsonify({
            "image": encoded_img.decode(),
            "message": "Prediction done"
        })
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({"message": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    print("Starting Flask App...")
    app.run(host="127.0.0.1", port=8080, debug=True, use_reloader=False)
