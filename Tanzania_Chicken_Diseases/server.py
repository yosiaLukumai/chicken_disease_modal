from flask import Flask, request, jsonify
from ultralytics import YOLO  
from PIL import Image
import io

app = Flask(__name__)

model = YOLO('best.pt')  

@app.route('/')
def home():
    return "Chicken Disease Detection Model is running!"

@app.route('/predict', methods=['POST'])
def predict():
    print("Hitted /predict")
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Read image
        img = Image.open(file.stream)

        results = model(img)

        predictions = []
        for box in results[0].boxes:
            prediction = {
                "label": int(box.cls[0].item()),  
                "confidence": float(box.conf[0].item())  
            }
            predictions.append(prediction)

        print(predictions)
        return jsonify({
            "predictions": predictions
        })

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

