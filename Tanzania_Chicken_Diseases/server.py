# # # from flask import Flask, request, jsonify
# # # from flask_cors import CORS

# # # import torch
# # # from PIL import Image
# # # from io import BytesIO
# # # import base64
# # # import numpy as np
# # # import cv2



# # # # Initialize Flask app
# # # app = Flask(__name__)
# # # CORS(app)


# # # # Load YOLOv8 model
# # # model = torch.hub.load('ultralytics/yolov8', 'custom', path='best.pt')

# # # # Route for testing the server
# # # @app.route('/')
# # # def home():
# # #     return "YOLOv8 Flask Server is running!"

# # # # Route to handle image upload and inference
# # # @app.route('/predict', methods=['POST'])
# # # def predict():
# # #     try:
# # #         # Get the image from the request
# # #         if 'file' not in request.files:
# # #             return jsonify({"error": "No file part"}), 400
        
# # #         file = request.files['file']
        
# # #         if file.filename == '':
# # #             return jsonify({"error": "No selected file"}), 400
        
# # #         # Read image
# # #         img = Image.open(file.stream)

# # #         # Perform inference
# # #         results = model(img)

# # #         # Process results
# # #         img_array = np.array(results.ims[0])  # Convert to numpy array
# # #         ret, buffer = cv2.imencode('.jpg', img_array)
# # #         img_bytes = buffer.tobytes()
# # #         img_b64 = base64.b64encode(img_bytes).decode('utf-8')

# # #         return jsonify({
# # #             "prediction": results.pandas().xywh.to_dict(orient="records"),  # Add detailed predictions
# # #             "image": img_b64  # Return the processed image in base64
# # #         })

# # #     except Exception as e:
# # #         return jsonify({"error": str(e)}), 500

# # # if __name__ == '__main__':
# # #     app.run(debug=True, host='0.0.0.0', port=5000)



# # from flask import Flask, request, jsonify
# # from ultralytics import YOLO  # Import the YOLO class from ultralytics
# # from PIL import Image
# # import io
# # import base64
# # import numpy as np
# # import cv2

# # # Initialize Flask app
# # app = Flask(__name__)

# # # Load YOLOv8 model using the correct method
# # model = YOLO('best.pt')  # Your YOLOv8 model

# # # Route for testing the server
# # @app.route('/')
# # def home():
# #     return "Chicken Disease! Detection Model is running!"

# # # Route to handle image upload and inference
# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     print("Predicting")
# #     try:
# #         # Get the image from the request
# #         if 'file' not in request.files:
# #             return jsonify({"error": "No file part"}), 400
        
# #         file = request.files['file']
        
# #         if file.filename == '':
# #             return jsonify({"error": "No selected file"}), 400
        
# #         # Read image
# #         img = Image.open(file.stream)

# #         # Perform inference
# #         results = model(img)  # Perform inference with YOLOv8 model

# #         # Process results
# #         img_array = np.array(results.ims[0])  # Convert to numpy array
# #         ret, buffer = cv2.imencode('.jpg', img_array)
# #         img_bytes = buffer.tobytes()
# #         img_b64 = base64.b64encode(img_bytes).decode('utf-8')

# #         return jsonify({
# #             "prediction": results.pandas().xywh.to_dict(orient="records"),  # Add detailed predictions
# #             "image": img_b64  # Return the processed image in base64
# #         })

# #     except Exception as e:
# #         print(e)
# #         return jsonify({"error": str(e)}), 500

# # if __name__ == '__main__':
# #     app.run(debug=True, host='0.0.0.0', port=5000)



# # from flask import Flask, request, jsonify
# # from ultralytics import YOLO  # Import the YOLO class from ultralytics
# # from PIL import Image
# # import io
# # import base64
# # import numpy as np
# # import cv2

# # # Initialize Flask app
# # app = Flask(__name__)

# # # Load YOLOv8 model using the correct method
# # model = YOLO('best.pt')  # Your YOLOv8 model

# # # Route for testing the server
# # @app.route('/')
# # def home():
# #     return "Chicken Disease! Detection Model is running!"

# # # Route to handle image upload and inference
# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     print("Predicting")
# #     try:
# #         # Get the image from the request
# #         if 'file' not in request.files:
# #             return jsonify({"error": "No file part"}), 400
        
# #         file = request.files['file']
        
# #         if file.filename == '':
# #             return jsonify({"error": "No selected file"}), 400
        
# #         # Read image
# #         img = Image.open(file.stream)

# #         # Perform inference
# #         results = model(img)  # Perform inference with YOLOv8 model
        
# #         # Extract predictions from the result
# #         predictions = results.pandas().xywh[0].to_dict(orient="records")  # Get the predictions

# #         # Process image for return
# #         img_array = np.array(results.orig_img)  # Use the original image from the results
# #         ret, buffer = cv2.imencode('.jpg', img_array)
# #         img_bytes = buffer.tobytes()
# #         img_b64 = base64.b64encode(img_bytes).decode('utf-8')

# #         return jsonify({
# #             "prediction": predictions,  # Add detailed predictions
# #             "image": img_b64  # Return the processed image in base64
# #         })

# #     except Exception as e:
# #         print(e)
# #         return jsonify({"error": str(e)}), 500

# # if __name__ == '__main__':
# #     app.run(debug=True, host='0.0.0.0', port=5000)


# # from flask import Flask, request, jsonify
# # from ultralytics import YOLO  # Import the YOLO class from ultralytics
# # from PIL import Image
# # import io
# # import base64
# # import numpy as np
# # import cv2

# # # Initialize Flask app
# # app = Flask(__name__)

# # # Load YOLOv8 model using the correct method
# # model = YOLO('best.pt')  # Your YOLOv8 model

# # # Route for testing the server
# # @app.route('/')
# # def home():
# #     return "Chicken Disease! Detection Model is running!"

# # # Route to handle image upload and inference
# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     print("Predicting")
# #     try:
# #         # Get the image from the request
# #         if 'file' not in request.files:
# #             return jsonify({"error": "No file part"}), 400
        
# #         file = request.files['file']
        
# #         if file.filename == '':
# #             return jsonify({"error": "No selected file"}), 400
        
# #         # Read image
# #         img = Image.open(file.stream)

# #         # Perform inference
# #         results = model(img)  # Perform inference with YOLOv8 model

# #         # Extract predictions from results
# #         predictions = []
# #         for box in results[0].boxes:
# #             # For each box, we can get the label, confidence, and bounding box coordinates
# #             prediction = {
# #                 "label": int(box.cls.item()),  # Get the predicted class label as integer
# #                 "confidence": float(box.conf.item()),  # Get the confidence score as float
# #                 "x_min": float(box.xyxy[0].item()),  # Get the x_min of the bounding box
# #                 "y_min": float(box.xyxy[1].item()),  # Get the y_min of the bounding box
# #                 "x_max": float(box.xyxy[2].item()),  # Get the x_max of the bounding box
# #                 "y_max": float(box.xyxy[3].item()),  # Get the y_max of the bounding box
# #             }
# #             predictions.append(prediction)

# #         # Process image for return (optional: draw bounding boxes on the image)
# #         img_array = np.array(results[0].orig_img)  # Use the original image from the results
# #         ret, buffer = cv2.imencode('.jpg', img_array)
# #         img_bytes = buffer.tobytes()
# #         img_b64 = base64.b64encode(img_bytes).decode('utf-8')

# #         return jsonify({
# #             "prediction": predictions,  # Add detailed predictions
# #             "image": img_b64  # Return the processed image in base64
# #         })

# #     except Exception as e:
# #         # print(e)
# #         return jsonify({"error": str(e)}), 500

# # if __name__ == '__main__':
# #     app.run(debug=True, host='0.0.0.0', port=5000)



# from flask import Flask, request, jsonify
# from ultralytics import YOLO
# from PIL import Image
# import io
# import base64
# import numpy as np
# import cv2

# # Initialize Flask app
# app = Flask(__name__)

# # Load YOLOv8 model
# model = YOLO('best.pt')

# # Route for testing the server
# @app.route('/')
# def home():
#     return "Chicken Disease Detection Model is running!"

# # Route to handle image upload and inference
# @app.route('/predict', methods=['POST'])
# def predict():
#     print("Predicting")
#     try:
#         # Get the image from the request
#         if 'file' not in request.files:
#             return jsonify({"error": "No file part"}), 400

#         file = request.files['file']

#         if file.filename == '':
#             return jsonify({"error": "No selected file"}), 400

#         # Read image
#         img = Image.open(file.stream)

#         # Perform inference
#         results = model(img)

#         # Extract predictions from results
#         predictions = []
#         for box in results[0].boxes:
#             prediction = {
#                 "label": int(box.cls.item()),
#                 "confidence": float(box.conf.item()),
#                 "x_min": float(box.xyxy[0].item()),
#                 "y_min": float(box.xyxy[1].item()),
#                 "x_max": float(box.xyxy[2].item()),
#                 "y_max": float(box.xyxy[3].item()),
#             }
#             predictions.append(prediction)

#         # Process image for return (optional: draw bounding boxes on the image)
#         img_array = np.array(results[0].orig_img)
#         ret, buffer = cv2.imencode('.jpg', img_array)
#         img_bytes = buffer.tobytes()
#         img_b64 = base64.b64encode(img_bytes).decode('utf-8')

#         return jsonify({
#             "prediction": predictions,
#             "image": img_b64
#         })

#     except Exception as e:
#         print(e)
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)


# from ultralytics import YOLO
# from PIL import Image

# # Load the YOLOv8 model
# model = YOLO('best.pt') 

# # Path to your image
# image_path = 'cocci.jpg' 

# try:
#     # Load the image
#     img = Image.open(image_path)

#     # Perform inference
#     results = model(img)

#     # Access and print predictions
#     for *xyxy, conf, cls in results[0].boxes.xyxy: 
#         class_id = int(cls) 
#         class_name = model.names[class_id] 
#         confidence = float(conf) 

#         print(f"Class: {class_name}, Confidence: {confidence:.2f}") 
#         # Print bounding box coordinates (optional)
#         # print(f"Bounding Box: {xyxy}") 

# except Exception as e:
#     print(f"An error occurred: {e}")



# from ultralytics import YOLO  # Import the YOLO class from ultralytics
# from PIL import Image
# import numpy as np
# import cv2

# # Load YOLOv8 model using the correct method
# model = YOLO('best.pt')  # Path to your YOLOv8 model

# # Load image from relative path
# image_path = 'cocci.jpg'  # Specify your image path here
# img = Image.open(image_path)

# # Perform inference with YOLOv8 model
# results = model(img)

# # Extract predictions from results
# predictions = []
# print("=================")

# for box in results[0].boxes:
#     # For each box, get the label, confidence, and bounding box coordinates
#     prediction = {
#         "label": int(box.cls.item()),  # Class label (scalar)
#         "confidence": float(box.conf.item()),  # Confidence score (scalar)
#         "x_min": float(box.xyxy[0].item()),  # x_min of the bounding box (scalar)
#         "y_min": float(box.xyxy[1].item()),  # y_min of the bounding box (scalar)
#         "x_max": float(box.xyxy[2].item()),  # x_max of the bounding box (scalar)
#         "y_max": float(box.xyxy[3].item()),  # y_max of the bounding box (scalar)
#     }
#     predictions.append(prediction)

# # Print the prediction results
# for pred in predictions:
#     print(f"Class: {pred['label']}, Confidence: {pred['confidence']:.2f}")
#     print(f"Bounding Box: ({pred['x_min']}, {pred['y_min']}) to ({pred['x_max']}, {pred['y_max']})")

# # Optionally, you can draw the bounding boxes on the image
# img_array = np.array(results[0].orig_img)  # Get the original image with results
# for pred in predictions:
#     x_min, y_min, x_max, y_max = pred['x_min'], pred['y_min'], pred['x_max'], pred['y_max']
#     cv2.rectangle(img_array, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

# # Display the image with bounding boxes
# cv2.imshow("Detected Image", img_array)
# cv2.waitKey(0)  # Wait for a key press to close the image
# cv2.destroyAllWindows()



# import cv2
# from ultralytics import YOLO
# # from ultralytics import Annotator


# model = YOLO('best.pt')
# results = model("cocci.jpg")
# results[0].show()
# predictions = []

# for box in results[0].boxes:
#     # For each box, get the label, confidence, and bounding box coordinates
#     # prediction = {
#     #     "label": int(box.cls.item()),  # Class label (scalar)
#     #     "confidence": float(box.conf.item()),  # Confidence score (scalar)
#     #     "x_min": float(box.xyxy[0].item()),  # x_min of the bounding box (scalar)
#     #     "y_min": float(box.xyxy[1].item()),  # y_min of the bounding box (scalar)
#     #     "x_max": float(box.xyxy[2].item()),  # x_max of the bounding box (scalar)
#     #     "y_max": float(box.xyxy[3].item()),  # y_max of the bounding box (scalar)
#     # }
#     print(box)
#     print(type(b))
#     predictions.append(box)
# # x_line = 100

# # img = cv2.imread('cocci.jpg')
# # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # results = model.predict(img, conf=0.5, classes=0)
# # annotator = Annotator(img)

# # for r in results:
# #     for box in r.boxes:
# #         b = box.xyxy[0]
# #         if b[1] > x_line:
# #             c = box.cls
# #             annotator.box_label(b, f"{r.names[int(c)]} {float(box.conf):.2}")

# # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# # cv2.line(img, (0, x_line), (img.shape[1] - 1, x_line), (255, 0, 0), 2)

# # cv2.imshow("YOLO", img)
# # cv2.waitKey(0)

# # cv2.destroyAllWindows()


# from ultralytics import YOLO  # Import the YOLO class from ultralytics
# from PIL import Image
# import numpy as np
# import cv2

# # Load YOLOv8 model using the correct method
# model = YOLO('best.pt')  # Path to your YOLOv8 model

# # Load image from relative path
# image_path = 'cocci.jpg'  # Specify your image path here
# img = Image.open(image_path)

# # Perform inference with YOLOv8 model
# results = model(img)

# # Extract predictions from results
# predictions = []
# for box in results[0].boxes:
#     # For each box, get the label, confidence, and bounding box coordinates
#     prediction = {
#         "label": int(box.cls[0].item()),  # Class label (convert tensor to scalar)
#         "confidence": float(box.conf[0].item()),  # Confidence score (convert tensor to scalar)
#         "x_min": float(box.xyxy[0][0].item()),  # xmin (convert tensor to scalar)
#         "y_min": float(box.xyxy[0][1].item()),  # ymin (convert tensor to scalar)
#         "x_max": float(box.xyxy[0][2].item()),  # xmax (convert tensor to scalar)
#         "y_max": float(box.xyxy[0][3].item()),  # ymax (convert tensor to scalar)
#     }
#     predictions.append(prediction)

# # Print the prediction results
# for pred in predictions:
#     print(f"Class: {pred['label']}, Confidence: {pred['confidence']:.2f}")
#     print(f"Bounding Box: ({pred['x_min']}, {pred['y_min']}) to ({pred['x_max']}, {pred['y_max']})")

# # Optionally, you can draw the bounding boxes on the image
# img_array = np.array(results[0].orig_img)  # Get the original image with results
# for pred in predictions:
#     x_min, y_min, x_max, y_max = pred['x_min'], pred['y_min'], pred['x_max'], pred['y_max']
#     cv2.rectangle(img_array, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)



# height, width = img_array.shape[:2]
# max_dim = 800  # You can adjust this value based on your preference

# # Calculate the new size while maintaining the aspect ratio
# if width > height:
#     new_width = max_dim
#     new_height = int((max_dim / width) * height)
# else:
#     new_height = max_dim
#     new_width = int((max_dim / height) * width)

# # Resize the image
# resized_img = cv2.resize(img_array, (new_width, new_height))

# # Display the resized image with bounding boxes
# cv2.imshow("Resized Detected Image", resized_img)
# cv2.waitKey(0)  # Wait for a key press to close the image
# cv2.destroyAllWindows()

from flask import Flask, request, jsonify
from ultralytics import YOLO  # Import the YOLO class from ultralytics
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Load YOLOv8 model
model = YOLO('best.pt')  # Path to your YOLOv8 model

# Route for testing the server
@app.route('/')
def home():
    return "Chicken Disease Detection Model is running!"

# Route to handle image upload and inference
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image from the request
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Read image
        img = Image.open(file.stream)

        # Perform inference with YOLOv8 model
        results = model(img)

        # Extract predictions from results
        predictions = []
        for box in results[0].boxes:
            # For each box, get the label, confidence, and bounding box coordinates
            prediction = {
                "label": int(box.cls[0].item()),  # Class label (convert tensor to scalar)
                "confidence": float(box.conf[0].item())  # Confidence score (convert tensor to scalar)
            }
            predictions.append(prediction)

        # Return the results as JSON
        print(predictions)
        return jsonify({
            "predictions": predictions
        })

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

