
from ultralytics import YOLO  # Import the YOLO class from ultralytics
from PIL import Image
import numpy as np
import cv2

# Load YOLOv8 model using the correct method
model = YOLO('best.pt')  # Path to your YOLOv8 model

# Load image from relative path
image_path = 'cocci.jpg'  # Specify your image path here
img = Image.open(image_path)

# Perform inference with YOLOv8 model
results = model(img)

# Extract predictions from results
predictions = []
for box in results[0].boxes:
    # For each box, get the label, confidence, and bounding box coordinates
    prediction = {
        "label": int(box.cls[0].item()),  # Class label (convert tensor to scalar)
        "confidence": float(box.conf[0].item()),  # Confidence score (convert tensor to scalar)
        "x_min": float(box.xyxy[0][0].item()),  # xmin (convert tensor to scalar)
        "y_min": float(box.xyxy[0][1].item()),  # ymin (convert tensor to scalar)
        "x_max": float(box.xyxy[0][2].item()),  # xmax (convert tensor to scalar)
        "y_max": float(box.xyxy[0][3].item()),  # ymax (convert tensor to scalar)
    }
    predictions.append(prediction)

# Print the prediction results
for pred in predictions:
    print(f"Class: {pred['label']}, Confidence: {pred['confidence']:.2f}")
    print(f"Bounding Box: ({pred['x_min']}, {pred['y_min']}) to ({pred['x_max']}, {pred['y_max']})")

# Optionally, you can draw the bounding boxes and class labels with confidence on the image
img_array = np.array(results[0].orig_img)  # Get the original image with results

# You can define a list of class names for the detected classes
class_names = ['cocci', 'healthy', 'ncd', "salmo"]  # Replace with your actual class names

for pred in predictions:
    x_min, y_min, x_max, y_max = pred['x_min'], pred['y_min'], pred['x_max'], pred['y_max']
    label = pred['label']
    confidence = pred['confidence']
    
    # Draw bounding box
    cv2.rectangle(img_array, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
    
    # Add text for the class and confidence
    label_text = f"{class_names[label]}: {confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_array, label_text, (int(x_min), int(y_min) - 10), font, 2, (0, 255, 0), 5)

# Resize the image to fit the screen
height, width = img_array.shape[:2]
max_dim = 800  # You can adjust this value based on your preference

# Calculate the new size while maintaining the aspect ratio
if width > height:
    new_width = max_dim
    new_height = int((max_dim / width) * height)
else:
    new_height = max_dim
    new_width = int((max_dim / height) * width)

# Resize the image
resized_img = cv2.resize(img_array, (new_width, new_height))

# Display the resized image with bounding boxes and labels
cv2.imshow("Resized Detected Image", resized_img)
cv2.waitKey(0)  # Wait for a key press to close the image
cv2.destroyAllWindows()
