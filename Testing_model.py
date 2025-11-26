# ===============================================================
# Bone Fracture Detection – Test Script
# Description:
#   - Loads a trained YOLOv8n model
#   - Runs inference on a single test X-ray image
#   - Draws predicted bounding boxes (Green)
#   - Draws ground-truth bounding boxes (Red)
#   - Calculates fracture area percentage
#
# Environment:
#   - Python 3.12.4
#   - Ultralytics 8.3.220
#   - Torch 2.5.1 (CUDA 12.1)
#   - GPU: NVIDIA Quadro P1000 (4GB)
# ===============================================================

from ultralytics import YOLO
import cv2
import os

# ---------------------------------------------------------------
# Input paths (test image + annotation file)
# ---------------------------------------------------------------

imgTest = r"c:\Users\Pc\Downloads\DEPI_Final_Project\bone fracture.v1i.yolov8\valid\images\image-6-_jpg.rf.9613f1c894910f3d9050d487e2709575.jpg"

imgAnnot = r"c:\Users\Pc\Downloads\DEPI_Final_Project\bone fracture.v1i.yolov8\valid\labels\image-6-_jpg.rf.9613f1c894910f3d9050d487e2709575.txt"

# ---------------------------------------------------------------
# Load trained model (best.pt)
# ---------------------------------------------------------------

model_path = os.path.join(r"c:\Users\Pc\Downloads\DEPI_Final_Project\bone fracture.v1i.yolov8", "YOLOv8n_QuadroP1000", "weights","best.pt")

threshold = 0.5   #----> Minimum confidence score for detection

# ---------------------------------------------------------------
# Load image
# ---------------------------------------------------------------

img = cv2.imread(imgTest)
if img is None:
    print(f"Error loading image: {imgTest}")
    exit()

H, W, _ = img.shape
image_area = H * W
total_fracture_area = 0

# ---------------------------------------------------------------
# Load YOLO model
# ---------------------------------------------------------------

model = YOLO(model_path)

# Make a copy for drawing predictions
imgPredict = img.copy()

# Run inference
results = model(imgPredict)[0]

print(f"\n--- Results for: {os.path.basename(imgTest)} ---")


# ---------------------------------------------------------------
# Draw predicted bounding boxes (Green)
# ---------------------------------------------------------------

for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result
    
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    if score >= threshold:    # Calculate bounding box area
        box_width = x2 - x1
        box_height = y2 - y1
        box_area = box_width * box_height
        total_fracture_area += box_area
        
        cv2.rectangle(imgPredict, (x1, y1), (x2, y2), (0, 255, 0), 2) # Draw prediction in green
        class_name = results.names[int(class_id)].upper()
        
        
        text = f"{class_name}: {score:.2f}"
        cv2.putText(imgPredict, text, (x1, y1 - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                       cv2.LINE_AA)


# ---------------------------------------------------------------
# Calculate fracture percentage relative to image area
# ---------------------------------------------------------------     

fracture_percentage = 0
if image_area > 0 and total_fracture_area > 0:
    fracture_percentage = (total_fracture_area / image_area) * 100

print(f"Predicted Total Fracture Area: {total_fracture_area} pixels")
print(f"Image Area: {image_area} pixels")
print(f"Fracture Percentage (Prediction): {fracture_percentage:.2f}%")

# Write percentage text on image

percentage_text = f"Pred. Fracture: {fracture_percentage:.2f}%"
cv2.putText(imgPredict, percentage_text, (10, 30),
             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
               cv2.LINE_AA)
        

# ===============================================================
# Ground Truth (Red Boxes)
# ===============================================================


imgTruth = img.copy()
if os.path.exists(imgAnnot):
    with open(imgAnnot, 'r') as file:
        annotations = []
        for line in file.readlines():
            values = line.split()
            if len(values) >= 5:
                label = values[0]
                x,y,w,h = map(float, values[1:])
                annotations.append((label, x, y, w, h))

    for annotation in annotations:
        label, x, y, w, h = annotation
        # Class name from YOLO dictionary
        try:
            
            class_name_truth = results.names[int(label)].upper()
        except (KeyError, ValueError):
            class_name_truth = f"CLASS_{label}"

        # Convert YOLO (x,y,w,h) → pixel box coordinates
        x1 = int((x - w / 2) * W)
        y1 = int((y - h / 2) * H)
        x2 = int((x + w / 2) * W)
        y2 = int((y + h / 2) * H)
        
        cv2.rectangle(imgTruth, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw ground truth in red
        
        
        cv2.putText(imgTruth, class_name_truth, (x1, y1 - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                       cv2.LINE_AA)
else:
    cv2.putText(imgTruth, "NO ANNOTATION FILE", (10, 30),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                   cv2.LINE_AA)

# ===============================================================
# Display windows
# ===============================================================

cv2.imshow("1. Prediction (Green Boxes)", imgPredict)
cv2.imshow("2. Ground Truth (Red Boxes)", imgTruth)
cv2.imshow("3. Original Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()