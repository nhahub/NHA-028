
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import io
from PIL import Image

app = FastAPI(title="MRI Segmentation API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the model
MODEL_PATH = r"D:\projects_archive\mri-scanners\runs\segment\mri_segmentation_V1.09\weights\best.pt"
try:
    model = YOLO(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    raise RuntimeError(f"Failed to load model: {e}")

def encode_image_to_base64(image_array):
    """Encodes a numpy image array to a base64 string."""
    _, buffer = cv2.imencode('.jpg', image_array)
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image file, runs segmentation, and returns the label and annotated image.
    """
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
             raise HTTPException(status_code=400, detail="Could not decode image.")

        # Run inference
        results = model.predict(img)
        
        # Process results
        result = results[0] # We only process one image at a time
        
        # Get annotated image (with bounding boxes and masks)
        # plot() returns a BGR numpy array
        annotated_img = result.plot()
        
        # Encode annotated image to base64
        annotated_img_base64 = encode_image_to_base64(annotated_img)
        
        # Extract labels
        labels = []
        if result.boxes:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                conf = float(box.conf[0])
                labels.append({"class": class_name, "confidence": conf})
        
        return JSONResponse(content={
            "labels": labels,
            "image_base64": annotated_img_base64
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
