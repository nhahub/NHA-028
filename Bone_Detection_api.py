from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from ultralytics import YOLO
import numpy as np
import cv2
import io, base64

app = FastAPI(
    title="Bone Fracture Detection API",
    description="Detect bone fractures using YOLOv8",
    version="1.0"
)

# -----------------------------
# Load Bone Model ONLY
# -----------------------------
MODEL_PATH = r"c:\Users\Pc\Downloads\DEPI_Final_Project\bone fracture_part_yolov8\YOLOv8n_QuadroP1000\weights\best.pt"
model = YOLO(MODEL_PATH)


# -----------------------------
# Helper: Convert image → Base64
# -----------------------------
def to_base64(img):
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode()


# -----------------------------
# Endpoint 1 — JSON + Base64
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)

        result = model(img)[0]
        annotated = result.plot()

        detections = []
        for box in result.boxes:
            cls = model.names[int(box.cls[0])]
            conf = float(box.conf[0]) * 100
            detections.append({"class": cls, "confidence": round(conf, 2)})

        status = "Normal Bone" if len(detections) == 0 else "Fracture Detected"
        b64 = to_base64(annotated)

        return JSONResponse({
            "status": status,
            "num_fractures": len(detections),
            "detections": detections,
            "annotated_image": b64
        })

    except Exception:
        raise HTTPException(500, "Prediction failed")


# -----------------------------
# Endpoint 2 — Return ONLY Image (JPEG)
# -----------------------------
@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    try:
        content = await file.read()
        img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)

        result = model(img)[0]
        annotated = result.plot()

        ok, buffer = cv2.imencode(".jpg", annotated)
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")

    except Exception:
        raise HTTPException(500, "Image output failed")


# -----------------------------
# Endpoint 3 — HTML View
# -----------------------------
@app.post("/show")
async def show(file: UploadFile = File(...)):
    try:
        content = await file.read()
        img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)

        result = model(img)[0]
        annotated = result.plot()

        status = "Normal Bone" if len(result.boxes) == 0 else "Fracture Detected"
        b64 = to_base64(annotated)

        html = f"""
        <html>
        <body style="text-align:center; font-family:Arial">
            <h1>{status}</h1>
            <img src="data:image/jpeg;base64,{b64}" style="max-width:600px;border:2px solid black;">
        </body>
        </html>
        """

        return HTMLResponse(html)

    except Exception:
        raise HTTPException(500, "HTML output failed")


# -----------------------------
# Run API
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

