import os
import io
import base64
from typing import List, Dict, Any, Optional

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ML deps
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from torchvision import transforms
from PIL import Image, ImageOps

import cv2
import numpy as np
import requests

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None  # Fallback if ultralytics isn't available

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
PRIMARY_DIR = os.path.join(ROOT, 'models', 'pimary class')  # note: folder name as provided
MRI_DIR = os.path.join(ROOT, 'models', 'mri_seg')
Bone_DIR = os.path.join(ROOT, 'models', 'Bone_Detection_Model')

PRIMARY_MODEL_PATH = os.path.join(PRIMARY_DIR, 'scan_classifier_model.pth')
MRI_MODEL_PATH = os.path.join(MRI_DIR, 'best.pth')
BONE_MODEL_PATH = os.path.join(Bone_DIR, 'bone_best.pth')
# External model API endpoints (can be overridden via env vars)
MRI_API_ENDPOINT = os.environ.get('MRI_API', 'http://localhost:8000/predict')
# Bone detection API included in the workspace — default to its local port
FRACTURE_API_ENDPOINT = os.environ.get('FRACTURE_API', 'http://localhost:8001/predict')
LUNG_API_ENDPOINT = os.environ.get('LUNG_API', 'http://localhost:8004/predict')

# ----------------------------------------------------------------------------
# Flask app
# ----------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# ----------------------------------------------------------------------------
# Primary classifier (EfficientNet-B0) setup
# ----------------------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_NAMES = {
    0: 'Brain',
    1: 'Bone',
    2: 'Chest',
}


def create_primary_model():
    model = efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(num_features, 24),
        nn.ReLU(),
        nn.Linear(24, 24),
        nn.ReLU(),
        nn.Linear(24, 3),
    )
    return model


primary_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

primary_model = None
primary_loaded_error: Optional[str] = None

try:
    if os.path.exists(PRIMARY_MODEL_PATH):
        primary_model = create_primary_model()
        state_dict = torch.load(PRIMARY_MODEL_PATH, map_location=DEVICE)
        primary_model.load_state_dict(state_dict)
        primary_model.to(DEVICE)
        primary_model.eval()
        print(f"Primary classifier loaded from {PRIMARY_MODEL_PATH}")
    else:
        primary_loaded_error = f"Primary model file not found at {PRIMARY_MODEL_PATH}"
        print(primary_loaded_error)
except Exception as e:
    primary_loaded_error = f"Failed to load primary model: {e}"
    print(primary_loaded_error)

# ----------------------------------------------------------------------------
# MRI segmentation (YOLO) setup
# ----------------------------------------------------------------------------
seg_model = None
seg_loaded_error: Optional[str] = None

# Bone (fracture) detection model
bone_model = None
bone_loaded_error: Optional[str] = None

if YOLO is None:
    seg_loaded_error = "ultralytics not installed; MRI segmentation unavailable"
else:
    try:
        if os.path.exists(MRI_MODEL_PATH):
            seg_model = YOLO(MRI_MODEL_PATH)
            print(f"MRI segmentation model loaded from {MRI_MODEL_PATH}")
        else:
            seg_loaded_error = f"Segmentation model file not found at {MRI_MODEL_PATH}"
            print(seg_loaded_error)
    except Exception as e:
        seg_loaded_error = f"Failed to load MRI segmentation model: {e}"
        print(seg_loaded_error)
    # Try loading local bone model if available
    try:
        if os.path.exists(BONE_MODEL_PATH):
            bone_model = YOLO(BONE_MODEL_PATH)
            print(f"Bone detection model loaded from {BONE_MODEL_PATH}")
        else:
            bone_loaded_error = f"Bone model file not found at {BONE_MODEL_PATH}"
            print(bone_loaded_error)
    except Exception as e:
        bone_loaded_error = f"Failed to load bone detection model: {e}"
        print(bone_loaded_error)

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def pil_to_cv2(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def enhance_image_cv2(bgr_img: np.ndarray) -> np.ndarray:
    # Basic enhancement/repair: denoise + CLAHE on luminance
    if bgr_img is None:
        return bgr_img
    try:
        # Denoise lightly
        denoised = cv2.fastNlMeansDenoisingColored(bgr_img, None, 3, 3, 7, 21)
        # Convert to LAB for CLAHE
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return enhanced
    except Exception:
        return bgr_img


def encode_image_to_base64(bgr_img: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', bgr_img)
    return base64.b64encode(buffer).decode('utf-8')


def convert_pil_to_black_and_white(pil_img: Image.Image, jpeg_quality: int = 90) -> bytes:
    """Convert a PIL image to black-and-white (grayscale), return JPEG bytes.

    The function converts the image to grayscale, then back to RGB so downstream
    code and models that expect 3 channels continue to work. It returns JPEG bytes
    which can be forwarded to model APIs. Use these bytes as the canonical image
    passed to local inference and to external services.
    """
    try:
        # Convert to grayscale, then back to RGB to keep 3 channels
        bw = ImageOps.grayscale(pil_img).convert('RGB')
        buf = io.BytesIO()
        bw.save(buf, format='JPEG', quality=jpeg_quality)
        return buf.getvalue()
    except Exception:
        # Fallback to original image bytes if conversion fails
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG', quality=jpeg_quality)
        return buf.getvalue()


def forward_image_to_api(url: str, img_bytes: bytes, timeout: int = 10) -> Dict[str, Any]:
    """Send image bytes to a model API (FastAPI style) and return JSON response.
    Expects the API to accept form file field named 'file' and return JSON with keys
    like 'labels' and 'image_base64' or classification keys depending on the model.
    """
    try:
        files = {'file': ('image.jpg', img_bytes, 'image/jpeg')}
        resp = requests.post(url, files=files, timeout=timeout)
        resp.raise_for_status()
        return {'ok': True, 'json': resp.json()}
    except Exception as e:
        return {'ok': False, 'error': str(e)}


def decode_base64_to_bgr(b64: str) -> Optional[np.ndarray]:
    try:
        data = base64.b64decode(b64)
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------

# Serve frontend
@app.get('/')
def index_root():
    return send_from_directory(ROOT, 'index.html')

@app.get('/index.html')
def index_html():
    return send_from_directory(ROOT, 'index.html')

@app.get('/css/<path:filename>')
def serve_css(filename: str):
    return send_from_directory(os.path.join(ROOT, 'css'), filename)

@app.get('/js/<path:filename>')
def serve_js(filename: str):
    return send_from_directory(os.path.join(ROOT, 'js'), filename)

@app.get('/images/<path:filename>')
@app.get('/Images/<path:filename>')
def serve_images(filename: str):
    # Support both lowercase and capitalized folder names
    img_dir_lower = os.path.join(ROOT, 'images')
    img_dir_upper = os.path.join(ROOT, 'Images')
    if os.path.isdir(img_dir_lower):
        return send_from_directory(img_dir_lower, filename)
    return send_from_directory(img_dir_upper, filename)

@app.get('/favicon.ico')
def favicon():
    # Try to serve an existing favicon.ico; otherwise, return 204 No Content
    try:
        return send_from_directory(ROOT, 'favicon.ico')
    except Exception:
        return ('', 204)

@app.get('/health')
def health():
    # detect if local bone detection API code exists in workspace
    bone_model_folder = os.path.join(ROOT, 'models', 'Bone_Detection_Model')
    fracture_present = os.path.isdir(bone_model_folder)
    fracture_error = None if fracture_present else 'Fracture model not integrated'
    return jsonify({
        'status': 'ok',
        'models': {
            'primary': {
                'loaded': primary_model is not None,
                'error': primary_loaded_error,
            },
            'brain_tumor': {
                'loaded': seg_model is not None,
                'error': seg_loaded_error,
            },
            'fracture': {
                'loaded': fracture_present,
                'error': fracture_error,
            },
            'lung_disease': {
                'loaded': False,
                'error': 'Lung disease model not integrated',
            },
        },
    })


@app.post('/api/analyze')
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded with name "file"'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    model_sel = request.form.get('model', 'primary')

    try:
        img_bytes = file.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    except Exception:
        return jsonify({'error': 'Invalid image'}), 400

    # Optional enhancement via PIL auto-contrast as a pre-step
    pil_img = ImageOps.autocontrast(pil_img)

    # Convert to black & white (grayscale -> RGB) before any model processing
    try:
        img_bytes = convert_pil_to_black_and_white(pil_img)
        # Recreate a PIL image from the B/W bytes for further local processing
        pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    except Exception:
        # If conversion fails, continue with original image bytes/pil_img
        pass

    # Prepare image once (use the B/W version)
    bgr_img = pil_to_cv2(pil_img)
    bgr_enhanced = enhance_image_cv2(bgr_img)

    classification: Dict[str, Any] = {}
    seg_result_labels: List[Dict[str, Any]] = []
    annotated_bgr = bgr_enhanced
    seg_performed = False
    notes = ''

    if model_sel == 'primary':
        # Primary classification
        if primary_model is None:
            classification['error'] = primary_loaded_error or 'Primary model unavailable'
            notes = 'Primary classification unavailable.'
        else:
            with torch.no_grad():
                input_tensor = primary_transform(pil_img).unsqueeze(0).to(DEVICE)
                outputs = primary_model(input_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                class_id = int(torch.argmax(probs).item())
                class_name = CLASS_NAMES.get(class_id, 'Unknown')
                confidence = float(probs[class_id].item())
                classification.update({
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                })
            # Route to specialized models depending on detected body part
            cid = classification.get('class_id')
            # Brain -> try local seg_model first, otherwise forward to MRI API
            if cid == 0:
                if seg_model is not None:
                    try:
                        results = seg_model.predict(bgr_enhanced)
                        res = results[0]
                        annotated_bgr = res.plot()
                        if getattr(res, 'boxes', None):
                            for box in res.boxes:
                                cls_id = int(box.cls[0])
                                conf = float(box.conf[0])
                                class_name = seg_model.names.get(cls_id, str(cls_id)) if hasattr(seg_model, 'names') else str(cls_id)
                                seg_result_labels.append({'class': class_name, 'confidence': conf})
                        seg_performed = True
                    except Exception as e:
                        seg_result_labels.append({'error': f'Segmentation failed: {e}'})
                else:
                    # Try forwarding to external MRI API
                    forward = forward_image_to_api(MRI_API_ENDPOINT, img_bytes)
                    if forward.get('ok'):
                        data = forward.get('json', {})
                        seg_result_labels = data.get('labels', []) or data.get('labels', [])
                        b64 = data.get('image_base64') or data.get('annotated_image_base64')
                        if b64:
                            img = decode_base64_to_bgr(b64)
                            if img is not None:
                                annotated_bgr = img
                                seg_performed = True
                    else:
                        seg_result_labels.append({'error': f"MRI API forward failed: {forward.get('error')}"})

            # Bone -> try local bone model first, otherwise forward to fracture API
            elif cid == 1:
                if bone_model is not None:
                    try:
                        results = bone_model.predict(bgr_enhanced)
                        res = results[0]
                        annotated_bgr = res.plot()
                        if getattr(res, 'boxes', None):
                            for box in res.boxes:
                                cls_id = int(box.cls[0])
                                conf = float(box.conf[0])
                                class_name = bone_model.names.get(cls_id, str(cls_id)) if hasattr(bone_model, 'names') else str(cls_id)
                                seg_result_labels.append({'class': class_name, 'confidence': conf})
                        seg_performed = True
                        notes = 'Bone fracture detection completed using local model.'
                    except Exception as e:
                        seg_result_labels.append({'error': f'Bone detection failed: {e}'})
                        notes = 'Bone detection attempted but encountered an error.'
                else:
                    forward = forward_image_to_api(FRACTURE_API_ENDPOINT, img_bytes)
                    if forward.get('ok'):
                        data = forward.get('json', {})
                        # Bone API returns 'detections' and 'annotated_image'
                        seg_result_labels = data.get('detections', []) or data.get('labels', []) or data.get('predictions', [])
                        b64 = data.get('annotated_image') or data.get('image_base64') or data.get('annotated_image_base64')
                        # Also include status/summary if present
                        status = data.get('status') or data.get('summary')
                        if status:
                            seg_result_labels.insert(0, {'status': status})
                        if b64:
                            img = decode_base64_to_bgr(b64)
                            if img is not None:
                                annotated_bgr = img
                                seg_performed = True
                        notes = 'Routed to fracture detection API.'
                    else:
                        notes = 'Fracture detection model not available; forwarded attempt failed.'

            # Chest -> forward to lung disease API
            elif cid == 2:
                forward = forward_image_to_api(LUNG_API_ENDPOINT, img_bytes)
                if forward.get('ok'):
                    data = forward.get('json', {})
                    seg_result_labels = data.get('labels', []) or data.get('predictions', [])
                    b64 = data.get('image_base64') or data.get('annotated_image_base64')
                    if b64:
                        img = decode_base64_to_bgr(b64)
                        if img is not None:
                            annotated_bgr = img
                            seg_performed = True
                    notes = 'Routed to lung disease detection API.'
                else:
                    notes = 'Lung disease detection model not available; forwarded attempt failed.'

            # Final notes if not set
            if not notes:
                notes = _compose_notes(classification, seg_result_labels)

    elif model_sel == 'brain_tumor':
        # Prefer local seg_model, otherwise forward to MRI API
        if seg_model is not None:
            try:
                results = seg_model.predict(bgr_enhanced)
                res = results[0]
                annotated_bgr = res.plot()
                if getattr(res, 'boxes', None):
                    for box in res.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = seg_model.names.get(cls_id, str(cls_id)) if hasattr(seg_model, 'names') else str(cls_id)
                        seg_result_labels.append({'class': class_name, 'confidence': conf})
                seg_performed = True
                notes = 'Brain tumor detection completed. Review highlighted regions.'
            except Exception as e:
                seg_result_labels.append({'error': f'Brain tumor detection failed: {e}'})
                notes = 'Brain tumor detection attempted but encountered an error.'
        else:
            forward = forward_image_to_api(MRI_API_ENDPOINT, img_bytes)
            if forward.get('ok'):
                data = forward.get('json', {})
                seg_result_labels = data.get('labels', [])
                b64 = data.get('image_base64') or data.get('annotated_image_base64')
                if b64:
                    img = decode_base64_to_bgr(b64)
                    if img is not None:
                        annotated_bgr = img
                        seg_performed = True
                notes = 'Brain tumor detection performed via external API.'
            else:
                seg_result_labels.append({'error': forward.get('error')})
                notes = 'Brain tumor detection API not available.'

    elif model_sel == 'fracture':
        # Prefer local bone model; otherwise forward to fracture API
        if bone_model is not None:
            try:
                results = bone_model.predict(bgr_enhanced)
                res = results[0]
                annotated_bgr = res.plot()
                detections = []
                if getattr(res, 'boxes', None):
                    for box in res.boxes:
                        cls = bone_model.names.get(int(box.cls[0]), str(int(box.cls[0]))) if hasattr(bone_model, 'names') else str(int(box.cls[0]))
                        conf = float(box.conf[0])
                        detections.append({'class': cls, 'confidence': conf})
                seg_result_labels = detections
                seg_performed = True
                notes = 'Fracture detection completed using local bone model.'
            except Exception as e:
                seg_result_labels.append({'error': f'Fracture detection failed: {e}'})
                notes = 'Fracture detection attempted but encountered an error.'
        else:
            forward = forward_image_to_api(FRACTURE_API_ENDPOINT, img_bytes)
            if forward.get('ok'):
                data = forward.get('json', {})
                seg_result_labels = data.get('detections', []) or data.get('labels', []) or data.get('predictions', [])
                b64 = data.get('annotated_image') or data.get('image_base64') or data.get('annotated_image_base64')
                status = data.get('status')
                if status:
                    seg_result_labels.insert(0, {'status': status, 'num_fractures': data.get('num_fractures')})
                if b64:
                    img = decode_base64_to_bgr(b64)
                    if img is not None:
                        annotated_bgr = img
                        seg_performed = True
                notes = 'Fracture detection performed via external API.'
            else:
                notes = 'Fracture detection model not integrated or API unreachable.'

    elif model_sel == 'lung_disease':
        # Forward to lung disease API if available
        forward = forward_image_to_api(LUNG_API_ENDPOINT, img_bytes)
        if forward.get('ok'):
            data = forward.get('json', {})
            seg_result_labels = data.get('labels', []) or data.get('predictions', [])
            b64 = data.get('image_base64') or data.get('annotated_image_base64')
            if b64:
                img = decode_base64_to_bgr(b64)
                if img is not None:
                    annotated_bgr = img
                    seg_performed = True
            notes = 'Lung disease detection performed via external API.'
        else:
            notes = 'Lung disease detection model not integrated or API unreachable.'

    else:
        notes = f'Unknown model selection: {model_sel}'

    image_base64 = encode_image_to_base64(annotated_bgr)

    response = {
        'selected_model': model_sel,
        'classification': classification,
        'segmentation': {
            'labels': seg_result_labels,
            'performed': seg_performed,
        },
        'annotated_image_base64': image_base64,
        'notes': notes,
    }

    return jsonify(response)


def _compose_notes(classification: Dict[str, Any], labels: List[Dict[str, Any]]) -> str:
    cls_name = classification.get('class_name')
    if not cls_name:
        return 'Primary classification unavailable.'
    if cls_name != 'Brain':
        return f"Detected body part: {cls_name}. Specialized MRI segmentation is not required."
    if labels and not any('error' in x for x in labels):
        return 'MRI segmentation completed. Review highlighted regions in the preview image.'
    if labels and any('error' in x for x in labels):
        return 'MRI segmentation attempted but encountered an error. Preview shows enhanced input.'
    return 'MRI segmentation model unavailable. Preview shows enhanced input.'


if __name__ == '__main__':
    # Default host/port, can be overridden by environment variables
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', '8000'))
    app.run(host=host, port=port, debug=True)
