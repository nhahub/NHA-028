# AI Doc - Intelligent Medical Diagnostics Platform

![AI Doc Architecture](Images/proAr.png)

**AI Doc** is a state-of-the-art medical diagnostics platform leveraging advanced Artificial Intelligence to assist healthcare professionals. It provides real-time analysis of medical imaging (X-rays, MRIs, CT scans) and an intelligent chat assistant for symptom assessment.

## 🚀 Features

### 1. Multi-Modal Image Analysis
*   **Primary Classification (EfficientNet-B0)**: Automatically categorizes scans into **Brain**, **Bone**, or **Chest** with high accuracy.
*   **Bone Fracture Detection (YOLOv8)**:
    *   Detects fractures in X-ray images.
    *   Provides precise bounding boxes and confidence scores.
    *   Real-time inference (< 50ms).
*   **Brain Tumor Segmentation (YOLOv8-Seg)**:
    *   Performs instance segmentation on MRI scans.
    *   Generates pixel-perfect masks to delineate tumor boundaries.
    *   Calculates tumor area/volume for surgical planning.
*   **Lung Disease Detection (Custom CNN)**:
    *   Classifies chest X-rays into **Normal**, **Benign**, or **Malignant**.
    *   Uses a custom-designed Deep Convolutional Neural Network (Keras/TensorFlow).

### 2. Intelligent Assistant
*   **AI Chatbot**: Integrated conversational agent to assist with symptom analysis and provide preliminary guidance based on patient inputs.

### 3. Modern Web Interface
*   **Responsive Design**: Built with semantic HTML5 and modern CSS3 for a seamless experience across devices.
*   **Real-Time Feedback**: Async fetch API ensures instant results without page reloads.
*   **Interactive Presentation**: Includes a built-in technical deep dive presentation (`presentation.html`).

## 🛠️ Tech Stack

### Frontend
*   **HTML5 / CSS3**: Custom responsive design with CSS variables and glassmorphism effects.
*   **JavaScript (Vanilla)**: Lightweight and fast client-side logic.
*   **Font Awesome**: For intuitive iconography.

### Backend
*   **Python 3.x**: Core logic.
*   **Flask**: Lightweight web server for API endpoints.
*   **PyTorch**: Inference engine for EfficientNet and YOLOv8 models.
*   **TensorFlow / Keras**: Inference engine for the custom Lung Disease CNN.
*   **OpenCV & Pillow**: Advanced image preprocessing (CLAHE, Denoising, Auto-contrast).

## 📂 Project Structure

```
AI-Doc/
├── css/                # Stylesheets
├── js/                 # Frontend JavaScript logic
├── models/             # Pre-trained AI models (YOLO, Keras, etc.)
├── Images/             # Assets and demo images
├── server.py           # Main Flask application
├── index.html          # Main dashboard
├── presentation.html   # Technical deep dive slides
├── chat.html           # AI Chat interface
└── README.md           # Project documentation
```

## ⚡ Getting Started

### Prerequisites
*   Python 3.8+
*   pip (Python package manager)
*   CUDA-capable GPU (Recommended for faster inference)

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/ai-doc.git
    cd ai-doc
    ```

2.  **Install Dependencies**
    ```bash
    pip install flask torch torchvision ultralytics tensorflow opencv-python pillow
    ```

### Running the Application

1.  **Start the Backend Server**
    ```bash
    python server.py
    ```
    *The server will start on `http://localhost:5000` (or the port specified in server.py).*

2.  **Launch the Frontend**
    *   Open `index.html` in your web browser.
    *   Or serve it using a simple HTTP server:
        ```bash
        python -m http.server 8000
        ```
        Then navigate to `http://localhost:8000`.

## 🛡️ Privacy & Security
AI Doc is designed with privacy in mind. All image processing is performed locally or on secure servers, ensuring patient data remains confidential.

## 🤝 Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
