# Emotion Detection Using Deep Learning

## Overview
This Python script detects emotions in facial images using deep learning. It leverages the ResNet-18 architecture pre-trained on ImageNet for accurate emotion classification.

### Key Steps:
1. Detect faces in images.
2. Preprocess detected face regions.
3. Predict emotions for each face.

---

## Code Breakdown

### **Class: `EmotionDetector`**
This class encapsulates the emotion detection pipeline.

#### **Initialization (`__init__`)**
```python
class EmotionDetector:
    def __init__(self, model_path):
        self.emotions = {0: 'Neutral', 1: 'Happy', 2: 'Surprise', 3: 'Sad', 4: 'Anger',
                         5: 'Disgust', 6: 'Fear', 7: 'Contempt', 8: 'Unknown', 9: 'NF'}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._create_model()
        self.model.to(self.device)
        self._load_weights(model_path)
        self.model.eval()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
```

**Purpose:** Sets up key components, including:
- **Emotion Mapping:** Matches emotion IDs to labels.
- **Device Selection:** Chooses between GPU and CPU.
- **Model and Weights:** Loads the ResNet-18 model and its parameters.
- **Face Detection:** Initializes a Haar Cascade Classifier.
- **Preprocessing:** Defines input transformations to align with the model.

---

#### **Model Creation (`_create_model`)**
```python
def _create_model(self):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(model.fc.in_features, len(self.emotions))
    return model
```

**Purpose:** Adapts ResNet-18 for emotion detection:
- Modifies the first layer to process grayscale images.
- Updates the final layer to classify emotions.

---

#### **Weight Loading (`_load_weights`)**
```python
def _load_weights(self, model_path):
    state_dict = torch.load(model_path, map_location=self.device)
    new_state_dict = {k if not (k.startswith("conv1") or k.startswith("fc")) else f"model.{k}": v for k, v in state_dict.items()}
    self.model.load_state_dict(new_state_dict, strict=False)
```

**Purpose:** Loads pre-trained weights while accounting for architectural adjustments.

---

#### **Face Detection (`detect_faces`)**
```python
def detect_faces(self, image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
```

**Purpose:** Detects faces in an image using Haar Cascade, returning bounding boxes.

---

#### **Face Processing (`process_face`)**
```python
def process_face(self, face_img):
    if face_img.shape[2] == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_img)
    img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
    with torch.no_grad():
        output = self.model(img_tensor)
        _, predicted = torch.max(output, 1)
        return self.emotions[predicted.item()]
```

**Purpose:** Prepares face regions for model inference and predicts emotions.

---

#### **Image Analysis (`analyze_image`)**
```python
def analyze_image(self, image, return_faces=False):
    faces = self.detect_faces(image)
    results = []
    for (x, y, w, h) in faces:
        face_img = image[y:y+h, x:x+w]
        try:
            emotion = self.process_face(face_img)
            results.append((emotion, (x, y, w, h)) if return_faces else emotion)
        except Exception as e:
            print(f"Error processing face: {str(e)}")
    return results
```

**Purpose:** Runs the entire pipeline to detect faces and classify their emotions. Optionally returns face coordinates.

---

### **Main Execution Block**
Demonstrates usage:
1. Detect emotions in a single image.
2. Display results with bounding boxes.
3. Perform real-time emotion detection via webcam.

---

## Notes
- **Face Detection:** Haar Cascades are efficient but may underperform in complex scenarios. Alternatives like MTCNN can improve accuracy.
- **Emotion Classification:** Relies on prior fine-tuning of the model on labeled datasets for better performance.

---

## Example Output
Detects faces in an image, classifies their emotions, and overlays bounding boxes and labels for visualization.

---

## Glossary

**Emotion Mapping:** Converts numeric emotion IDs (e.g., 0, 1, 2) into meaningful emotion labels such as 'Happy', 'Sad', or 'Neutral'. This ensures human-readable outputs from the model predictions.

**ResNet-18:** A convolutional neural network (CNN) known for its residual connections, pre-trained on ImageNet. It is used here as a backbone for extracting image features and classifying emotions efficiently.

**CUDA:** A parallel computing platform developed by NVIDIA. It accelerates computations by leveraging the GPU, which is ideal for tasks like deep learning.

**Haar Cascade:** A traditional machine learning-based method for object detection. It uses cascades of features to identify objects such as faces in an image.

**Transfer Learning:** A technique in machine learning where a model trained on one task (e.g., ImageNet classification) is adapted to a similar but distinct task (e.g., emotion detection). This saves time and resources.

**Grayscale Conversion:** Reduces a color image to one channel, simplifying it to shades of gray. This reduces input complexity while retaining essential features for face and emotion detection.

**Bounding Box:** A rectangular frame around detected objects (e.g., faces). The box provides coordinates used to extract and analyze the object.

**Normalization:** Adjusts pixel intensity values to a consistent range, typically for more stable and effective training or inference of deep learning models.

**PIL (Python Imaging Library):** A Python library for opening, manipulating, and saving images. In this script, it prepares images for transformation into tensor format.

**State Dictionary:** A data structure in PyTorch that holds the model's parameters (weights and biases). It is used to save and reload models during training and deployment.

**Inference:** The process of using a trained model to make predictions on unseen data. This script uses inference to classify emotions from detected faces in images or videos.

