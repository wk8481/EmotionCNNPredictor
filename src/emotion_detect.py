import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
from PIL import Image

class EmotionDetector:
    """A class for efficient emotion detection from facial images."""
    
    def __init__(self, model_path):
        """
        Initialize the emotion detector with a pre-trained model.
        
        Args:
            model_path (str): Path to the trained model weights
        """
        # Define emotion mapping
        self.emotions = {
            0: 'Neutral', 1: 'Happy', 2: 'Surprise', 3: 'Sad', 4: 'Anger',
            5: 'Disgust', 6: 'Fear', 7: 'Contempt', 8: 'Unknown', 9: 'NF'
        }
        
        # Initialize the model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Load trained weights
        self._load_weights(model_path)
        self.model.eval()
        
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Grayscale(),  # Convert to grayscale
            transforms.Resize((48, 48)),  # Resize to match training size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])

    def _create_model(self):
        """Create the neural network model architecture."""
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Modify first layer for grayscale input
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), 
                               stride=(2, 2), padding=(3, 3), bias=False)
        # Modify final layer for emotion classification
        model.fc = nn.Linear(model.fc.in_features, len(self.emotions))
        return model

    def _load_weights(self, model_path):
        """Load and adjust pre-trained weights."""
        state_dict = torch.load(model_path, map_location=self.device)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("conv1") or k.startswith("fc"):
                new_state_dict[f"model.{k}"] = v
            else:
                new_state_dict[k] = v
        self.model.load_state_dict(new_state_dict, strict=False)

    def detect_faces(self, image):
        """
        Detect faces in an image using Haar Cascade Classifier.
        
        Args:
            image: numpy array of image (BGR format)
            
        Returns:
            list of (x, y, w, h) tuples for detected faces
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

    def process_face(self, face_img):
        """
        Process a face image and return the predicted emotion.
        
        Args:
            face_img: numpy array of face image
            
        Returns:
            str: Predicted emotion label
        """
        # Convert to PIL Image and apply transformations
        if face_img.shape[2] == 3:  # Check if image is BGR
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_img)
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(img_tensor)
            _, predicted = torch.max(output, 1)
            return self.emotions[predicted.item()]

    def analyze_image(self, image, return_faces=False):
        """
        Analyze an image and return emotions for all detected faces.
        
        Args:
            image: numpy array of image (BGR format)
            return_faces: bool, whether to return face coordinates
            
        Returns:
            list of (emotion, coordinates) tuples if return_faces=True
            list of emotions if return_faces=False
        """
        faces = self.detect_faces(image)
        results = []
        
        for (x, y, w, h) in faces:
            face_img = image[y:y+h, x:x+w]
            try:
                emotion = self.process_face(face_img)
                if return_faces:
                    results.append((emotion, (x, y, w, h)))
                else:
                    results.append(emotion)
            except Exception as e:
                print(f"Error processing face: {str(e)}")
                continue
                
        return results

# Example usage:
if __name__ == "__main__":
    # Initialize detector
    detector = EmotionDetector('../models/checkpoint_epoch_10.pth')
    
    # Example with single image
    image = cv2.imread('../data/FER2013Valid/fer0028653.png')
    emotions = detector.analyze_image(image)
    print(f"Detected emotions: {emotions}")
    
    # Example with visualization
    image = cv2.imread('../data/FER2013Valid/fer0028653.png')
    results = detector.analyze_image(image, return_faces=True)
    
    for emotion, (x, y, w, h) in results:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image, emotion, (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    cv2.imshow('Emotion Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        results = detector.analyze_image(frame, return_faces=True)
        
        # Draw results on frame
        for emotion, (x, y, w, h) in results:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()