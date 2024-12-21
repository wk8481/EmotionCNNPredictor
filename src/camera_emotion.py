import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import torchvision.models as models

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the checkpoint
checkpoint = torch.load(
    'results/cuda_test/ResNet18_epoch40_bs128_lr0.1_momentum0.9_wd0.0001_seed0_smoothTrue_mixupTrue_schedulerreduce_cuda_test/checkpoints/best_checkpoint.tar')

# Print out the keys in the checkpoint to understand its structure
print("Checkpoint keys:", checkpoint.keys())

# If there's a model definition in the checkpoint, use that
if 'model_arch' in checkpoint:
    model = checkpoint['model_arch']
else:
    # Custom ResNet18-like model
    class CustomResNet18(nn.Module):
        def __init__(self, num_classes=7):
            super(CustomResNet18, self).__init__()
            # Input is grayscale (1 channel)
            self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            # Basic block structure similar to ResNet18
            self.layer1 = self._make_layer(64, 64, 2)
            self.layer2 = self._make_layer(64, 128, 2, stride=2)
            self.layer3 = self._make_layer(128, 256, 2, stride=2)
            self.layer4 = self._make_layer(256, 512, 2, stride=2)

            # Final classification layer
            self.fc = nn.Linear(512, num_classes)

        def _make_layer(self, in_channels, out_channels, blocks, stride=1):
            layers = []
            # First block might need to downsample
            layers.append(self._block(in_channels, out_channels, stride))

            # Subsequent blocks
            for _ in range(1, blocks):
                layers.append(self._block(out_channels, out_channels))

            return nn.Sequential(*layers)

        def _block(self, in_channels, out_channels, stride=1):
            # Basic residual block
            shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )

            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            # Global average pooling
            x = torch.mean(x, dim=[2, 3])
            x = self.fc(x)
            return x


    # Create the model
    model = CustomResNet18()

# Load state dict with partial matching
state_dict = checkpoint['model_state_dict']
model_dict = model.state_dict()

# Filter out unexpected keys and match existing keys
filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}

# Load the filtered state dict
model.load_state_dict(filtered_state_dict, strict=False)

# Move to device
model = model.to(device)
model.eval()

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

cap = cv2.VideoCapture(0)
while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    if not ret:
        break

    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]

        # Preprocess image for PyTorch
        cropped_img = cv2.resize(roi_gray, (48, 48))
        input_tensor = torch.FloatTensor(cropped_img).unsqueeze(0).unsqueeze(0) / 255.0
        input_tensor = input_tensor.to(device)  # Move to device

        # Predict using PyTorch model
        # Inside your prediction loop
        with torch.no_grad():
            prediction = model(input_tensor)
            print("Raw predictions:", prediction)  # Print full prediction tensor
            print("Softmax probabilities:", torch.softmax(prediction, dim=1))  # See confidence distribution
            maxindex = torch.argmax(prediction, dim=1).item()

        cv2.putText(frame, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

    cv2.imshow('Video', cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()