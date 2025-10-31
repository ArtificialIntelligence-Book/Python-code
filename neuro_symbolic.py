"""
Simple Neuro-Symbolic System Combining Neural Network Image Classification
with Symbolic Rule-Based Reasoning.

This example demonstrates:
- A neural network classifier (using a pretrained model) to classify images into object categories.
- A symbolic reasoning system with rules that infer additional knowledge or decisions based on the classified objects.
- Integration of neural perception with symbolic reasoning to form a neuro-symbolic system.

Key components:
1. Neural Perception Module:
   - Uses a pretrained torchvision model (ResNet18) to classify images.
   - Maps predicted class indices to labels.

2. Symbolic Reasoning Module:
   - Defines simple rules to reason about objects.
   - Uses logical conditions on classified objects to infer conclusions.

3. Integration:
   - The system classifies an image to an object label.
   - The symbolic system reasons about the object label and returns insights.

Dependencies:
- torch, torchvision, PIL, numpy

Install via:
pip install torch torchvision pillow numpy
"""

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import numpy as np

# === Neural Perception Module ===
class NeuralClassifier:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        # Load pretrained ResNet18 model
        self.model = resnet18(pretrained=True).to(self.device)
        self.model.eval()
        # ImageNet class labels (simplified subset for demo)
        # For full labels, download from official source or use torchvision.datasets
        self.idx_to_label = {
            207: 'golden retriever',
            281: 'tabby cat',
            340: 'red fox',
            555: 'orange',
            651: 'umbrella',
            817: 'toilet tissue',
            924: 'banana',
            978: 'pizza',
        }
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def classify(self, image: Image.Image):
        """
        Classify input PIL image and return top predicted class label and confidence.
        """
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        # Get top predicted index and probability
        top_prob, top_idx = torch.topk(probabilities, k=1)
        top_idx = top_idx.item()
        top_prob = top_prob.item()

        # Map to known label or 'unknown'
        label = self.idx_to_label.get(top_idx, 'unknown')

        return label, top_prob

# === Symbolic Reasoning Module ===
class SymbolicReasoner:
    def __init__(self):
        # Define symbolic rules as functions or a rule base
        # For demo, simple rules based on object label

        # Example rule base:
        # If object is 'golden retriever' or 'tabby cat' => "It's a pet animal."
        # If object is 'banana' or 'orange' or 'pizza' => "It's food."
        # If object is 'umbrella' => "It's a rain protection item."
        # If object is 'toilet tissue' => "It's a hygiene product."
        # Else => "Unknown object category."

        self.rules = [
            (lambda obj: obj in ['golden retriever', 'tabby cat'], "It's a pet animal."),
            (lambda obj: obj in ['banana', 'orange', 'pizza'], "It's food."),
            (lambda obj: obj == 'umbrella', "It's a rain protection item."),
            (lambda obj: obj == 'toilet tissue', "It's a hygiene product."),
        ]

    def reason(self, object_label):
        """
        Apply symbolic rules to classified object label
        and return reasoning conclusions.
        """
        conclusions = []
        for condition, conclusion in self.rules:
            if condition(object_label):
                conclusions.append(conclusion)

        if not conclusions:
            conclusions.append("Unknown object category.")

        return conclusions

# === Integration: Neuro-Symbolic System ===
class NeuroSymbolicSystem:
    def __init__(self, device='cpu'):
        self.classifier = NeuralClassifier(device=device)
        self.reasoner = SymbolicReasoner()

    def analyze_image(self, image_path):
        """
        Load image, classify it, and apply symbolic reasoning.
        Returns classification and reasoning results.
        """
        image = Image.open(image_path).convert('RGB')
        label, confidence = self.classifier.classify(image)
        conclusions = self.reasoner.reason(label)

        return label, confidence, conclusions

# === Demo Usage ===
if __name__ == "__main__":
    import sys

    # Example image URLs or local paths could be used.
    # For demo, user should supply path to an image file compatible with ImageNet classes.

    if len(sys.argv) < 2:
        print("Usage: python neuro_symbolic.py <image_path>")
        print("Example: python neuro_symbolic.py dog.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    system = NeuroSymbolicSystem(device='cuda' if torch.cuda.is_available() else 'cpu')

    label, confidence, conclusions = system.analyze_image(image_path)

    print(f"Image classified as: {label} (confidence: {confidence:.2f})")
    print("Symbolic reasoning conclusions:")
    for c in conclusions:
        print(f"- {c}")