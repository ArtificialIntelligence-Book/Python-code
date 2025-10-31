"""
Simple Neuro-Symbolic Agent for Virtual Environment Navigation

This example demonstrates a neuro-symbolic agent that integrates:
- Neural perception: a simple CNN to classify the current perceived scene/image.
- Symbolic planning: a rule-based planner that decides the next move based on the perceived state and goal.

The virtual environment is a grid world with rooms represented by images.
The agent perceives the room via image classification (neural perception),
and plans moves symbolically to reach a goal room.

Key components:
1. VirtualEnvironment: Simulates a grid of "rooms" with images as perceptual inputs.
2. NeuralPerceptionModule: CNN that classifies the current room image.
3. SymbolicPlanner: Simple symbolic planner using predefined transition rules to reach the goal.
4. NeuroSymbolicAgent: Combines perception and planning to navigate.

Dependencies:
- torch, torchvision, PIL, numpy

Install via:
pip install torch torchvision pillow numpy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
import random

# === 1. Virtual Environment ===
class VirtualEnvironment:
    """
    Simple grid environment with rooms represented by images.
    Each room has a semantic label (e.g., 'kitchen', 'hall', 'bedroom').
    The agent perceives the current room's image and can move in four directions.
    """

    def __init__(self, grid_size=(3, 3), seed=42):
        self.grid_size = grid_size
        self.rooms = {}
        self.room_labels = ['kitchen', 'hall', 'bedroom', 'bathroom', 'office', 'garden']
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        # Initialize grid with random room labels
        for r in range(grid_size[0]):
            for c in range(grid_size[1]):
                label = random.choice(self.room_labels)
                self.rooms[(r, c)] = label

        # Agent start position
        self.agent_pos = (0, 0)

        # Create simple synthetic images per room label
        self.image_cache = {}
        self._create_room_images()

    def _create_room_images(self):
        """
        Create synthetic PIL images for each room type (solid color + label text).
        """
        from PIL import ImageDraw, ImageFont

        colors = {
            'kitchen': (255, 223, 186),
            'hall': (186, 225, 255),
            'bedroom': (255, 186, 201),
            'bathroom': (186, 255, 201),
            'office': (255, 255, 186),
            'garden': (186, 255, 255)
        }
        font = ImageFont.load_default()

        for label in self.room_labels:
            img = Image.new('RGB', (64, 64), color=colors[label])
            draw = ImageDraw.Draw(img)
            w, h = draw.textsize(label, font=font)
            draw.text(((64 - w) / 2, (64 - h) / 2), label, fill=(0, 0, 0), font=font)
            self.image_cache[label] = img

    def get_current_image(self):
        """
        Returns the PIL image of the current room the agent is in.
        """
        label = self.rooms[self.agent_pos]
        return self.image_cache[label]

    def get_current_label(self):
        """
        Returns the semantic label of the current room.
        """
        return self.rooms[self.agent_pos]

    def move(self, direction):
        """
        Move the agent in the environment if possible.
        direction: one of 'up', 'down', 'left', 'right'
        Returns True if move successful, False otherwise.
        """
        r, c = self.agent_pos
        if direction == 'up' and r > 0:
            self.agent_pos = (r - 1, c)
            return True
        elif direction == 'down' and r < self.grid_size[0] - 1:
            self.agent_pos = (r + 1, c)
            return True
        elif direction == 'left' and c > 0:
            self.agent_pos = (r, c - 1)
            return True
        elif direction == 'right' and c < self.grid_size[1] - 1:
            self.agent_pos = (r, c + 1)
            return True
        else:
            return False

# === 2. Neural Perception Module ===
class SimpleCNN(nn.Module):
    """
    Simple CNN to classify room images into room labels.
    """

    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (batch, 16, 32, 32)
        x = self.pool(F.relu(self.conv2(x)))  # (batch, 32, 16, 16)
        x = x.view(-1, 32 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class NeuralPerceptionModule:
    def __init__(self, labels, device='cpu'):
        self.labels = labels
        self.device = torch.device(device)
        self.model = SimpleCNN(num_classes=len(labels)).to(self.device)
        self.transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
        ])
        self.label_to_idx = {label: i for i, label in enumerate(labels)}
        self.idx_to_label = {i: label for i, label in enumerate(labels)}

        # For demo, randomly initialize model weights (no training)
        # In practice, train on labeled room images.
        self.model.eval()

    def perceive(self, image):
        """
        Classify the room image and return predicted label.
        """
        x = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            pred_idx = torch.argmax(logits, dim=1).item()
        return self.idx_to_label[pred_idx]

# === 3. Symbolic Planner ===
class SymbolicPlanner:
    """
    Simple planner with symbolic rules to navigate from current room to goal room label.
    Uses grid coordinates and room labels to plan moves.
    """

    def __init__(self, env: VirtualEnvironment):
        self.env = env

    def find_path(self, start_pos, goal_label):
        """
        Find a shortest path from start_pos to any room with goal_label using BFS.
        Returns list of positions to move through.
        """
        from collections import deque

        grid_size = self.env.grid_size
        visited = set()
        queue = deque()
        queue.append((start_pos, []))
        visited.add(start_pos)

        while queue:
            pos, path = queue.popleft()
            if self.env.rooms[pos] == goal_label:
                return path  # path is list of moves leading to goal room

            # Explore neighbors
            r, c = pos
            neighbors = []
            if r > 0:
                neighbors.append((r - 1, c))
            if r < grid_size[0] - 1:
                neighbors.append((r + 1, c))
            if c > 0:
                neighbors.append((r, c - 1))
            if c < grid_size[1] - 1:
                neighbors.append((r, c + 1))

            for npos in neighbors:
                if npos not in visited:
                    visited.add(npos)
                    queue.append((npos, path + [npos]))

        return None  # no path found

    def plan_moves(self, current_pos, goal_label):
        """
        Converts position path to move directions.
        """
        path = self.find_path(current_pos, goal_label)
        if not path:
            return []

        moves = []
        prev = current_pos
        for pos in path:
            r0, c0 = prev
            r1, c1 = pos
            if r1 == r0 - 1:
                moves.append('up')
            elif r1 == r0 + 1:
                moves.append('down')
            elif c1 == c0 - 1:
                moves.append('left')
            elif c1 == c0 + 1:
                moves.append('right')
            prev = pos

        return moves

# === 4. Neuro-Symbolic Agent ===
class NeuroSymbolicAgent:
    """
    Agent combining neural perception and symbolic planning to navigate environment.
    """

    def __init__(self, env: VirtualEnvironment, perception: NeuralPerceptionModule, planner: SymbolicPlanner):
        self.env = env
        self.perception = perception
        self.planner = planner

    def navigate_to(self, goal_label):
        """
        Navigate from current position to target room label using perception and planning.
        """
        steps_taken = 0
        max_steps = 50  # avoid infinite loops

        while steps_taken < max_steps:
            current_image = self.env.get_current_image()
            perceived_label = self.perception.perceive(current_image)

            print(f"Step {steps_taken}: Current perceived room: {perceived_label} at position {self.env.agent_pos}")

            if perceived_label == goal_label:
                print(f"Goal '{goal_label}' reached at position {self.env.agent_pos}!")
                return True

            # Plan moves to goal from current position
            moves = self.planner.plan_moves(self.env.agent_pos, goal_label)
            if not moves:
                print("No path found to goal.")
                return False

            # Take first planned move
            move = moves[0]
            success = self.env.move(move)

            if not success:
                print(f"Failed to move {move} from position {self.env.agent_pos}.")
                return False

            steps_taken += 1

        print("Max steps reached without finding goal.")
        return False

# === Demo ===
if __name__ == "__main__":
    # Initialize environment
    env = VirtualEnvironment(grid_size=(3, 3))

    # Labels known to perception module
    labels = env.room_labels

    # Initialize perception module (untrained CNN for demo)
    perception = NeuralPerceptionModule(labels=labels, device='cpu')

    # Initialize planner
    planner = SymbolicPlanner(env)

    # Create neuro-symbolic agent
    agent = NeuroSymbolicAgent(env, perception, planner)

    # Define goal room label to navigate to
    goal = 'garden'

    print(f"Starting navigation to goal room: '{goal}'\n")

    success = agent.navigate_to(goal)

    if success:
        print("\nNavigation successful!")
    else:
        print("\nNavigation failed.")