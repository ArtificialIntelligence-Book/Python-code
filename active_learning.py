"""
Active Learning Overview and Simple Implementation Examples in Python

Active learning is a machine learning approach where the model
selectively queries the most informative unlabeled data points 
for annotation, reducing labeling costs and improving efficiency.

This script provides:
1. A brief summary of active learning applications across domains.
2. Simple code examples illustrating active learning for:
   - Image classification (uncertainty sampling)
   - Text classification (uncertainty sampling)
3. Key descriptions and comments explaining active learning concepts.

Note:
- Real-world active learning involves complex pipelines and data.
- Here, synthetic or toy datasets are used to demonstrate principles.
- You can extend these examples with domain-specific models and data.

Dependencies:
- numpy
- scikit-learn
- torch (for image example)
- transformers (for NLP example)

Install via:
pip install numpy scikit-learn torch torchvision transformers

"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# For NLP example
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# ======================
# 1. Active Learning Summary (Comments)
# ======================
"""
Active Learning Applications Summary:

- Computer Vision:
  * Image Classification: Select most uncertain images to label.
  * Semantic Segmentation: Query uncertain image regions/superpixels.
  * Object Detection: Prioritize ambiguous object regions (e.g., MIAOD).
  * Techniques: Uncertainty sampling, diversity sampling, adversarial uncertainty.

- Natural Language Processing:
  * Named Entity Recognition: Select sentences with uncertain entity predictions.
  * Sentiment Analysis: Query ambiguous sentiment texts.
  * Question Answering: Focus on difficult queries.
  * Techniques: Entropy-based uncertainty, margin sampling, query-by-committee.

- Audio Processing:
  * Speech Recognition: Select diverse or uncertain audio clips.
  * Speaker Identification: Prioritize hard-to-classify speaker samples.
  * Emotion Recognition: Focus on ambiguous emotional utterances.
"""

# ======================
# 2. Simple Active Learning for Image Classification
# ======================
def active_learning_image_classification():
    """
    Demonstrates uncertainty sampling active learning for image classification.
    Here, we simulate with tabular data for simplicity.
    """
    print("\n--- Active Learning: Image Classification Example ---")

    # Generate synthetic data (replace with real images + features)
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

    # Split into initial labeled pool and unlabeled pool
    X_train, X_unlabeled, y_train, y_unlabeled = train_test_split(X, y, train_size=0.1, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    labeled_indices = list(range(len(X_train)))
    unlabeled_indices = list(range(len(X_unlabeled)))

    for iteration in range(5):
        # Train model on current labeled data
        model.fit(X_train, y_train)

        # Predict probabilities on unlabeled data
        probs = model.predict_proba(X_unlabeled)

        # Compute uncertainty = 1 - max predicted class probability (least confident sampling)
        uncertainty = 1 - np.max(probs, axis=1)

        # Select top k most uncertain samples to label (simulate oracle)
        k = 20
        query_indices = np.argsort(uncertainty)[-k:]

        # Add queried samples to labeled dataset
        X_new = X_unlabeled[query_indices]
        y_new = y_unlabeled[query_indices]

        X_train = np.vstack([X_train, X_new])
        y_train = np.hstack([y_train, y_new])

        # Remove queried samples from unlabeled pool
        mask = np.ones(len(X_unlabeled), dtype=bool)
        mask[query_indices] = False
        X_unlabeled = X_unlabeled[mask]
        y_unlabeled = y_unlabeled[mask]

        # Evaluate on a held-out test set (simulate)
        X_test, y_test = make_classification(n_samples=200, n_features=20, n_classes=2, random_state=999)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Iteration {iteration+1}: Training size = {len(X_train)} | Test Accuracy = {acc:.3f}")

# ======================
# 3. Simple Active Learning for Text Classification (Sentiment Analysis)
# ======================
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer.encode_plus(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def active_learning_text_classification():
    """
    Demonstrates uncertainty sampling active learning for text classification
    using a pretrained BERT model fine-tuned on a small dataset.
    """

    print("\n--- Active Learning: NLP Text Classification Example ---")

    # Toy dataset: positive/negative movie reviews (simulate)
    # In real use, load real unlabeled & labeled datasets
    labeled_texts = [
        "I love this movie!", "This film was terrible.", "Amazing acting and story.",
        "I did not like this movie at all.", "Fantastic visuals and soundtrack."
    ]
    labeled_labels = [1, 0, 1, 0, 1]  # 1=positive, 0=negative

    unlabeled_texts = [
        "Worst movie I've seen.", "I enjoyed the film a lot.", "Not great but watchable.", 
        "Absolutely fantastic!", "Meh, it was okay."
    ]

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    # Initial training dataset
    train_dataset = TextDataset(labeled_texts, labeled_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    # Initial fine-tuning on labeled data
    for epoch in range(3):
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        print(f"Initial training epoch {epoch+1} done.")

    # Active learning loop: select most uncertain unlabeled samples to label
    model.eval()
    unlabeled_encodings = tokenizer(unlabeled_texts, padding=True, truncation=True, return_tensors='pt')
    input_ids = unlabeled_encodings['input_ids'].to(device)
    attention_mask = unlabeled_encodings['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        # Uncertainty = entropy of predicted class probabilities
        entropy = -(probs * probs.log()).sum(dim=1)

    # Select top k uncertain samples
    k = 2
    uncertain_indices = torch.topk(entropy, k).indices.cpu().numpy()

    # Simulate oracle labeling: assume we know true labels (for demo)
    oracle_labels = [0, 1, 0, 1, 0]  # hypothetical true labels for unlabeled_texts
    new_texts = [unlabeled_texts[i] for i in uncertain_indices]
    new_labels = [oracle_labels[i] for i in uncertain_indices]

    print(f"Selected samples for labeling (most uncertain): {new_texts}")

    # Add new labeled data to training set
    labeled_texts.extend(new_texts)
    labeled_labels.extend(new_labels)

    # Retrain model with expanded labeled dataset (simplified)
    train_dataset = TextDataset(labeled_texts, labeled_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    model.train()
    for epoch in range(2):
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        print(f"Retraining epoch {epoch+1} done.")

# ======================
# Main execution
# ======================
if __name__ == "__main__":
    active_learning_image_classification()
    active_learning_text_classification()