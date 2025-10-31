"""
Personalized Healthcare System with Continual Learning (Python Example)

This example demonstrates a simplified personalized healthcare predictive model
that uses continual learning to update its knowledge as new patient data arrives.

This template can be extended to more complex models, larger datasets, more features, and incorporate privacy-preserving methods (e.g., federated learning) for real personalized healthcare systems.

Key features:
- Uses a simple neural network (MLP) for health risk prediction.
- Supports continual learning by incrementally training on new patient data batches.
- Maintains personalized model updates without retraining from scratch.
- Includes key steps: data preprocessing, model definition, training, evaluation, and continual updates.

Note:
- This is a conceptual example; real healthcare systems require rigorous validation and privacy compliance.
- Dataset generation here is synthetic for demonstration purposes.

Dependencies:
- tensorflow
- numpy
- scikit-learn

Install via:
pip install tensorflow numpy scikit-learn
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ==== Synthetic Dataset Generation ====
def generate_patient_data(num_samples=1000):
    """
    Generate synthetic patient data for demonstration.
    Features: age, bmi, blood_pressure, cholesterol
    Label: binary health risk (0 = low, 1 = high)
    """
    np.random.seed(42)
    age = np.random.randint(20, 80, size=num_samples)
    bmi = np.random.normal(25, 5, size=num_samples)
    blood_pressure = np.random.normal(120, 15, size=num_samples)
    cholesterol = np.random.normal(200, 30, size=num_samples)

    # Simple risk rule for label: high risk if bmi>30 or bp>140 or cholesterol>240 or age>60
    risk = ((bmi > 30) | (blood_pressure > 140) | (cholesterol > 240) | (age > 60)).astype(int)

    X = np.vstack([age, bmi, blood_pressure, cholesterol]).T
    y = risk
    return X, y

# ==== Data Preprocessing ====
class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def fit_transform(self, X):
        self.scaler.fit(X)
        self.fitted = True
        return self.scaler.transform(X)

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Scaler not fitted yet.")
        return self.scaler.transform(X)

# ==== Model Definition ====
def build_model(input_shape):
    """
    Build a simple MLP model for binary classification.
    """
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # probability output
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# ==== Continual Learning System ====
class ContinualHealthcareModel:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.model = None
        self.is_trained = False

    def initial_train(self, X_train, y_train, epochs=10):
        """
        Initial training on first batch of patient data.
        """
        X_train_scaled = self.data_processor.fit_transform(X_train)
        self.model = build_model(input_shape=X_train_scaled.shape[1:])
        print("Starting initial training...")
        self.model.fit(X_train_scaled, y_train, epochs=epochs, verbose=1)
        self.is_trained = True
        print("Initial training complete.")

    def continual_update(self, X_new, y_new, epochs=5):
        """
        Update model continually with new patient data.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet. Train initial model first.")

        X_new_scaled = self.data_processor.transform(X_new)
        print("Continual learning update with new data...")
        self.model.fit(X_new_scaled, y_new, epochs=epochs, verbose=1)
        print("Update complete.")

    def predict(self, X):
        """
        Predict health risk probabilities for patient data.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")
        X_scaled = self.data_processor.transform(X)
        prob = self.model.predict(X_scaled).flatten()
        return prob

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test data.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")
        X_test_scaled = self.data_processor.transform(X_test)
        preds = (self.model.predict(X_test_scaled).flatten() > 0.5).astype(int)
        acc = accuracy_score(y_test, preds)
        print(f"Model accuracy: {acc:.4f}")
        return acc

# ==== Demonstration ====
if __name__ == "__main__":
    # Generate initial dataset (e.g., historical patient data)
    X_init, y_init = generate_patient_data(num_samples=1000)

    # Split into train and test
    train_size = int(0.8 * len(X_init))
    X_train, y_train = X_init[:train_size], y_init[:train_size]
    X_test, y_test = X_init[train_size:], y_init[train_size:]

    # Initialize continual learning system
    healthcare_system = ContinualHealthcareModel()

    # Initial training
    healthcare_system.initial_train(X_train, y_train, epochs=15)

    # Evaluate initial model
    healthcare_system.evaluate(X_test, y_test)

    # Simulate continual learning with new incoming patient data batches
    for batch_num in range(1, 4):
        print(f"\n--- New patient batch #{batch_num} ---")
        X_new, y_new = generate_patient_data(num_samples=200)
        healthcare_system.continual_update(X_new, y_new, epochs=5)
        healthcare_system.evaluate(X_test, y_test)

    # Predict example patient risk
    example_patient = np.array([[45, 28, 130, 210]])  # age, bmi, bp, cholesterol
    risk_prob = healthcare_system.predict(example_patient)[0]
    print(f"\nPredicted health risk probability for example patient: {risk_prob:.3f}")