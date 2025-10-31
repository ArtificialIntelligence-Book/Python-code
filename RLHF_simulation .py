"""
Simulated RLHF Training of a Simple Language Model to Generate Positive and Encouraging Responses

This example simulates a simplified Reinforcement Learning from Human Feedback (RLHF) loop to train
a rudimentary language model (LLM) to produce positive and encouraging responses to user prompts
about overcoming challenges.

Key Components:
1. SimpleLanguageModel: Generates responses from a fixed set or templates with some randomness.
2. Environment: A set of user prompts related to challenges.
3. RewardSystem: Scores responses based on positivity and encouragement keywords.
4. RLHF Loop: Iteratively generates responses, scores them, and updates model parameters (simulated).
5. Parameter update is simulated by biasing the model's response selection towards higher-reward responses.

Note:
- This is a toy simulation without real neural network training.
- Real RLHF involves complex LLM fine-tuning with human feedback data and policy optimization.

"""

import random

# === 1. Simple Language Model ===
class SimpleLanguageModel:
    def __init__(self):
        # Model "parameters" simulated as weights for positive templates
        self.templates = [
            "Keep going! Every step forward is progress.",
            "You are stronger than you think; don't give up!",
            "Challenges are opportunities in disguise, stay positive!",
            "Remember, difficult roads often lead to beautiful destinations.",
            "Believe in yourself — you can overcome this!",
            "Stay focused and keep pushing; success is near.",
            "Every setback is a setup for a comeback.",
            "Your perseverance will pay off; keep the faith!",
            "Tough times build stronger people. You got this!",
            "Don't be discouraged, your effort will be rewarded."
        ]
        # Initialize weights equally
        self.weights = [1.0 for _ in self.templates]

    def generate_response(self, prompt):
        """
        Generate a response by sampling templates weighted by current weights.
        """
        total_weight = sum(self.weights)
        probabilities = [w / total_weight for w in self.weights]
        chosen_template = random.choices(self.templates, weights=probabilities, k=1)[0]
        return chosen_template

    def update_weights(self, chosen_index, reward, learning_rate=0.1):
        """
        Update the weight of the chosen template based on received reward.
        Simple gradient ascent on weights (simulated).
        """
        # Increase weight proportional to reward
        self.weights[chosen_index] += learning_rate * reward
        # Optional: normalize weights to prevent explosion
        norm = sum(self.weights)
        self.weights = [w / norm for w in self.weights]

# === 2. Environment: User Prompts ===
user_prompts = [
    "I'm struggling with my new job role.",
    "I failed my exam and feel hopeless.",
    "It's hard to stay motivated during this tough project.",
    "I feel overwhelmed by all the obstacles.",
    "I'm scared to try again after my last failure.",
    "Work-life balance has been difficult recently.",
    "I can't seem to get past this difficult phase.",
    "I'm losing confidence in myself.",
    "I feel stuck and don't know how to move forward.",
    "This challenge feels impossible to overcome."
]

# === 3. Reward System ===
class RewardSystem:
    def __init__(self):
        # Define positive keywords for scoring
        self.positive_keywords = [
            'keep going', 'stronger', 'opportunities', 'believe', 'success',
            'perseverance', 'comeback', 'faith', 'tough times', 'don\'t give up',
            'motivated', 'progress', 'focused', 'rewarded', 'beautiful destinations'
        ]

    def score_response(self, response):
        """
        Scores the response based on presence of positive keywords.
        Returns reward in [0,1].
        """
        response_lower = response.lower()
        hits = sum(1 for kw in self.positive_keywords if kw in response_lower)
        max_hits = len(self.positive_keywords)
        reward = hits / max_hits
        return reward

# === 4. RLHF Training Loop ===
def rlhf_training_loop(model, environment, reward_system, epochs=30):
    """
    Runs a simplified RLHF loop:
    - For each prompt, generate a response.
    - Score the response.
    - Update the model weights to favor positive responses.
    """
    print("Starting RLHF Training Loop...\n")

    for epoch in range(1, epochs + 1):
        total_reward = 0
        print(f"Epoch {epoch}")

        for prompt in environment:
            # Generate weighted response index and text
            total_w = sum(model.weights)
            # Probabilities for each template
            probs = [w / total_w for w in model.weights]
            chosen_index = random.choices(range(len(model.templates)), weights=probs, k=1)[0]
            response = model.templates[chosen_index]

            # Score response
            reward = reward_system.score_response(response)
            total_reward += reward

            # Update model weights based on reward
            model.update_weights(chosen_index, reward)

            print(f"Prompt: {prompt}")
            print(f"Response: {response}")
            print(f"Reward: {reward:.3f}\n")

        avg_reward = total_reward / len(environment)
        print(f"Average Reward this epoch: {avg_reward:.3f}\n{'-'*40}")

    print("Training complete.\n")

# === 5. Demo ===
if __name__ == "__main__":
    model = SimpleLanguageModel()
    reward_system = RewardSystem()

    # Run the RLHF training loop
    rlhf_training_loop(model, user_prompts, reward_system, epochs=10)

    # Demonstrate final model behavior
    print("=== Final Model Responses ===")
    for prompt in user_prompts:
        response = model.generate_response(prompt)
        print(f"Prompt: {prompt}")
        print(f"Response: {response}\n")