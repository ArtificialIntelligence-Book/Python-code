"""
Cognitive AI with Adaptive Memory System

This implementation highlights core cognitive AI features: 
remembering past interactions, learning contextually, and adapting responses dynamically for personalized and context-aware communication.

This example implements a simple Cognitive AI system capable of:
- Remembering past interactions (Interaction Memory).
- Learning and adapting contextually from interactions (Contextual Learning).
- Generating adaptive responses based on current input and memory (Adaptive Response Generation).

Overview:
- InteractionMemory: Stores past user inputs and AI responses.
- ContextualLearner: Extracts topics/concepts from interactions and updates context.
- AdaptiveResponseGenerator: Generates responses using current input and contextual memory.

This system simulates cognitive computing principles:
- Dynamic understanding adapting over continuous interactions.
- Personalized and context-aware responses.

Note:
- This is a simplified illustrative implementation.
- Real systems use advanced NLP, embeddings, and deep learning models.

"""

import re
from collections import deque, Counter
import random

# === 1. Interaction Memory ===
class InteractionMemory:
    def __init__(self, capacity=50):
        # Store recent interactions as (user_input, ai_response)
        self.capacity = capacity
        self.history = deque(maxlen=capacity)

    def add_interaction(self, user_input, ai_response):
        self.history.append((user_input, ai_response))

    def get_recent_interactions(self, n=5):
        # Return last n interactions
        return list(self.history)[-n:]

    def __str__(self):
        return "\n".join([f"User: {u}\nAI: {a}" for u, a in self.history])

# === 2. Contextual Learner ===
class ContextualLearner:
    def __init__(self):
        # Track word/concept frequencies to model context
        self.concept_counts = Counter()

    def extract_concepts(self, text):
        """
        Extract simple concepts (keywords) from text.
        For demo: lowercase words excluding common stopwords.
        """
        stopwords = set(['the', 'is', 'at', 'which', 'on', 'and', 'a', 'an', 'in', 'to', 'for', 'of', 'it'])
        words = re.findall(r'\b\w+\b', text.lower())
        concepts = [w for w in words if w not in stopwords and len(w) > 2]
        return concepts

    def update_context(self, user_input, ai_response):
        """
        Update concept counts based on new interaction.
        """
        user_concepts = self.extract_concepts(user_input)
        ai_concepts = self.extract_concepts(ai_response)
        for c in user_concepts + ai_concepts:
            self.concept_counts[c] += 1

    def get_top_concepts(self, n=5):
        """
        Return top n frequently occurring concepts.
        """
        return [c for c, _ in self.concept_counts.most_common(n)]

# === 3. Adaptive Response Generator ===
class AdaptiveResponseGenerator:
    def __init__(self, context_learner):
        self.context_learner = context_learner

    def generate_response(self, user_input, memory):
        """
        Generate a response based on user input and adaptive context.
        - Uses simple template with context concepts.
        - Personalizes by referencing recent topics from memory.
        """
        # Extract concepts from user input
        input_concepts = self.context_learner.extract_concepts(user_input)
        top_context = self.context_learner.get_top_concepts()

        # Personalization: check if user mentioned previous topics
        recent_interactions = memory.get_recent_interactions()
        recent_topics = set()
        for u, a in recent_interactions:
            recent_topics.update(self.context_learner.extract_concepts(u))
            recent_topics.update(self.context_learner.extract_concepts(a))

        # Response templates
        responses = []

        # If user input contains known context concepts, acknowledge and extend
        matched_concepts = set(input_concepts).intersection(top_context)
        if matched_concepts:
            concept_list = ", ".join(matched_concepts)
            responses.append(f"I see you're interested in {concept_list}. Could you tell me more?")
        
        # If user input mentions recent topics, make it personalized
        matched_recent = set(input_concepts).intersection(recent_topics)
        if matched_recent:
            recent_list = ", ".join(matched_recent)
            responses.append(f"That's related to what we talked about earlier about {recent_list}.")

        # Default fallback responses
        responses.append("Tell me more about that.")
        responses.append("Interesting! Can you elaborate?")
        responses.append("I'm here to listen. Please continue.")

        # Select response prioritizing context match
        for resp in responses:
            if resp.startswith("I see") or resp.startswith("That's related"):
                return resp
        return random.choice(responses)

# === 4. Cognitive AI System Combining All Components ===
class CognitiveAI:
    def __init__(self):
        self.memory = InteractionMemory()
        self.context_learner = ContextualLearner()
        self.response_generator = AdaptiveResponseGenerator(self.context_learner)

    def interact(self, user_input):
        """
        Process user input, update memory and context, generate adaptive response.
        """
        # Generate response
        response = self.response_generator.generate_response(user_input, self.memory)

        # Update memory and context
        self.memory.add_interaction(user_input, response)
        self.context_learner.update_context(user_input, response)

        return response

    def show_memory(self):
        """
        Display recent interaction history.
        """
        print("=== Interaction Memory ===")
        print(self.memory)

    def show_context(self):
        """
        Display current top concepts learned.
        """
        print("=== Learned Context Concepts ===")
        concepts = self.context_learner.get_top_concepts()
        print(", ".join(concepts) if concepts else "No context learned yet.")

# === 5. Demo Usage ===
if __name__ == "__main__":
    ai = CognitiveAI()

    print("Welcome to the Cognitive AI with Adaptive Memory System!")
    print("Type 'exit' to quit, 'memory' to view interactions, 'context' to view learned context.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        elif user_input.lower() == 'memory':
            ai.show_memory()
            continue
        elif user_input.lower() == 'context':
            ai.show_context()
            continue

        response = ai.interact(user_input)
        print(f"AI: {response}")