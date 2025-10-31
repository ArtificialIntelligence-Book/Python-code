"""
Examples of AI Agentic Design Patterns: ReAct and Reflection

This code demonstrates two agentic design patterns used in AI agents:

1. ReAct (Reasoning + Acting):
   - The agent alternates between reasoning about the environment or problem and taking actions.
   - Shows an explicit loop of thought and action steps.

2. Reflection:
   - The agent reviews its previous actions and outcomes to improve future decisions.
   - Demonstrates self-monitoring and adaptive behavior.

Each pattern is implemented as a simple class with a demo interaction loop.

Key Descriptions:
- Thought steps produce reasoning or decision-making.
- Action steps perform tasks or generate outputs.
- Reflection inspects past history to adapt or improve.

"""

import random

# === 1. ReAct Agent Pattern ===
class ReActAgent:
    """
    Agent that alternates between reasoning (thought) and acting.
    """

    def __init__(self):
        self.history = []

    def think(self, observation):
        """
        Simulate reasoning based on observation and past history.
        Returns a thought string.
        """
        thought = f"I observe '{observation}'. Let me think..."
        # Simple heuristic reasoning example
        if 'error' in observation.lower():
            thought += " There seems to be an error, I should try to fix it."
        else:
            thought += " Everything looks normal."
        self.history.append({'type': 'thought', 'content': thought})
        return thought

    def act(self):
        """
        Simulate an action decision.
        Returns an action string.
        """
        possible_actions = ['retry operation', 'log issue', 'proceed', 'request help']
        action = random.choice(possible_actions)
        self.history.append({'type': 'action', 'content': action})
        return action

    def run(self, observations):
        """
        Runs ReAct agent on a sequence of observations.
        """
        for obs in observations:
            thought = self.think(obs)
            print(f"Thought: {thought}")
            action = self.act()
            print(f"Action: {action}\n")


# === 2. Reflection Agent Pattern ===
class ReflectionAgent:
    """
    Agent that reviews past actions and outcomes to improve decisions.
    """

    def __init__(self):
        # Store history of (observation, action, outcome)
        self.history = []

    def decide_action(self, observation):
        """
        Decide an action based on observation and reflection on history.
        """
        # Reflect on past similar observations
        similar_past = [h for h in self.history if h['observation'] == observation]
        if similar_past:
            # Check if past action was successful
            successes = [h for h in similar_past if h['outcome'] == 'success']
            if successes:
                action = successes[-1]['action']  # reuse successful action
                reflection = f"Based on past success, repeating action '{action}'."
            else:
                action = "try_alternate_action"
                reflection = "Past action failed, trying alternate."
        else:
            action = "default_action"
            reflection = "No past data, using default action."

        self.history.append({'observation': observation, 'action': action, 'outcome': None})
        print(f"Reflection: {reflection}")
        return action

    def update_outcome(self, observation, action, outcome):
        """
        Update outcome of a specific observation-action pair.
        """
        for record in reversed(self.history):
            if record['observation'] == observation and record['action'] == action and record['outcome'] is None:
                record['outcome'] = outcome
                print(f"Outcome updated: {outcome} for action '{action}' on observation '{observation}'")
                break

    def run(self, observations_outcomes):
        """
        Run Reflection agent on list of (observation, outcome) tuples.
        """
        for observation, actual_outcome in observations_outcomes:
            action = self.decide_action(observation)
            print(f"Action taken: {action}")
            # Simulate getting outcome for action
            self.update_outcome(observation, action, actual_outcome)
            print("")


# === Demo Usage ===
if __name__ == "__main__":
    print("=== ReAct Agent Demo ===")
    react_agent = ReActAgent()
    observations = [
        "System running smoothly.",
        "Error: Connection lost.",
        "Warning: Low memory.",
        "All services operational."
    ]
    react_agent.run(observations)

    print("\n=== Reflection Agent Demo ===")
    reflection_agent = ReflectionAgent()
    # Each tuple is (observation, outcome_of_last_action)
    interactions = [
        ("low battery", "failure"),
        ("low battery", "failure"),
        ("low battery", "success"),
        ("network issue", "failure"),
        ("network issue", "success"),
        ("low battery", "success"),
    ]
    reflection_agent.run(interactions)