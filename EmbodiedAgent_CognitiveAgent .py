"""
Simple Illustration of Embodied AI and Cognitive AI in Python. 
This example provides a simple, clear distinction between embodied AI (grounded in interaction) and cognitive AI (based on symbolic reasoning).

This example demonstrates two AI paradigms:

1. Embodied AI:
   - An agent interacts with a simple simulated environment (a grid world).
   - The agent perceives its surroundings, moves, and learns from interaction.
   - Emphasizes sensorimotor experience and environment interaction.

2. Cognitive AI:
   - An agent that performs symbolic reasoning and planning without physical interaction.
   - Uses abstract knowledge representation and logical inference.
   - Emphasizes internal mental processes like reasoning, problem-solving.

Key components:
- EmbodiedAgent: navigates a grid environment, senses surroundings, and moves.
- CognitiveAgent: performs reasoning over symbolic facts to answer queries.
- Simple demonstration of both agents' behaviors.

Dependencies:
- None (pure Python)
"""

# === Embodied AI Example ===
class EmbodiedEnvironment:
    """
    Simple grid environment where the agent can move and sense obstacles.
    """

    def __init__(self, size=5):
        self.size = size
        self.grid = [['.' for _ in range(size)] for _ in range(size)]
        # Place some obstacles
        self.grid[1][2] = '#'
        self.grid[3][3] = '#'
        # Agent start position
        self.agent_pos = (0, 0)

    def sense(self):
        """
        Returns a dictionary of adjacent cell states (up, down, left, right).
        '.' = free space, '#' = obstacle, None = out of bounds.
        """
        r, c = self.agent_pos
        directions = {
            'up': (r - 1, c),
            'down': (r + 1, c),
            'left': (r, c - 1),
            'right': (r, c + 1),
        }
        senses = {}
        for dir_, (rr, cc) in directions.items():
            if 0 <= rr < self.size and 0 <= cc < self.size:
                senses[dir_] = self.grid[rr][cc]
            else:
                senses[dir_] = None
        return senses

    def move(self, direction):
        """
        Moves agent if possible (no obstacle and within bounds).
        Returns True if moved, False otherwise.
        """
        r, c = self.agent_pos
        moves = {
            'up': (r - 1, c),
            'down': (r + 1, c),
            'left': (r, c - 1),
            'right': (r, c + 1),
        }
        if direction not in moves:
            return False

        rr, cc = moves[direction]
        if 0 <= rr < self.size and 0 <= cc < self.size and self.grid[rr][cc] == '.':
            self.agent_pos = (rr, cc)
            return True
        else:
            return False

    def display(self):
        """
        Prints grid with agent position.
        """
        for r in range(self.size):
            row = ''
            for c in range(self.size):
                if (r, c) == self.agent_pos:
                    row += 'A '
                else:
                    row += self.grid[r][c] + ' '
            print(row)
        print()


class EmbodiedAgent:
    """
    Agent that interacts with the embodied environment.
    """

    def __init__(self, env):
        self.env = env

    def explore(self):
        """
        Simple exploration: try moving in all directions if free.
        """
        senses = self.env.sense()
        print(f"Agent senses: {senses}")

        for direction, state in senses.items():
            if state == '.':
                moved = self.env.move(direction)
                if moved:
                    print(f"Agent moved {direction} to {self.env.agent_pos}")
                    return
        print("Agent could not move this turn.")


# === Cognitive AI Example ===
class CognitiveAgent:
    """
    Agent that uses symbolic reasoning to answer questions.
    """

    def __init__(self):
        # Knowledge base: facts and rules
        self.knowledge = {
            'facts': {
                'Socrates is a man',
                'All men are mortal',
            },
            'rules': [
                # Simple modus ponens rule: If "X is a man" and "All men are mortal" => "X is mortal"
                lambda kb: kb.get('Socrates is a man') and kb.get('All men are mortal'),
            ]
        }

    def infer(self):
        """
        Performs simple inference based on knowledge.
        """
        kb = {fact: True for fact in self.knowledge['facts']}
        for rule in self.knowledge['rules']:
            if rule(kb):
                kb['Socrates is mortal'] = True
        return kb

    def answer_question(self, question):
        """
        Answers questions based on inferred knowledge.
        """
        kb = self.infer()
        if question.lower() == 'is socrates mortal?':
            return "Yes, Socrates is mortal." if kb.get('Socrates is mortal') else "I don't know."
        else:
            return "I cannot answer that question."


# === Demo ===
if __name__ == "__main__":
    print("=== Embodied AI Demo ===")
    env = EmbodiedEnvironment()
    agent = EmbodiedAgent(env)
    env.display()

    for step in range(5):
        print(f"Step {step + 1}:")
        agent.explore()
        env.display()

    print("\n=== Cognitive AI Demo ===")
    cog_agent = CognitiveAgent()
    questions = [
        "Is Socrates mortal?",
        "Is Plato mortal?"
    ]

    for q in questions:
        print(f"Q: {q}")
        print(f"A: {cog_agent.answer_question(q)}\n")