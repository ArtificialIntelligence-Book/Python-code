class SOAR:
    def __init__(self):
        self.rules = {}  # Learned knowledge
        self.working_memory = {}
        
    def perceive(self, state, goal):
        self.working_memory = {'state': state, 'goal': goal}
    
    def propose_operators(self):
        # Generate possible actions based on current state
        state = self.working_memory['state']
        return ['move_forward', 'turn_left', 'turn_right', 'pick_up']
    
    def select_operator(self, operators):
        # Check if learned rule applies
        state_key = str(self.working_memory['state'])
        if state_key in self.rules:
            return self.rules[state_key]
        
        # Impasse: use deliberate reasoning (simplified as random choice)
        import random
        return random.choice(operators)
    
    def execute(self, operator):
        print(f"Executing: {operator}")
        # Simulate action execution
        return {'success': True, 'new_state': 'updated_state'}
    
    def learn(self, state, operator, result):
        # Chunk successful sequence into rule
        if result['success']:
            self.rules[str(state)] = operator
    
    def cycle(self, state, goal):
        self.perceive(state, goal)
        operators = self.propose_operators()
        selected = self.select_operator(operators)
        result = self.execute(selected)
        self.learn(state, selected, result)
        return result

# Example usage
agent = SOAR()
result = agent.cycle(state={'position': [0, 0]}, goal={'position': [5, 5]})