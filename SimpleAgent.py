import numpy as np
import gym

class SimpleAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))

    def choose_action(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state, done):
        predict = self.q_table[state, action]
        target = reward + self.gamma * np.max(self.q_table[next_state]) * (1 - done)
        self.q_table[state, action] += self.lr * (target - predict)


def run_simple_agent():
    env = gym.make("FrozenLake-v1", is_slippery=False)
    agent = SimpleAgent(env)

    for episode in range(500):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
    
    print("Q-table learned:")
    print(agent.q_table)

if __name__ == "__main__":
    run_simple_agent()