import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras import layers

class REINFORCEAgent:
    def __init__(self, n_actions, n_states, lr=0.01):
        self.n_actions = n_actions
        self.n_states = n_states
        self.model = self.build_model()
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.gamma = 0.99
    
    def build_model(self):
        model = tf.keras.Sequential([
            layers.Dense(24, activation='relu', input_shape=(self.n_states,)),
            layers.Dense(self.n_actions, activation='softmax')
        ])
        return model
    
    def choose_action(self, state):
        state = state[np.newaxis, :]
        prob = self.model(state)
        action = np.random.choice(self.n_actions, p=prob.numpy()[0])
        return action, prob[0, action]
    
    def train(self, states, actions, rewards):
        # Compute discounted rewards
        discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards) + 1e-8

        with tf.GradientTape() as tape:
            loss = 0
            for state, action, Gt in zip(states, actions, discounted_rewards):
                state = tf.expand_dims(state, 0)
                probs = self.model(state)
                log_prob = tf.math.log(probs[0, action])
                loss += -log_prob * Gt
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


def run_reinforce():
    env = gym.make('CartPole-v1')
    agent = REINFORCEAgent(env.action_space.n, env.observation_space.shape[0])

    for episode in range(300):
        state = env.reset()
        done = False
        states, actions, rewards = [], [], []
        while not done:
            action, prob = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
        
        agent.train(states, actions, rewards)
        if episode % 20 == 0:
            print(f"Episode {episode} finished")

if __name__ == "__main__":
    run_reinforce()