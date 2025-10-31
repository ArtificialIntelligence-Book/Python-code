import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import gym

class ActorCritic:
    def __init__(self, state_dim, action_dim, actor_lr=0.001, critic_lr=0.005):
        # Actor network returns action probability distribution
        self.actor = tf.keras.Sequential([
            layers.Dense(24, activation='relu', input_shape=(state_dim,)),
            layers.Dense(action_dim, activation='softmax')
        ])

        # Critic network returns state-value estimate
        self.critic = tf.keras.Sequential([
            layers.Dense(24, activation='relu', input_shape=(state_dim,)),
            layers.Dense(1)
        ])

        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.gamma = 0.99

    def choose_action(self, state):
        state = state[np.newaxis, :]
        probs = self.actor(state)
        action = np.random.choice(probs.shape[1], p=probs.numpy()[0])
        return action

    def train(self, state, action, reward, next_state, done):
        state = state[np.newaxis, :]
        next_state = next_state[np.newaxis, :]

        with tf.GradientTape(persistent=True) as tape:
            state_value = self.critic(state)[0, 0]
            next_state_value = self.critic(next_state)[0, 0]
            target = reward + (1 - done) * self.gamma * next_state_value
            delta = target - state_value  # advantage estimate

            probs = self.actor(state)
            log_prob = tf.math.log(probs[0, action])
            actor_loss = -log_prob * delta
            critic_loss = delta ** 2

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        del tape

def run_actor_critic():
    env = gym.make("CartPole-v1")
    agent = ActorCritic(env.observation_space.shape[0], env.action_space.n)

    for episode in range(200):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.train(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        if episode % 20 == 0:
            print(f"Episode {episode}, total reward: {total_reward}")

if __name__ == "__main__":
    run_actor_critic()