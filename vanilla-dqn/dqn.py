import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import math
from collections import deque
import tensorflow as tf


class DQN_AGENT(object):
    def __init__(self, action_size, state_size, batch_size):
        self.action_space = action_size
        self.state_space = state_size[0]
        self.q_network = self.make_net(state_size)
        self.buff = 10000
        self.states = np.zeros((self.buff, self.state_space))
        self.actions = np.zeros((self.buff, 1))
        self.rewards = np.zeros((self.buff, 1))
        self.dones = np.zeros((self.buff, 1))
        self.next_states = np.zeros((self.buff, self.state_space))
        self.counter = 0
        self.batch = batch_size
        # Q Learning Parameters
        self.gamma = 0.95 # DISCOUNT FACTOR, CLOSE TO 1 = LONG TERM
        self.epsilon = 1.0 # Exploration rate
        self.epsilon_decay = 0.9
        self.epsilon_min = 0.01
        self.opt = tf.keras.optimizers.Adam(lr=3e-3)

    def make_net(self, state):
        inputs = tf.keras.layers.Input(shape=(state))
        x = tf.keras.layers.Dense(8, activation='relu')(inputs)
        x = tf.keras.layers.Dense(32, activation='relu')(inputs)
        #x = tf.keras.layers.Dense(64, activation='relu', name='dense2')(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.Dense(self.action_space, name='output')(x)
        #x = tf.keras.layers.Dense(self.action_space, name='output')(inputs)
        model = tf.keras.models.Model(inputs=inputs, outputs=x)
        model.summary()
        return model

    def remember(self, state, action, reward, next_state, done):
        i = self.counter % self.buff
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = int(done)
        self.counter += 1

    def get_action(self, obs):
        if random.random() < self.epsilon: 
            return np.random.choice(self.action_space)
        else:
            return np.argmax(self.q_network(np.array([obs]))[0])

    def train(self):
        batch_indices = np.random.choice(min(self.counter, self.buff), self.batch)
        state_batch = tf.convert_to_tensor(self.states[batch_indices])
        action_batch = tf.convert_to_tensor(self.actions[batch_indices].flatten(), dtype=tf.int32)
        #print(action_batch)
        action_batch = [[i, action_batch[i]] for i in range(len(action_batch))]
        reward_batch = tf.convert_to_tensor(self.rewards[batch_indices], dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_states[batch_indices])
        dones_batch = tf.convert_to_tensor(self.dones[batch_indices], dtype=tf.float32)
        reward_batch = tf.reshape(reward_batch, [len(reward_batch), 1])
        dones_batch = tf.reshape(dones_batch, [len(dones_batch), 1])

        with tf.GradientTape() as tape:
            next_q = self.q_network(next_state_batch)
            y = reward_batch + (1 - dones_batch) * self.gamma * next_q
            q = self.q_network(state_batch, training=True)
            #print(y, action_batch)
            target = tf.reshape(tf.gather_nd(y, action_batch), [self.batch, 1])
            pred = tf.gather_nd(q, action_batch)
            pred = tf.reshape(pred, [self.batch, 1])
            msbe = tf.math.reduce_mean(tf.math.square(target - pred))
        grads = tape.gradient(msbe, self.q_network.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.q_network.trainable_variables))


        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Hyperparameters
ITERATIONS = 200
batch_size = 32
windows = 50
learn_delay = 1000

env = gym.make("CartPole-v0")
'''env.observation_space.shape'''
print(env.action_space)
print(env.observation_space, env.observation_space.shape)
agent = DQN_AGENT(env.action_space.n, env.observation_space.shape, batch_size)
rewards = []
avg_reward = deque(maxlen=ITERATIONS)
best_avg_reward = -math.inf
rs = deque(maxlen=windows)

for i in range(ITERATIONS):
    s1 = env.reset()
    total_reward = 0
    done = False
    #s1 = [0 if i < 0 else 1 for i in s1]
    while not done:
        action = agent.get_action(s1)
        s2, reward, done, info = env.step(action)
        total_reward += reward
        #s2 = [0 if i < 0 else 1 for i in s2]
        agent.remember(s1, action, reward, s2, done)
        if agent.counter > learn_delay and done:
            agent.train()
        if done:
            rewards.append(total_reward)
            rs.append(total_reward)
        s1 = s2
    avg = np.mean(rs)
    avg_reward.append(avg)
    if avg > best_avg_reward:
        best_avg_reward = avg
    
    print("\rEpisode {}/{} || Best average reward {}, Current Iteration Reward {}".format(i, ITERATIONS, best_avg_reward, total_reward))

plt.ylim(0,200)
plt.plot(rewards, color='olive', label='Reward')
plt.plot(avg_reward, color='red', label='Average')
plt.legend()
plt.ylabel('Reward')
plt.xlabel('Generation')
plt.show()
