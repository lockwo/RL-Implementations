import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import math
from collections import deque
import tensorflow as tf

class double_dqn_agent(object):
    def __init__(self, action_size, state_size, batch_size):
        self.action_space = action_size
        self.q_network = self.make_net(state_size)
        self.target = self.make_net(state_size)
        self.move_weights()
        self.memory = deque(maxlen=20000)
        self.batch = batch_size
        # Q Learning Parameters
        self.gamma = 0.95 # DISCOUNT FACTOR, CLOSE TO 1 = LONG TERM
        self.epsilon = 1.0 # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.tau = 0.01

    def make_net(self, state):
        inputs = tf.keras.layers.Input(shape=(state))
        x = tf.keras.layers.Dense(32, activation='relu', name='dense1')(inputs)
        x = tf.keras.layers.Dense(64, activation='relu', name='dense2')(x)
        x = tf.keras.layers.Dense(32, activation='relu', name='dense3')(x)
        x = tf.keras.layers.Dense(self.action_space, name='output')(x)
        model = tf.keras.models.Model(inputs=inputs, outputs=x)
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Huber())
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def move_weights(self):
        self.target.set_weights(self.q_network.get_weights())

    def update_prime(self):
        qweights = np.asarray(self.q_network.get_weights())
        qprimeweights = np.asarray(self.target.get_weights())
        qprimeweights = self.tau * qweights + (1 - self.tau) * qprimeweights
        self.target.set_weights(qprimeweights)

    def get_action(self, obs):
        if random.random() < self.epsilon: 
            return np.random.choice(self.action_space)
        else:
            return np.argmax(self.q_network.predict(np.array([obs,]))[0])

    def train(self, iteration):
        minibatch = random.sample(self.memory, self.batch)
        for state, action, reward, next_state, done in minibatch:
            state = np.array([state,])
            next_state = np.array([next_state,])
            target_f = self.q_network.predict(state)[0]
            if done:
                target_f[action] = reward
            else:
                q_pred = np.amax(self.target.predict(next_state)[0])
                target_f[action] = reward + self.gamma*q_pred
            target_f = np.array([target_f,])
            self.q_network.fit(state, target_f, epochs=1, verbose=0)
        if iteration % 10 == 0:
            self.move_weights()
        else:
            self.update_prime()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Hyperparameters
ITERATIONS = 2000
batch_size = 32
windows = 100
learn_delay = 1000

# This is the standard stuff for Open AI Gym. Be sure to check out their docs if you need more help.
env = gym.make("CartPole-v1")
'''env.observation_space.shape'''
print(env.action_space)
print(env.observation_space, env.observation_space.shape)
agent = double_dqn_agent(env.action_space.n, env.observation_space.shape, batch_size)
rewards = []
# Uncomment the line before to load model
#agent.q_network = tf.keras.models.load_model("cartpole.h5")
avg_reward = deque(maxlen=ITERATIONS)
best_avg_reward = -math.inf
rs = deque(maxlen=windows)

for i in range(ITERATIONS):
    s1 = env.reset()
    total_reward = 0
    done = False
    while not done:
        #env.render()
        action = agent.get_action(s1)
        s2, reward, done, info = env.step(action)
        total_reward += reward
        agent.remember(s1, action, reward, s2, done)
        if len(agent.memory) > learn_delay and done:
            agent.train(i)
        if done:
            rewards.append(total_reward)
            rs.append(total_reward)
        else:
            s1 = s2
    if i >= windows:
        avg = np.mean(rs)
        avg_reward.append(avg)
        if avg > best_avg_reward:
            best_avg_reward = avg
            agent.q_network.save("cartpole_doubledqn.h5")
    else: 
        avg_reward.append(0)
    
    print("\rEpisode {}/{} || Best average reward {}, Current Iteration Reward {}".format(i, ITERATIONS, best_avg_reward, total_reward) , end='', flush=True)

np.save("rewards", np.asarray(rewards))
np.save("averages", np.asarray(avg_reward))
plt.ylim(0,220)
plt.plot(rewards, color='olive', label='Reward')
plt.plot(avg_reward, color='red', label='Average')
plt.legend()
plt.ylabel('Reward')
plt.xlabel('Generation')
plt.show()
