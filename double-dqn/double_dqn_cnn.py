import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import math
from collections import deque
import tensorflow as tf
import PIL as pil
import time


class double_dqn_agent(object):
    def __init__(self, action_size, batch_size):
        self.history = 4
        self.action_space = action_size
        self.q_network = self.make_net()
        self.target = self.make_net()
        self.move_weights()
        self.memory = deque(maxlen=1000000)
        self.batch = batch_size
        # Q Learning Parameters
        self.gamma = 0.95 # DISCOUNT FACTOR, CLOSE TO 1 = LONG TERM
        self.epsilon = 1.0 # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.tau = 0.01

    def make_net(self):
        inputs = tf.keras.layers.Input(shape=(84,84,self.history))
        x = tf.keras.layers.Conv2D(32, (8,8), strides=4, activation='relu', name='conv1')(inputs)
        x = tf.keras.layers.Conv2D(64, (4,4), strides=2, activation='relu', name='conv2')(x)
        x = tf.keras.layers.Conv2D(64, (3,3), strides=1, activation='relu', name='conv3')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(512, activation='relu', name='dense1')(x)
        x = tf.keras.layers.Dense(self.action_space, name='output')(x)
        model = tf.keras.models.Model(inputs=inputs, outputs=x)
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss="huber_loss", metrics=['mae'])
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
            obs = np.concatenate([obs[0], obs[1], obs[2], obs[3]], axis=2)
            obs = obs / 255.
            return np.argmax(self.q_network.predict(np.array([obs,]))[0])

    def train(self, iteration):
        minibatch = random.sample(self.memory, self.batch)
        states, targets = [], []
        for state, action, reward, next_state, done in minibatch:
            state = np.concatenate((state[0], state[1], state[2], state[3]), axis=2)
            state = state / 255.
            states.append(state)
            state = np.array([state,])
            next_state = np.concatenate((next_state[0], next_state[1], next_state[2], next_state[3]), axis=2)
            next_state = next_state / 255.
            next_state = np.array([next_state,])
            self.q_network.predict(state)
            target_f = self.q_network.predict(state)[0]
            if done:
                target_f[action] = reward
            else:
                q_pred = np.amax(self.target.predict(next_state)[0])
                target_f[action] = reward + self.gamma * q_pred
            targets.append(targets)
            target_f = np.array([target_f,])
            self.q_network.fit(state, target_f, epochs=1, verbose=0)
        #self.q_network.fit(np.array(states), np.array(targets), batch_size=self.batch, epochs=1, verbose=1) # Batch training
        
        if iteration % 10 == 0:
            self.move_weights()
        else:
            self.update_prime()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def preprocess(encode_frame):
    img = pil.Image.fromarray(encode_frame, 'RGB').convert('L').resize((84,110))
    img = img.crop((0, 18, 84, 102))
    img = np.asarray(img.getdata(), dtype=np.uint8).reshape(img.size[0], img.size[1], 1)
    return img
    

# Hyperparameters
ITERATIONS = 20000
batch_size = 32
windows = 100
learn_delay = 1000

env = gym.make("Pong-v0").env

print(env.action_space)
print(env.observation_space, env.observation_space.shape)
agent = double_dqn_agent(env.action_space.n, batch_size)
rewards = []
# Uncomment the line before to load model
#agent.q_network = tf.keras.models.load_model("pong_ddqn.h5")
avg_reward = deque(maxlen=ITERATIONS)
best_avg_reward = -math.inf
rs = deque(maxlen=windows)
frames = 0
for i in range(ITERATIONS):
    env.reset()
    s1, reward, done, info = env.step(env.action_space.sample())
    s2 = preprocess(s1)
    states = [s2, s2, s2, s2]
    total_reward = 0
    j = 0
    while not done:
        #env.render()
        action = agent.get_action(states)
        s2, reward, done, info = env.step(action)
        total_reward += reward
        prev = states
        states = states[1:]
        states.append(preprocess(s2))
        agent.remember(prev, action, reward, states, done)
        if len(agent.memory) > learn_delay and done:
            agent.train(i)
        if done:
            rewards.append(total_reward)
            rs.append(total_reward)
        else:
            s1 = s2
        j += 1
        frames += 1
        #print(frames)
    if i >= windows:
        avg = np.mean(rs)
        avg_reward.append(avg)
        if avg > best_avg_reward:
            best_avg_reward = avg
            agent.q_network.save("pong_ddqn.h5")
    else: 
        avg_reward.append(-21)
    
    print("\rEpisode {}/{} || Best average reward {}, Current Iteration Reward {}, Frames {}".format(i, ITERATIONS, best_avg_reward, total_reward, frames))#, end='', flush=True)

np.save("drewards", np.asarray(rewards))
np.save("daverages", np.asarray(avg_reward))
'''
plt.ylim(0,200)
plt.plot(rewards, color='olive', label='Reward')
plt.plot(avg_reward, color='red', label='Average')
plt.legend()
plt.ylabel('Reward')
plt.xlabel('Generation')
plt.show()
'''
