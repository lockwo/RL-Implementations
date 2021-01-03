import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import math
from collections import deque
import tensorflow as tf
import PIL as pil
import time
import gym_moving_dot

class DQN_NN(object):
    def __init__(self, action_size):
        self.history = 4
        self.action_space = action_size
        self.q_network = self.make_net()
        self.q_target = self.make_net()
        self.move_weights()
        self.buff = 1000000
        self.states = np.zeros((self.buff, 84, 84, 4), dtype=np.uint8)
        self.actions = np.zeros((self.buff, 1))
        self.rewards = np.zeros((self.buff, 1))
        self.dones = np.zeros((self.buff, 1))
        self.next_states = np.zeros((self.buff, 84, 84, 4), dtype=np.uint8)
        self.counter = 0
        self.batch = 32
        self.gamma = 0.99 
        self.epsilon = 1.0
        self.epsilon_decay_frames = 1000000
        self.epsilon_min = 0.02
        self.learning_rate = 3e-4
        self.opt = tf.keras.optimizers.Adam(lr=self.learning_rate)
        self.tau = 0.001
        self.iter = 0
        self.training_frequency = 4
        self.update = 10000

    def move_weights(self):
        self.q_target.set_weights(self.q_network.get_weights())

    def make_net(self):
        init = tf.keras.initializers.VarianceScaling(scale=2)
        inputs = tf.keras.layers.Input(shape=(84,84,self.history))
        x = tf.keras.layers.Conv2D(32, 8, strides=(4,4), activation='relu', kernel_initializer=init)(inputs)
        x = tf.keras.layers.Conv2D(64, 4, strides=(3,3), activation='relu', kernel_initializer=init)(x)
        x = tf.keras.layers.Conv2D(64, 3, strides=(1,1), activation='relu', kernel_initializer=init)(x)
        #x = tf.keras.layers.Conv2D(128, 3, strides=(1,1), activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024, activation='relu', kernel_initializer=init)(x)
        x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer=init)(x)
        x = tf.keras.layers.Dense(self.action_space, name='output')(x)
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
        if self.epsilon > self.epsilon_min:
            self.epsilon -= 1/self.epsilon_decay_frames
        if random.random() < self.epsilon: 
            return np.random.choice(self.action_space)
        else:
            obs = obs / 255.
            act = self.q_network(np.array([obs,])).numpy()[0]
            act = np.argmax(act)
            return act 

    @tf.function
    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    #@tf.function
    def train(self):
        batch_indices = np.random.choice(min(self.counter, self.buff), self.batch)
        state_batch = tf.convert_to_tensor(self.states[batch_indices] / 255.)
        action_batch = tf.convert_to_tensor(self.actions[batch_indices])
        action_batch = [[i, action_batch[i][0]] for i in range(len(action_batch))]
        reward_batch = tf.convert_to_tensor(self.rewards[batch_indices])
        dones_batch = tf.convert_to_tensor(self.dones[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        action_batch = tf.cast(action_batch, dtype=tf.int32)
        dones_batch = tf.cast(dones_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_states[batch_indices] /255.)
        # Train critic
        with tf.GradientTape() as tape:
            next_q = self.q_target(next_state_batch)
            y = reward_batch + (1 - dones_batch) * self.gamma * next_q
            q = self.q_network(state_batch, training=True)
            pred = tf.gather_nd(q, action_batch)
            pred = tf.reshape(pred, [self.batch, 1])
            msbe = tf.math.reduce_mean(tf.math.square(y - pred))
        grads = tape.gradient(msbe, self.q_network.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.q_network.trainable_variables))

        if self.iter % self.update == 0:
            self.update_target(self.q_target.trainable_variables, self.q_network.trainable_variables, 1)
        else:
            self.update_target(self.q_target.trainable_variables, self.q_network.trainable_variables, self.tau)

        self.iter += 1

def preprocess(encode_frame):
    img = pil.Image.fromarray(encode_frame, 'RGB').convert('L').resize((84,110))
    img = img.crop((0, 18, 84, 102))
    #img.show()
    img = np.asarray(img.getdata(), dtype=np.uint8).reshape(img.size[0], img.size[1], 1)
    return img

def framestack(obs):
    return np.concatenate([obs[0], obs[1], obs[2], obs[3]], axis=2)

# Hyperparameters
ITERATIONS = 10000
batch_size = 32
windows = 10
learn_delay = 50000

#env = gym.make("Breakout-v4")
env = gym.make("Pong-v4")
#env = gym.make("MovingDotDiscrete-v0")

print(env.action_space)
print(env.observation_space, env.observation_space.shape)
agent = DQN_NN(env.action_space.n)
fs = []
agent.q_network = tf.keras.models.load_model("pong_nn.h5")
agent.q_target = tf.keras.models.load_model("pong_nn.h5")
avg_reward = deque(maxlen=ITERATIONS)
best_avg_reward = -math.inf
rs = deque(maxlen=windows)
frames = 0
i = 0

try:
    avg_reward = deque(list(np.load("avg_reward.npy")), maxlen=ITERATIONS)
except:
    pass
try:
    fs = list(np.load("frames.npy"))
    learn_delay = 0
    frames = fs[-1]
    agent.iter = frames
except:
    pass

i = len(fs)
agent.epsilon = 0.55
while i < ITERATIONS:
    env.reset()
    s1, reward, done, info = env.step(env.action_space.sample())
    s1 = preprocess(s1)
    states = [s1, s1, s1, s1]
    total_reward = 0
    j = 0
    while not done:
        #env.render()
        action = agent.get_action(framestack(states))
        s2, reward, done, info = env.step(action)
        total_reward += reward
        prev = states.copy()
        states = states[1:]
        states.append(preprocess(s2))
        agent.remember(framestack(prev), action, reward, framestack(states), done)
        if agent.counter > learn_delay and frames % agent.training_frequency == 0:
            #agent.iter += 1
            agent.train()
        s1 = s2
        j += 1
        frames += 1
    rs.append(total_reward)
    fs.append(frames)
    avg = np.mean(rs)
    avg_reward.append(avg)
    if i >= windows:
        if i % 20 == 0:
            agent.q_network.save("pong_nn.h5")
            temp = np.array(avg_reward)
            temp2 = np.array(fs)
            np.save("frames", temp2)
            np.save("avg_reward", temp)
        if avg > best_avg_reward:
            best_avg_reward = avg
    
    print("\rEpisode {}/{} || Best average reward {}, Current Average {}, Current Iteration Reward {}, Frames {}, # Updates {}, Epislon {:.4f}".format(i, ITERATIONS, best_avg_reward, avg, total_reward, frames, agent.iter, agent.epsilon), end='', flush=True)
    i += 1
avg_reward = np.array(avg_reward)
print(avg_reward.shape)
plt.plot(avg_reward, color='red', label='Average')
plt.legend()
plt.ylabel('Reward')
plt.xlabel('Iteration')
plt.show()

plt.plot(fs, avg_reward, color='red', label='Average')
plt.legend()
plt.ylabel('Reward')
plt.xlabel('Frames')
plt.show()
