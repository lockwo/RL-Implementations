import gym
import sys  
import numpy as np
import random
import math
from collections import deque
import tensorflow as tf
from wrappers import wrap_deepmind, make_atari
from replay_buffer import ReplayBuffer

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class DQN_NN(object):
    def __init__(self, action_size):
        self.action_space = action_size
        self.q_network = self.make_net()
        self.q_target = self.make_net()
        self.move_weights()
        self.buff = 1000000
        self.rb = ReplayBuffer(self.buff)
        self.batch = 32
        self.gamma = 0.99 
        self.epsilon = 1.0
        self.epsilon_decay_frames = 0.5 * 1e6
        self.epsilon_min = 0.02
        self.learning_rate = 3e-4
        self.opt = tf.keras.optimizers.Adam(lr=self.learning_rate)
        self.tau = 0.001
        self.iter = 0
        self.training_frequency = 4
        self.counter = 0
        self.update = 10000

    def move_weights(self):
        self.q_target.set_weights(self.q_network.get_weights())

    def make_net(self):
        init = tf.keras.initializers.VarianceScaling(scale=2)
        inputs = tf.keras.layers.Input(shape=(84, 84, 4))
        x = tf.keras.layers.Conv2D(32, 8, strides=(4,4), activation='relu', kernel_initializer=init)(inputs)
        x = tf.keras.layers.Conv2D(64, 4, strides=(3,3), activation='relu', kernel_initializer=init)(x)
        x = tf.keras.layers.Conv2D(64, 3, strides=(1,1), activation='relu', kernel_initializer=init)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024, activation='relu', kernel_initializer=init)(x)
        x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer=init)(x)
        x = tf.keras.layers.Dense(self.action_space, name='output')(x)
        model = tf.keras.models.Model(inputs=inputs, outputs=x)
        #model.summary()
        return model

    def remember(self, state, action, reward, next_state, done):
        self.rb.add(state, action, reward, next_state, done)
        self.counter += 1

    def get_action(self, obs):
        if random.random() < self.epsilon: 
            return np.random.choice(self.action_space)
        else:
            act = self.q_network(np.array([obs])/255.0).numpy()[0]
            act = np.argmax(act)
            return act 

    @tf.function
    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    @tf.function
    def grad_update(self, state_batch, action_batch, reward_batch, next_state_batch, dones_batch):
        with tf.GradientTape() as tape:
            next_q = self.q_target(next_state_batch)
            next_q = tf.expand_dims(tf.reduce_max(next_q, axis=1), -1)
            y = reward_batch + tf.math.multiply((1 - dones_batch), self.gamma * next_q)
            q = self.q_network(state_batch, training=True)
            pred = tf.gather_nd(q, action_batch)
            pred = tf.reshape(pred, [self.batch, 1])
            msbe = tf.math.reduce_mean(tf.math.square(y - pred))
        grads = tape.gradient(msbe, self.q_network.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.q_network.trainable_variables))

    def train(self):
        states, acts, res, nexts, dones = self.rb.sample(self.batch)
        states = states / 255.0
        nexts = nexts / 255.0
        state_batch = tf.convert_to_tensor(states)
        action_batch = tf.convert_to_tensor(acts, dtype=tf.int32)
        action_batch = [[i, action_batch[i]] for i in range(len(action_batch))]
        reward_batch = tf.convert_to_tensor(res, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(nexts)
        dones_batch = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        reward_batch = tf.reshape(reward_batch, [len(reward_batch), 1])
        dones_batch = tf.reshape(dones_batch, [len(dones_batch), 1])

        self.grad_update(state_batch, action_batch, reward_batch, next_state_batch, dones_batch)

        if self.iter % self.update == 0:
            self.update_target(self.q_target.trainable_variables, self.q_network.trainable_variables, 1)
        else:
            self.update_target(self.q_target.trainable_variables, self.q_network.trainable_variables, self.tau)
      
        if self.epsilon > self.epsilon_min:
            self.epsilon -= 1/self.epsilon_decay_frames

        self.iter += 1

# args = env name, save name

save = True
load = True
name = "pong_nn_0"
env_name = "PongNoFrameskip-v4"

if len(sys.argv) > 1:
    env_name = sys.argv[1]
    name = sys.argv[2]

mode_file = name + ".h5"
npy = name + ".npy"
frame_name = name + "_frames.npy"

max_frames = int(1e7)
windows = 50
learn_delay = 80000

env = make_atari(env_name)
env = wrap_deepmind(env)

print(env.action_space)
print(env.observation_space, env.observation_space.shape)
agent = DQN_NN(env.action_space.n)
fs = []
avg_reward = []
best_avg_reward = -math.inf
rs = deque(maxlen=windows)
frames = 0

if load:
    try:
        agent.q_network.load_weights(mode_file)
        agent.q_target.load_weights(mode_file)
        learn_delay = 80000
        print("Loaded networks")
    except:
        pass
    try:
        avg_reward = list(np.load(npy))
        best_avg_reward = np.max(avg_reward)
        print("loaded rewards")
    except:
        pass
    try:
        fs = list(np.load(frame_name))
        learn_delay = 0
        frames = fs[-1]
        agent.iter = frames
        agent.epsilon = 1 - min(frames/agent.epsilon_decay_frames, 0.98)
        print("Loaded frames")
    except:
        pass

g = len(fs)
while frames < max_frames:
    env.reset()
    s1, reward, done, info = env.step(env.action_space.sample())
    total_reward = 0
    while not done:
        #env.render()
        action = agent.get_action(s1)
        s2, reward, done, info = env.step(action)
        total_reward += reward
        agent.remember(s1, action, reward, s2, done)
        if agent.counter > learn_delay and frames % agent.training_frequency == 0:
            agent.train()
        s1 = s2
        frames += 1
    g += 1
    rs.append(total_reward)
    fs.append(frames)
    avg = np.mean(rs)
    avg_reward.append(avg)
    if avg > best_avg_reward:
        best_avg_reward = avg
    if (g - 1) % 20 == 0 and save:
        agent.q_network.save_weights(mode_file)
        temp = np.array(avg_reward)
        temp2 = np.array(fs)
        np.save(frame_name, temp2)
        np.save(npy, temp)

    print("Frames {}/{} Game {} || Best average reward {}, Current Average {}, Current Iteration Reward {}, # Updates {}, Epislon {}".\
        format(frames, max_frames, g, best_avg_reward, avg, total_reward, agent.iter, round(agent.epsilon, 4)))

agent.q_network.save_weights(mode_file)
temp = np.array(avg_reward)
temp2 = np.array(fs)
np.save(frame_name, temp2)
np.save(npy, temp)
