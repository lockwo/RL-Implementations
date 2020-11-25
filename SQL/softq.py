import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import math
from collections import deque
import tensorflow as tf

#tf.compat.v1.disable_eager_execution()

class Soft_Q_Agent(object):
    def __init__(self, action_size, state_size, batch_size):
        self.action_space = action_size
        self.state_space = state_size
        self.q_network = self.make_net(state_size)
        self.q_target = self.make_net(state_size)
        self.opt = tf.keras.optimizers.Adam(lr=1e-4)
        self.move_weights()
        self.buff = 10000
        self.states = np.zeros((self.buff, self.state_space[0]))
        self.actions = np.zeros((self.buff, 1))
        self.rewards = np.zeros((self.buff, 1))
        self.dones = np.zeros((self.buff, 1))
        self.next_states = np.zeros((self.buff, self.state_space[0]))
        self.batch = batch_size
        self.gamma = 0.99 # Reward Discount Factor
        self.alpha = 4 # Entropy parameter
        self.iter = 0
        self.tau = 0.005
        self.update = 10
        self.counter = 0

    def make_net(self, state):
        inputs = tf.keras.layers.Input(shape=(state))
        x = tf.keras.layers.Dense(64, activation='relu', name='dense1')(inputs)
        x = tf.keras.layers.Dense(128, activation='relu', name='dense2')(x)
        x = tf.keras.layers.Dense(32, activation='relu', name='dense3')(x)
        x = tf.keras.layers.Dense(self.action_space, name='output')(x)
        model = tf.keras.models.Model(inputs=inputs, outputs=x)
        #model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        i = self.counter % self.buff
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = int(done)
        self.counter += 1

    def move_weights(self):
        self.q_target.set_weights(self.q_network.get_weights())

    def entropy(self, dist):
        return -sum([i * np.log(i) for i in dist])

    def get_action(self, obs):
        probs = self.q_network.predict(np.array([obs,]))[0]
        value = self.E_Value(probs, 0)
        #print(probs, value)
        dist = tf.math.exp((probs - value) / self.alpha).numpy()
        dist /= sum(dist)
        #print(dist, sum(dist))
        #input()
        try:
            act = np.random.choice(self.action_space, p=dist)
        except:
            print(dist)
            act = 1
        #print(dist)
        return act, self.entropy(dist)

    @tf.function
    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    @tf.function
    def E_Value(self, Q_est, ax):
        return self.alpha * tf.math.log(tf.math.reduce_sum(tf.math.exp(Q_est/self.alpha), axis=ax, keepdims=True))

    #@tf.function
    def train(self):
        batch_indices = np.random.choice(min(self.counter, self.buff), self.batch)
        state_batch = tf.convert_to_tensor(self.states[batch_indices])
        action_batch = tf.convert_to_tensor(self.actions[batch_indices])
        #action_batch = tf.reshape(action_batch, [self.batch])
        action_batch = [[i, action_batch[i][0]] for i in range(len(action_batch))]
        reward_batch = tf.convert_to_tensor(self.rewards[batch_indices])
        dones_batch = tf.convert_to_tensor(self.dones[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        action_batch = tf.cast(action_batch, dtype=tf.int32)
        dones_batch = tf.cast(dones_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_states[batch_indices])
        # Train critic
        with tf.GradientTape(persistent=True) as tape:
            next_q = self.q_target(next_state_batch)
            next_v = self.E_Value(next_q, 1)
            y = reward_batch + (1 - dones_batch) * self.gamma * next_v
            critic = self.q_network(state_batch, training=True)
            #print(critic, action_batch)
            pred = tf.gather_nd(critic, action_batch)
            pred = tf.reshape(pred, [self.batch, 1])
            #print(pred)
            #input()
            #print(y, pred)
            msbe = tf.math.reduce_mean(tf.math.square(y - pred))
            #msbe = tf.keras.losses.MSE(y, pred)
        #print(msbe)
        #input()
        grads = tape.gradient(msbe, self.q_network.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.q_network.trainable_variables))
        '''
        if self.iter % self.update == 0:
            self.move_weights()
            #print("MOVED")
        '''
        if self.iter % self.update == 0:
            self.update_target(self.q_target.trainable_variables, self.q_network.trainable_variables, 1)
        else:
            self.update_target(self.q_target.trainable_variables, self.q_network.trainable_variables, self.tau)
        
        self.iter += 1


# Hyperparameters
ITERATIONS = 300
batch_size = 32
windows = 10
learn_delay = 200 

env = gym.make("CartPole-v1")
'''env.observation_space.shape'''
print(env.action_space)
print(env.observation_space, env.observation_space.shape)
agent = Soft_Q_Agent(env.action_space.n, env.observation_space.shape, batch_size)
rewards = []
entropy = []
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
        action, ent = agent.get_action(s1)
        entropy.append(ent)
        s2, reward, done, info = env.step(action)
        total_reward += reward
        agent.remember(s1, action, reward, s2, done)
        if agent.counter > learn_delay:
            agent.train()
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
    else: 
        avg_reward.append(0)
    
    print("\rEpisode {}/{} || Best average reward {}, Current Iteration Reward {}".format(i, ITERATIONS, best_avg_reward, total_reward) , end='', flush=True)

#np.save("rewards", np.asarray(rewards))
#np.save("averages", np.asarray(avg_reward))
#plt.ylim(0,500)
plt.plot(rewards, color='olive', label='Reward')
plt.plot(avg_reward, color='red', label='Average')
plt.legend()
plt.ylabel('Reward')
plt.xlabel('Iteration')
plt.show()

plt.plot(entropy)
plt.ylabel('Entropy')
plt.xlabel('Iteration')
plt.show()
