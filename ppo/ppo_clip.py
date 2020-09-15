import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import math
from collections import deque
import tensorflow as tf

#tf.compat.v1.disable_eager_execution()
#tf.keras.backend.set_floatx('float64')

class PPO_agent(object):
    def __init__(self, action_size, state_size):
        self.action_space = action_size
        self.state_space = state_size[0]
        self.e = 0.2 # Policy distance
        self.actor, self.critic = self.make_net(state_size)
        self.gamma = 0.98 # DISCOUNT FACTOR, CLOSE TO 1 = LONG TERM
        self.K = 8 # Number of epochs
        self.T = 2048 # Horizon
        self.M = 64 # Batch size
        self.memory = deque(maxlen=self.T)
        self.opt = tf.keras.optimizers.Adam()
        self.temp = []
        self.rewards = []

    def ppo_loss(self, ytrue, ypred):
        #action = ytrue[:2]
        #log_prob = ytrue[2:]
        action, log_prob = tf.split(ytrue, num_or_size_splits=2, axis=1)
        ratio = tf.math.exp(tf.math.log(ypred) - log_prob)
        ratio = tf.clip_by_value(ratio, 1e-10, 10-1e-10)
        clipped = tf.clip_by_value(ratio, 1-self.e, 1+self.e)
        loss = -tf.reduce_mean(tf.math.minimum(tf.multiply(ratio, action), tf.multiply(clipped,action)))
        #print(loss)
        return loss

    def make_net(self, state):
        inputs = tf.keras.layers.Input(shape=(state))
        value = tf.keras.layers.Dense(64, activation='relu', name='dense1')(inputs)
        value = tf.keras.layers.Dense(128, activation='relu', name='dense2')(value)
        value = tf.keras.layers.Dense(64, activation='relu', name='value1')(value)
        value = tf.keras.layers.Dense(1, name='v_out')(value)
        policy = tf.keras.layers.Dense(64, activation='relu', name='dense1')(inputs)
        policy = tf.keras.layers.Dense(128, activation='relu', name='dense2')(policy) 
        policy = tf.keras.layers.Dense(64, activation='relu', name='policy1')(policy)
        policy = tf.keras.layers.Dense(self.action_space, activation='softmax', name='policy_out')(policy)
        v_model = tf.keras.models.Model(inputs=inputs, outputs=value)
        p_model = tf.keras.models.Model(inputs=inputs, outputs=policy)
        v_model.summary()
        p_model.summary()
        v_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
        p_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=self.ppo_loss)
        return p_model, v_model

    def remember(self, state, action, reward, value, log_probs):
        self.temp.append([state, action, reward, value, tf.math.log(log_probs)])
        self.rewards.append(reward)

    def get_action(self, obs):
        temp = self.actor.predict(np.array([obs,]))
        probs = temp[0]
        action = np.random.choice(self.action_space, p=probs)
        temp = self.critic.predict(np.array([obs,]))
        value = temp[0]
        return action, value, probs

    def discount_reward(self, rewards):
        d_rewards = np.zeros_like(rewards)
        Gt = 0
        # Discount rewards
        for i in reversed(range(len(rewards))):
            done = 1
            if i == len(rewards) - 1:
                done = 0
                Gt = 0
            else:
                Gt = rewards[i] + self.gamma * self.temp[i+1][3]
                #Gt = done * Gt * self.gamma + rewards[i]
            d_rewards[i] = Gt

        return d_rewards

    def train(self):
        state = np.zeros(shape=(self.M, self.state_space))
        action = np.zeros(shape=(self.M, self.action_space))
        rws = np.zeros(shape=(self.M))

        rewards = self.discount_reward(self.rewards)
        for counter, v in enumerate(rewards):
            self.memory.append((self.temp[counter][0], self.temp[counter][1], v, self.temp[counter][3], self.temp[counter][4]))
        if len(self.memory) < self.M:
            self.temp.clear()
            self.rewards.clear()
            return
        minibatch = random.sample(self.memory, self.M)
        i = 0
        f = np.zeros(shape=(self.M, 4))
        #f = np.zeros(shape=(self.M, 2, 2))
        for s, a, r, v, log_prob in minibatch:
            state[i] = s
            #print(s, r, v, a)
            rws[i] = r
            action[i][a] = r - v[0]
            f[i][:2] = action[i]
            f[i][2:] = log_prob
            #f[i][0] = action[i]
            #f[i][2]
            i += 1
        #print(state, f)
        #print(state.shape, f.shape)
        #f = tf.convert_to_tensor(f)
        self.actor.fit(state, f, epochs=self.K, verbose=0)
        self.critic.fit(state, rws, epochs=self.K, verbose=0)
        self.temp.clear()
        self.rewards.clear()



# Hyperparameters
ITERATIONS = 750
windows = 50

#env = gym.make("LunarLander-v2")
env = gym.make("CartPole-v1")
'''env.observation_space.shape'''
print(env.action_space)
print(env.observation_space, env.observation_space.shape)
agent = PPO_agent(env.action_space.n, env.observation_space.shape)
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
        action, value, lp = agent.get_action(s1)
        s2, reward, done, info = env.step(action)
        total_reward += reward
        agent.remember(s1, action, reward, value, lp)
        if done:
            agent.train()
            rewards.append(total_reward)
            rs.append(total_reward)
        else:
            s1 = s2
    if i >= windows:
        avg = np.mean(rs)
        avg_reward.append(avg)
        if avg > best_avg_reward:
            best_avg_reward = avg
            #agent.q_network.save("dqn_cartpole.h5")
    else: 
        avg_reward.append(8)
    
    print("\rEpisode {}/{} || Best average reward {}, Current Iteration Reward {}".format(i, ITERATIONS, best_avg_reward, total_reward) , end='', flush=True)

#np.save("rewards", np.asarray(rewards))
#np.save("averages", np.asarray(avg_reward))
plt.ylim(0,200)
plt.plot(rewards, color='olive', label='Reward')
plt.plot(avg_reward, color='red', label='Average')
plt.legend()
plt.ylabel('Reward')
plt.xlabel('Generation')
plt.show()
