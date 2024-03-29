import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import math
from collections import deque
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

def _entropy_loss(target, output):
    return -tf.math.reduce_mean(target*tf.math.log(tf.clip_by_value(output,1e-10,1-1e-10)))

class A2C_agent(object):
    def __init__(self, action_size, state_size):
        self.action_space = action_size
        self.state_space = state_size[0]
        self.actor, self.critic = self.make_net(state_size)
        # Q Learning Parameters
        self.gamma = 0.98 # DISCOUNT FACTOR, CLOSE TO 1 = LONG TERM
        self.states, self.rewards, self.values, self.actions = [], [], [], []

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
        p_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=_entropy_loss)
        return p_model, v_model

    def remember(self, state, action, reward, value):
        self.states.append(state)
        self.rewards.append(reward)
        self.values.append(value)
        self.actions.append(action)

    def get_action(self, obs):
        temp = self.actor.predict(np.array([obs,]))
        probs = temp[0]
        action = np.random.choice(self.action_space, p=probs)
        temp = self.critic.predict(np.array([obs,]))
        value = temp[0]
        return action, value

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
                Gt = rewards[i] + self.gamma * self.values[i + 1]
                #Gt = done * Gt * self.gamma + rewards[i]
            d_rewards[i] = Gt

        return d_rewards

    def train(self):
        batch_len = len(self.states)
        state = np.zeros(shape=(batch_len, self.state_space))
        action = np.zeros(shape=(batch_len, self.action_space))
        
        rewards = self.discount_reward(self.rewards)

        for i in range(batch_len):
            state[i] = self.states[i]
            action[i][self.actions[i]] = rewards[i] - self.values[i]
        
        self.actor.fit(state, action, epochs=1, verbose=0)
        self.critic.fit(state, rewards, epochs=1, verbose=0)
        self.states.clear()
        self.rewards.clear()
        self.values.clear()
        self.actions.clear()


# Hyperparameters
ITERATIONS = 1000
windows = 50

env = gym.make("LunarLander-v2")
'''env.observation_space.shape'''
print(env.action_space)
print(env.observation_space, env.observation_space.shape)
agent = A2C_agent(env.action_space.n, env.observation_space.shape)
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
        action, value = agent.get_action(s1)
        s2, reward, done, info = env.step(action)
        total_reward += reward
        agent.remember(s1, action, reward, value)
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
        avg_reward.append(-200)
    
    print("\rEpisode {}/{} || Best average reward {}, Current Iteration Reward {}".format(i, ITERATIONS, best_avg_reward, total_reward) , end='', flush=True)

#np.save("rewards", np.asarray(rewards))
#np.save("averages", np.asarray(avg_reward))
plt.ylim(-250,250)
plt.plot(rewards, color='olive', label='Reward')
plt.plot(avg_reward, color='red', label='Average')
plt.legend()
plt.ylabel('Reward')
plt.xlabel('Generation')
plt.show()
