import gym
import colorama
import numpy as np
import random
import matplotlib.pyplot as plt
import math
from collections import deque, defaultdict

class Q_Table_Agent(object):
    def __init__(self, action_size):
        self.table = defaultdict(lambda: np.zeros(action_size)) # The actual table is just a dictionary
        self.action_space = action_size
        # Q Learning Parameters
        self.gamma = 0.9 # DISCOUNT FACTOR, CLOSE TO 1 = LONG TERM
        self.epsilon = 1.0
        self.alpha = 0.05 # LEARNING RATE
        self.iter = 0

    def get_action(self, obs):
        if obs in self.table:
            if random.random() < self.epsilon: 
                return np.random.choice(self.action_space)
            else:
                return np.argmax(self.table[obs])
        else:
            return np.random.choice(self.action_space)

    def update_table(self, state1, action, reward, state2, iteration, done):
        if done:
            q2 = 0
            self.iter += 1
            self.epsilon = 1. / self.iter # Decreasing epislon over time
        else:
            q2 = self.table[state2][self.get_action(state2)] # THIS IS THE KEY DIFFERENCE
        self.table[state1][action] += (self.alpha) * (reward + self.gamma*q2-self.table[state1][action]) # BELLMAN EQUATION


# Hyperparameters
ITERATIONS = 20000
windows = 100

# This is because I am on windows and the rendering looks weird if you don't have it. 
colorama.init()
# This is the standard stuff for Open AI Gym. Be sure to check out their docs if you need more help.
env = gym.make("Taxi-v3").env
env.render()

# ACTION SPACE OF 6 (one int)
# OBS SPACE OF 500 (one int)
agent = Q_Table_Agent(env.action_space.n)
rewards = []

avg_reward = deque(maxlen=ITERATIONS)
best_avg_reward = -math.inf
rs = deque(maxlen=windows)

for i in range(ITERATIONS):
    s1 = env.reset()
    total_reward = 0
    done = False
    #print(agent.epsilon)
    while not done:
        action = agent.get_action(s1)
        s2, reward, done, info = env.step(action)
        total_reward += reward
        agent.update_table(s1, action, reward, s2, i+1, done)
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
        avg_reward.append(-1000)
    
    print("\rEpisode {}/{} || Best average reward {}".format(i, ITERATIONS, best_avg_reward), end='', flush=True)
    if best_avg_reward >= 9.7:
        print('\nEnvironment solved in {} episodes.'.format(i), end="", flush=True)
        break

print(rewards[19900:])
plt.plot(rewards, color='olive', label='Reward')
plt.plot(avg_reward, color='red', label='Average')
plt.legend()
plt.ylabel('Reward')
plt.xlabel('Generation')
plt.show()
