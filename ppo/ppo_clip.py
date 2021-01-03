import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import math
from collections import deque
import tensorflow as tf

#tf.keras.backend.set_floatx('float64')

class PPO_agent(object):
    def __init__(self, action_size, state_size):
        self.action_space = action_size
        self.state_space = state_size[0]
        self.e = 0.25 # Policy distance
        self.actor, self.critic = self.make_net(state_size)
        self.gamma = 0.99 # DISCOUNT FACTOR, CLOSE TO 1 = LONG TERM
        self.K = 4 # Number of epochs
        self.T = 2048 # Horizon
        self.M = 64 # Batch size
        self.ent = 0.01
        self.states = np.zeros((self.T, self.state_space))
        self.rewards = np.zeros((self.T, 1))
        self.actions = np.zeros((self.T, 1))
        self.log_probs = np.zeros((self.T, self.action_space))
        self.iter = 0
        self.policy_opt = tf.keras.optimizers.Adam(lr=3e-4)
        self.critic_opt = tf.keras.optimizers.Adam(lr=1e-3)

    def entropy(self, probs):
        return tf.reduce_mean(-probs * tf.math.log(probs))

    def ppo_loss(self, cur_pol, old_pol, advantages):
        #print(tf.math.log(cur_pol), old_pol, advantages)
        ratio = tf.math.exp(tf.math.log(cur_pol) - old_pol)
        ratio = tf.clip_by_value(ratio, 1e-10, 10-1e-10)
        #print(ratio)
        clipped = tf.clip_by_value(ratio, 1-self.e, 1+self.e)
        #print(clipped * advantages, ratio*advantages)
        loss = -tf.reduce_mean(tf.math.minimum(ratio * advantages, clipped * advantages)) + self.ent * self.entropy(cur_pol)
        #print(loss)
        #input()
        return loss

    def make_net(self, state):
        inputs = tf.keras.layers.Input(shape=(state))
        last_init = tf.random_uniform_initializer(minval=-0.001, maxval=0.001)
        value = tf.keras.layers.Dense(16, activation='relu')(inputs)
        value = tf.keras.layers.Dense(128, activation='relu')(value)
        value = tf.keras.layers.Dense(16, activation='relu')(value)
        value = tf.keras.layers.Dense(1)(value)
        policy = tf.keras.layers.Dense(32, activation='relu')(inputs)
        policy = tf.keras.layers.Dense(64, activation='relu')(policy) 
        policy = tf.keras.layers.Dense(32, activation='relu')(policy)
        policy = tf.keras.layers.Dense(self.action_space, activation='softmax', kernel_initializer=last_init)(policy)
        v_model = tf.keras.models.Model(inputs=inputs, outputs=value)
        p_model = tf.keras.models.Model(inputs=inputs, outputs=policy)
        v_model.summary()
        p_model.summary()
        return p_model, v_model

    def remember(self, state, reward, action, probs):
        i = self.iter
        self.states[i] = state
        self.rewards[i] = reward
        self.actions[i] = action
        self.log_probs[i] = tf.math.log(probs)
        self.iter += 1

    def get_action(self, obs):
        probs = self.actor(np.array([obs])).numpy()[0]
        action = np.random.choice(self.action_space, p=probs)
        # print(action, value, probs)
        return action, probs

    def discount_reward(self, rewards):
        d_rewards = np.zeros_like(rewards)
        Gt = 0
        # Discount rewards
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                Gt = 0
            else:
                Gt = rewards[i] + self.gamma * Gt
            d_rewards[i] = Gt

        return d_rewards

    def train(self):
        state_batch = tf.convert_to_tensor(self.states[:self.iter])
        log_batch = tf.convert_to_tensor(self.log_probs[:self.iter])
        action_batch = tf.convert_to_tensor(self.actions[:self.iter])
        action_batch = [[i, action_batch[i][0]] for i in range(len(action_batch))]
        log_batch = tf.cast(log_batch, dtype=tf.float32)
        action_batch = tf.cast(action_batch, dtype=tf.int32)

        rewards = self.discount_reward(self.rewards[:self.iter])

        for _ in range(self.K):
            with tf.GradientTape() as value_tape, tf.GradientTape() as policy_tape:
                value_pred = self.critic(state_batch, training=True)
                critic_loss = tf.math.reduce_mean(tf.math.square(value_pred - rewards))
                advantages = rewards - value_pred
                advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
                policy_pred = self.actor(state_batch, training=True)
                policy_loss = self.ppo_loss(tf.gather_nd(policy_pred, action_batch), tf.gather_nd(log_batch, action_batch), tf.squeeze(advantages))
            
            critic_grads = value_tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_opt.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
            policy_grads = policy_tape.gradient(policy_loss, self.actor.trainable_variables)
            self.policy_opt.apply_gradients(zip(policy_grads, self.actor.trainable_variables))

        self.iter = 0

if __name__ == '__main__':
    ITERATIONS = 300
    windows = 20

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
            action, p = agent.get_action(s1)
            s2, reward, done, info = env.step(action)
            total_reward += reward
            agent.remember(s1, reward, action, p)
            s1 = s2
        agent.train()
        rewards.append(total_reward)
        rs.append(total_reward)
        avg = np.mean(rs)
        avg_reward.append(avg)
        if i >= windows:
            if avg > best_avg_reward:
                best_avg_reward = avg
        
        print("\rEpisode {}/{} || Best average reward {}, Current Average {}, Current Iteration Reward {}".format(i, ITERATIONS, best_avg_reward, avg, total_reward), end='', flush=True)
        i += 1

    #np.save("rewards", np.asarray(rewards))
    #np.save("averages", np.asarray(avg_reward))
    #plt.ylim(-350,200)
    plt.plot(rewards, label='Reward')
    plt.plot(avg_reward, label='Average')
    plt.legend()
    plt.ylabel('Reward')
    plt.xlabel('Iteration')
    plt.show()
