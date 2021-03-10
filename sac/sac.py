import argparse
import gym
import pybulletgym
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import deque
import tensorflow as tf
import tensorflow_probability as tfp

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class Policy(tf.keras.Model):
    def __init__(self, n, action_spaces, a_range) -> None:
        super(Policy, self).__init__()
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        self.lay1 = tf.keras.layers.Dense(256, activation='relu')
        self.lay2 = tf.keras.layers.Dense(256, activation='relu')
        self.mu = tf.keras.layers.Dense(action_spaces, kernel_initializer=last_init)
        self.std = tf.keras.layers.Dense(action_spaces, activation='softplus')
        self.noise = n
        self.a_range = a_range

    def call(self, state):
        p1 = self.lay1(state)
        p2 = self.lay2(p1)
        mu = self.mu(p2)
        sigma = self.std(p2)
        sigma = tf.clip_by_value(sigma, self.noise, 1.0)

        return mu, sigma

    def sample_normal(self, state):
        means, stds = self.call(state)

        probs = tfp.distributions.Normal(means, stds)

        actions = probs.sample()
        action = tf.math.tanh(actions) * self.a_range

        log_probs = probs.log_prob(actions)
        log_probs -= tf.math.log(1 - tf.math.pow(action, 2) + self.noise)
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)
        return action, log_probs

class SAC(object):
    def __init__(self, action_size, state_size):
        self.action_space = action_size[0]
        self.state_space = state_size[0]
        self.act_range = 1
        self.noise = 1e-6
        self.alpha = tf.Variable(0.0, dtype=tf.float32) # Entropy
        self.alpha_train = True
        self.q1 = self.make_critic()
        self.q2 = self.make_critic()
        self.q1_target = self.make_critic()
        self.q2_target = self.make_critic()
        self.policy = Policy(self.noise, self.action_space, self.act_range)
        self.gamma = 0.99 # Discount Factor
        self.target_entropy = -tf.constant(self.action_space, dtype=tf.float32)
        self.log_stds = -1 * np.ones(self.action_space)
        self.move_weights()
        self.batch = 100
        self.buff = int(1e6)
        self.states = np.zeros((self.buff, self.state_space))
        self.actions = np.zeros((self.buff, self.action_space))
        self.rewards = np.zeros((self.buff, 1))
        self.dones = np.zeros((self.buff, 1))
        self.next_states = np.zeros((self.buff, self.state_space))
        self.log_probs = np.zeros((self.buff, 1))
        self.iter = 0
        self.counter = 0
        self.training = 2
        self.policy_opt = tf.keras.optimizers.Adam(lr=3e-4)
        self.alpha_opt = tf.keras.optimizers.Adam(lr=3e-4)
        self.critic_opt = tf.keras.optimizers.Adam(lr=3e-4)
        self.tau = 0.005

    def move_weights(self):
        self.q1_target.set_weights(self.q1.get_weights())
        self.q2_target.set_weights(self.q2.get_weights())

    def make_critic(self):
        state_ = tf.keras.layers.Input(shape=(self.state_space))
        action_ = tf.keras.layers.Input(shape=(self.action_space))
        x = tf.keras.layers.Concatenate()([state_, action_])
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(1, name='output')(x)
        model = tf.keras.models.Model(inputs=[state_, action_], outputs=x)
        return model
    
    def remember(self, state, action, reward, next_state, done, probs):
        i = self.counter
        self.states[i] = state
        self.rewards[i] = reward
        self.actions[i] = action
        self.next_states[i] = next_state
        self.log_probs[i] = probs
        self.dones[i] = int(done)
        self.counter += 1

    def get_action(self, obs):
        state = tf.convert_to_tensor([obs])
        actions, logs = self.policy.sample_normal(state)
        return actions[0], logs

    @tf.function
    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def train(self):
        if math.isnan(self.alpha.numpy()):
            print("\n ERROR \n")
            self.alpha = tf.Variable(0.0, dtype=tf.float32)
            self.alpha_train = False

        batch_indices = np.random.choice(min(self.counter, self.buff), self.batch)
        state_batch = tf.convert_to_tensor(self.states[batch_indices], dtype=tf.float32)
        action_batch = tf.convert_to_tensor(self.actions[batch_indices], dtype=tf.float32)
        reward_batch = tf.convert_to_tensor(self.rewards[batch_indices], dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_states[batch_indices], dtype=tf.float32)
        dones_batch = tf.convert_to_tensor(self.dones[batch_indices], dtype=tf.float32)
        log_batch = tf.convert_to_tensor(self.log_probs[batch_indices], dtype=tf.float32)
        
        with tf.GradientTape(persistent=True) as critic_tape:
            actions, log_probs = self.policy.sample_normal(state_batch)
            values = tf.math.minimum(self.q1_target([next_state_batch, actions]), self.q2_target([next_state_batch, actions])) - self.alpha * log_probs
            bellman = reward_batch + (1 - dones_batch) * self.gamma * values
            critic1 = self.q1([state_batch, action_batch], training=True)
            critic2 = self.q2([state_batch, action_batch], training=True)
            msbe1 = tf.math.reduce_mean(tf.math.square(bellman - critic1))
            msbe2 = tf.math.reduce_mean(tf.math.square(bellman - critic2))
    
        critic1_gradients = critic_tape.gradient(msbe1, self.q1.trainable_variables)
        self.critic_opt.apply_gradients(zip(critic1_gradients, self.q1.trainable_variables))
        critic2_gradients = critic_tape.gradient(msbe2, self.q2.trainable_variables)
        self.critic_opt.apply_gradients(zip(critic2_gradients, self.q2.trainable_variables))

        if self.iter % 1000 == 0:
            self.update_target(self.q1_target.trainable_variables, self.q1.trainable_variables, 1)
            self.update_target(self.q2_target.trainable_variables, self.q2.trainable_variables, 1)
        else:
            self.update_target(self.q1_target.trainable_variables, self.q1.trainable_variables, self.tau)
            self.update_target(self.q2_target.trainable_variables, self.q2.trainable_variables, self.tau)

        with tf.GradientTape() as policy_tape:
            actions, log_probs = self.policy.sample_normal(state_batch)
            qs = tf.math.minimum(self.q1([state_batch, actions]), self.q2([state_batch, actions]))
            critic = self.alpha * log_probs - qs
            policy_loss = tf.math.reduce_mean(critic)

        policy_grads = policy_tape.gradient(policy_loss, self.policy.trainable_variables)
        self.policy_opt.apply_gradients(zip(policy_grads, self.policy.trainable_variables))

        if self.alpha_train:
            with tf.GradientTape() as alpha_tape:
                alpha_loss = tf.reduce_mean(-self.alpha * (log_batch + self.target_entropy))

            variables = [self.alpha]
            grads = alpha_tape.gradient(alpha_loss, variables)
            self.alpha_opt.apply_gradients(zip(grads, variables))

        self.iter += 1

if __name__ == '__main__':
    # Hyperparameters
    steps = int(2e5)
    windows = 50
    learn_delay = 10000

    #env = gym.make("InvertedPendulumPyBulletEnv-v0")
    #env = gym.make("HumanoidFlagrunPyBulletEnv-v0")
    #env = gym.make("Pendulum-v0")
    env = gym.make("HalfCheetahPyBulletEnv-v0")
    '''env.observation_space.shape'''
    print(env.action_space, env.action_space.shape)
    print(env.observation_space, env.observation_space.shape)
    minn = -1
    maxx = 1
    agent = SAC(env.action_space.shape, env.observation_space.shape)
    rewards = []
    avg_reward = []
    best_avg_reward = -math.inf
    rs = deque(maxlen=windows)
    i = 0
    step = 0
    while True:
        s1 = env.reset()
        total_reward = 0
        done = False
        while not done:
            #env.render()
            action, p = agent.get_action(s1)
            s2, reward, done, info = env.step(action)
            total_reward += reward
            agent.remember(s1, action, reward, s2, done, p)
            if agent.counter > learn_delay and step % agent.training == 0:
                agent.train()
            s1 = s2
            step += 1
        rs.append(total_reward)
        avg = np.mean(rs)
        avg_reward.append(avg)
        rewards.append(total_reward)
        if avg > best_avg_reward:
            best_avg_reward = avg
        
        print("\rStep {}/{} Iteration {} || Alpha {} Best average reward {}, Current Average {}, Current Iteration Reward {}"\
            .format(step, steps, i, agent.alpha.numpy(), best_avg_reward, avg, total_reward), end='', flush=True)
        i += 1
        if step >= steps:
            break


    #np.save("rewards", np.asarray(rewards))
    #np.save("averages", np.asarray(avg_reward))
    #plt.ylim(-350,200)
    plt.plot(rewards, label='Reward')
    plt.plot(avg_reward, label='Average')
    plt.legend()
    plt.ylabel('Reward')
    plt.xlabel('Iteration')
    plt.show()
