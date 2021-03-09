import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from collections import deque
import pybulletgym  # register PyBullet enviroments with open ai gym

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class TD3(object):
    def __init__(self, action_size, state_size):
        self.action_space = action_size
        self.state_space = state_size
        self.act_range = 1.0
        self.actor_opt = tf.keras.optimizers.Adam(lr=1e-3)
        self.critic_opt = tf.keras.optimizers.Adam(lr=1e-3)
        self.q1 = self.make_critic()
        self.q2 = self.make_critic()
        self.q1_target = self.make_critic()
        self.q2_target = self.make_critic()
        self.policy = self.make_actor()
        self.policy_target = self.make_actor()
        #self.policy.summary()
        #self.q1.summary()
        self.policy_counter = 0
        self.move_weights()
        self.buff = int(1e6)
        self.states = np.zeros((self.buff, self.state_space[0]))
        self.actions = np.zeros((self.buff, self.action_space[0]))
        self.rewards = np.zeros((self.buff, 1))
        self.dones = np.zeros((self.buff, 1))
        self.next_states = np.zeros((self.buff, self.state_space[0]))
        self.counter = 0
        self.batch = 100
        self.gamma = 0.99 
        self.tau = 0.005
        self.act_noise = 0.1 * self.act_range
        self.target_noise = 0.2 * self.act_range
        self.noise_clip = 0.5
        self.policy_delay = 2
     
    def make_critic(self):
        state_ = tf.keras.layers.Input(shape=(self.state_space[0]))
        action_ = tf.keras.layers.Input(shape=(self.action_space[0]))
        x = tf.keras.layers.Concatenate()([state_, action_])
        x = tf.keras.layers.Dense(400, activation='relu')(x)
        x = tf.keras.layers.Dense(300, activation='relu')(x)
        x = tf.keras.layers.Dense(self.action_space[0], name='output')(x)
        model = tf.keras.models.Model(inputs=[state_, action_], outputs=x)
        return model
    
    def make_actor(self):
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        state_ = tf.keras.layers.Input(shape=(self.state_space[0]))
        x = tf.keras.layers.Dense(400, activation='relu')(state_)
        x = tf.keras.layers.Dense(300, activation='relu')(x)
        x = tf.keras.layers.Dense(self.action_space[0], activation='tanh', name='output', kernel_initializer=last_init)(x)
        x = x * self.act_range
        model = tf.keras.models.Model(inputs=state_, outputs=x)
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
        self.q1_target.set_weights(self.q1.get_weights())
        self.q2_target.set_weights(self.q2.get_weights())
        self.policy_target.set_weights(self.policy.get_weights())

    def get_action(self, obs):
        action = tf.squeeze(self.policy(np.array([obs])))
        act = action.numpy() + np.clip(np.random.normal(0, self.act_noise, 1)[0], -self.noise_clip, self.noise_clip)
        act = np.clip(act, -self.act_range, self.act_range)
        if isinstance(act, float):
            return [act]
        return act

    def train(self):
        batch_indices = np.random.choice(min(self.counter, self.buff), self.batch)
        state_batch = tf.convert_to_tensor(self.states[batch_indices])
        action_batch = tf.convert_to_tensor(self.actions[batch_indices])
        reward_batch = tf.convert_to_tensor(self.rewards[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_states[batch_indices])
        dones_batch = tf.convert_to_tensor(self.dones[batch_indices])
        dones_batch = tf.cast(dones_batch, dtype=tf.float32)
        
        with tf.GradientTape(persistent=True) as tape:
            targets_act = tf.clip_by_value(self.policy_target(next_state_batch) + tf.random.normal([self.batch, 1], 0, self.target_noise), -self.noise_clip, self.noise_clip)
            bellman = reward_batch + (1 - dones_batch) * self.gamma * tf.math.minimum(self.q1_target([next_state_batch, targets_act]), self.q2_target([next_state_batch, targets_act]))
            critic1 = self.q1([state_batch, action_batch])
            critic2 = self.q2([state_batch, action_batch])
            msbe1 = tf.math.reduce_mean(tf.math.square(bellman - critic1))
            msbe2 = tf.math.reduce_mean(tf.math.square(bellman - critic2))
        
        critic1_gradients = tape.gradient(msbe1, self.q1.trainable_variables)
        self.critic_opt.apply_gradients(zip(critic1_gradients, self.q1.trainable_variables))
        critic2_gradients = tape.gradient(msbe2, self.q2.trainable_variables)
        self.critic_opt.apply_gradients(zip(critic2_gradients, self.q2.trainable_variables))

        self.update_target(self.q1_target.trainable_variables, self.q1.trainable_variables)
        self.update_target(self.q2_target.trainable_variables, self.q2.trainable_variables)

        if self.policy_counter % self.policy_delay == 0:
            with tf.GradientTape() as tape:
                actions = self.policy(state_batch, training=True)
                #critic = tf.math.minimum(self.q1([state_batch, actions]), self.q2([state_batch, actions]))
                critic = self.q1([state_batch, actions])
                policy_loss = -tf.math.reduce_mean(critic)
            
            policy_gradients = tape.gradient(policy_loss, self.policy.trainable_variables)
            self.actor_opt.apply_gradients(zip(policy_gradients, self.policy.trainable_variables))

            self.update_target(self.policy_target.trainable_variables, self.policy.trainable_variables)
       
        self.policy_counter += 1
    
    @tf.function
    def update_target(self, target_weights, weights):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * self.tau + a * (1 - self.tau))

if __name__ == "__main__":
    # Hyperparameters
    steps = int(2e5)
    windows = 50
    learn_delay = int(1e4)

    env = gym.make("HalfCheetahPyBulletEnv-v0")
    #env = gym.make("HumanoidFlagrunPyBulletEnv-v0")
    '''env.observation_space.shape'''
    print(env.action_space, env.action_space.shape)
    print(env.observation_space, env.observation_space.shape)
    minn = -1
    maxx = 1
    agent = TD3(env.action_space.shape, env.observation_space.shape)
    rewards = []
    avg_reward = deque(maxlen=steps)
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
            if step < learn_delay:
                action = (maxx - minn) * np.random.random(env.action_space.shape) + minn
            else:
                action = agent.get_action(s1)
            s2, reward, done, info = env.step(action)
            total_reward += reward
            agent.remember(s1, action, reward, s2, done)
            if agent.counter > learn_delay:
                agent.train()
            s1 = s2
            step += 1
        rs.append(total_reward)
        avg = np.mean(rs)
        avg_reward.append(avg)
        if avg > best_avg_reward:
            best_avg_reward = avg
        
        print("\rStep {}/{} Iteration {} || Best average reward {}, Current Average {}, Current Iteration Reward {}".format(step, steps, i, best_avg_reward, avg, total_reward), end='', flush=True)
        i += 1
        if step >= steps:
            break

    #np.save("rewards", np.asarray(rewards))
    #np.save("tf_td3_ant_0", np.asarray(avg_reward))
    plt.plot(rewards, color='olive', label='Reward')
    plt.plot(avg_reward, color='red', label='Average')
    plt.legend()
    plt.title("Lunar Lander")
    plt.ylabel('Reward')
    plt.xlabel('Step')
    plt.show()
