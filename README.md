# RL-Implementation

From scratch implementations of prominent reinforcement learning techniques using using tensorflow/keras, Open AI Gym and potentially other environmemts. I will link all the papers and information in each folder. The code will be moderately commented, enough to hopefully understand what is going on. Each folder will also contain the trained model for an example task. 

This repository is meant to be an educational resource, both for those that visit it and for myself. I find that just reading papers can be hard to fully understand and implementing algorithms in code greatly helps my understanding. This repo isn't meant to be actually used like a package (if you want to download existing implementation of the algorithms there are many other great resources). Rather this is just documentation of my journey from early to contemporary techniques in reinforcement learning.

# Repo organization

Each algorithm will have its own folder that will be created when my implementation is created. The environments folder contains any environments that I have made from scratch. 

# Implementations

Below is a list of algorithms I intended to make (the checked ones have already been implemented).

- [X] Q-Table/SARSA
- [ ] DQN
- [ ] Double DQN
- [ ] Dueling DQN
- [ ] Full DQN (Double Dueling DQN with Prioritized Experience Replay)
- [ ] Deep Deterministic Policy Gradient (DDPG)
- [ ] Proximal Policy Optimization (PPO)
- [ ] Trust Region Policy Optimization (TRPO)
- [ ] Asynchronous Advantage Actor Critic (A3C)
- [ ] Twin Delayed Deep Deterministic Policy Gradients (TD3)
- [ ] Soft Actor Critic (SAC)
- [ ] AlphaZero (see https://github.com/lockwo/Learning_and_Advanced_Game_AI/tree/master/final)
- [ ] MARL? (I know very little about this, but I am very interested in learning, but it will probably be later)

# Notes for each implementation

## Q-Table:

This is the simplest implementation. Basically you make a 'table' that has states and actions and at each entry is the value of that action for that state. Uses the famous Bellman equation to update. 

Regularly reaches >8 in 20,000 iterations. I haven't really tested the hyperparameters much, but the same implementation is used to solve it in 1,000 iterations, so feel free to play with those. There is an exmaple reward graph in the folder. I didn't include a saved version because "training" from scratch takes like 2 minutes. 

## DQN:

DQN stands for Deep Q Network and it is very similar to Q learning, however, it uses a neural network. It is easy to see where Q learning fails. If you have a large number of states, although Q learning will converge, it may take an extraordinary amount of time. Where the Q-Table told use the value of actions at a given state, now our neural network will tell us this. There are convergence problems in this (which need not converge like the Q-Table), but that is where other factors will come in.
