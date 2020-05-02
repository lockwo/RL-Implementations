# RL-Implementation

From scratch implementations of prominent reinforcement learning techniques using using tensorflow/keras, Open AI Gym and potentially other environmemts. I will link all the papers and information in each folder. The code will be moderately commented, enough to hopefully understand what is going on. Each folder will also contain the trained model for an example task. 

This repository is meant to be an educational resource, both for those that visit it and for myself. I find that just reading papers can be hard to fully understand and implementing algorithms in code greatly helps my understanding. This repo isn't meant to be actually used like a package (if you want to download existing implementation of the algorithms there are many other great resources). Rather this is just documentation of my journey from early to contemporary techniques in reinforcement learning.

# Repo organization

Each algorithm will have its own folder that will be created when my implementation is created. The environments folder contains any environments that I have made from scratch. I will only upload easily trainable models or contemporary models. Due to my neophytic parallel programming skills, the way in which I would train things like DQN involve running many simultaneous game generation programs, storing that information, then training, alternating these batches in 6 hours shifts (that is the maximum job time). This requires code modification and aditional work, hence why I do not do it for all. I do run all programs and train them briefly, to verify they are (mostly) correct, but the likely ones to have trained models are full dqn, ddpg, ppo, and sac (maybe).

# Implementations

Below is a list of algorithms I intend to make (the checked ones have already been implemented).

- [X] Q-Table/SARSA
- [X] DQN
- [X] Double DQN
- [X] Dueling DQN
- [ ] Full DQN (Double Dueling DQN with Prioritized Experience Replay)
- [ ] Deep Deterministic Policy Gradient (DDPG)
- [ ] Twin Delayed Deep Deterministic Policy Gradients (TD3)
- [ ] REINFORCE/VPG
- [ ] Proximal Policy Optimization (PPO) / Trust Region Policy Optimization (TRPO)
- [ ] Asynchronous Advantage Actor Critic (A3C)
- [ ] Soft Actor Critic (SAC)
- [ ] World Models
- [ ] AlphaZero (see https://github.com/lockwo/Learning_and_Advanced_Game_AI/tree/master/final) Not exact, but I will improve later
- [ ] MuZero
- [X] Alpha-Beta Tree Search
- [ ] MARL? (I know very little about this, but I am very interested in learning, but it will probably be later)

# Notes for each implementation

## Q-Table:

This is the simplest implementation. Basically you make a 'table' that has states and actions and at each entry is the value of that action for that state. Uses the famous Bellman equation to update. 

Regularly reaches >8 in 20,000 iterations. I haven't really tested the hyperparameters much, but the same implementation is used to solve it in 1,000 iterations, so feel free to play with those. There is an exmaple reward graph in the folder. I didn't include a saved version because "training" from scratch takes like 2 minutes. 

SARSA or State-action-reward-state-action, is very similar to Q learning except that it is on policy. What does that mean? That means that in the updating of SARSA the next action is chosen as it would be if it were real (i.e. however you chose your actions, in my case epsilon greedy, apply that to your choosing). Q Learning just takes the max valued action (even if it wouldn't actually take that in a real simulation). There is SARSA code (it is almost identical) and it performs comparably on this task. 

## DQN:

DQN stands for Deep Q Network and it is very similar to Q learning, however, it uses a neural network. It is easy to see where Q learning fails. If you have a large number of states, although Q learning will converge, it may take an extraordinary amount of time. Where the Q-Table told use the value of actions at a given state, now our neural network will tell us this. There are convergence problems in this (which need not converge like the Q-Table), but that is where other factors will come in.

The DQN is a vanilla dense dqn, without hyperparameter optimization. It works for cartpole and acrobot, and the same approach could work for RAM versions of atari games, but for the atari games (in order to get as close to human interaction as possible), I will be using the methods linked in the papers to play atari games (aka CNN rather than dense network).

One of the strengths (in terms of computational efficiency) is DQN's ability to learn from the past. Because the network basically maps (s,a) pairs to associated values, any experience is good because (even if it is bad gameplay) it can learn from it. 

Due to the training time of this (and likely future versions) I don't provide a trained bot. I hopefully will, once I figure out how to use my computational resources. However, until then, I will only upload trained bots for current techniques (i.e. things that are still frequently used, like PPO). Considering that in the paper they had to train for >40 hours which is not something I have (especially since they have more parallel computing knowledge, but I will certianly work on it). However, I did just fix a bug that might enable me to train it much faster.


## Double DQN:

Double DQN is very similar to DQN. The difference is that double dqn has two neural networks that predict the Q values. The main network and the target network. The target network is used only to predict the future Q value in the Bellman equation and it updated via Polyak averaging every training step and a direct copying of weights every certain amount. The reason for having this target network is to increase stability and likelihood of convergance. Sometimes vanilla DQN trains to quickly and fails to converge on the correct solution and this target network is a way to minimize this failure. 

## Dueling DQN:

Another improvement on the vanilla DQN, this relies on the understanding of exactly what a Q value is. A Q value represents the value of that state in addition to the future potential value. These can be separated into their own neural network layers before being recombined. This is done so the network can learn to predict the value and action calculations independently to improve each. This is a very intuitive improvement and adds very little in terms of lines of code. 

## Alpha-Beta Pruning:

Alpha-Beta pruning is a tree search algorithm for turn based 2 player games, in the case of my implementation: chess. It is very similar to minimax, except that it "prunes" certain nodes to increase computational efficiency. There is a maximizing player and a minimizing player, and you go through a normal tree search, but keep track of alpha and beta. Alpha begins at -infinity and represents the best score you can attain as a maximizng player (and the opposite for beta). This then gets adjusted as you go through the search. If a path cannot get above alpha, then it is pruned (for the max player). 

Because of the depth complexity of chess, a complete tree search is not feasible (i.e. we cannot tree search to the end of the game). Thus an evaluation function is needed. I did not write mine, I pieced it together from different sources. It is not very good and as such you will often find that the computer does not make very good moves because it values things that it shouldn't. Improvements to the heuristic evaluation are left as an exercise for the user.

