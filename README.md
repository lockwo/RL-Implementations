# RL-Implementation

This repository contains from scratch implementations of prominent reinforcement learning techniques using using tensorflow 2, Open AI Gym and potentially other environmemts. I provide the PDFs of all papers in the associated folder. I usually just test the algorithms on simple environments (i.e. not atari) due to my limited access to computational power, however, I sometimes uplaod CNN versions and I may upload atari versions in the future. 

This repository is meant to be an educational resource. I find that just reading papers can be hard to fully understand and implementing algorithms in code greatly helps my understanding. This repo is meant to feature the simplest possible implementations of prominent RL algorithms that are easy to understand. 

# Repo organization

Each algorithm will have its own folder. The environments folder contains any environments that I have made from scratch. I will only upload easily trainable models or contemporary models. I do run all programs and train them briefly, to verify they are (mostly) correct.

# Algorithms

## Model Free Algorithms

### Q/Value Methods

- [X] Q-Table/SARSA
- [X] Deep Q Network (DQN)
- [X] Double DQN
- [X] Dueling DQN
- [X] Boostrapped DQN
- [ ] Full DQN (n-Step Bootstrapped Double Dueling DQN with Prioritized Experience Replay)

### Policy Methods

- [X] REINFORCE/VPG

### Actor-Critic Methods

- [ ] Deep Deterministic Policy Gradient (DDPG)
- [ ] Twin Delayed Deep Deterministic Policy Gradients (TD3)
- [X] Advantage Actor Critic (A2C)
- [ ] Actor Critic with Experience Replay (ACER)
- [ ] Actor Critic using Kronecker-Factored Trust Region (ACKTR)
- [ ] Trust Region Policy Optimization (TRPO)
- [X] Proximal Policy Optimization (PPO)
- [ ] Soft Actor Critic (SAC)

## Model-Based Algorithms

### Tree Search Methods

- [X] Alpha-Beta Tree Search
- [ ] AlphaZero (see https://github.com/lockwo/Learning_and_Advanced_Game_AI/tree/master/final) Not exact, but I will improve later
- [ ] MuZero

## Exploration Methods

- [ ] Intrinsic Curiosity Module (ICM) 

## Hierarchical Methods

- [ ] FeUdal Network (FuN)

## Meta-RL Methods

- [ ] RL^2
- [ ] Model-Agnostic Meta-Learning (MAML)

# Graph Comparisons

See Data Visual.

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

## REINFORCE:

This is the straight forward Monte Carlo Policy Gradient Method. Fixed the bug using a custom loss function. Regardless, in this algorithm the ANN represents the policy rather than the Q prediction. That means that the network tells you exactly what the probabilities of each action are (which you then choose from). You update the policy by doing discounted reward at the end of each policy rollout. 

## Advantage Actor Critic (A2C):

This algorithm is very similar to the REINFORCE algorithm. The policy network in REINFORCE is updated via <img src="https://github.com/lockwo/RL-Implementation/blob/master/equations/reinforce.PNG" width="350" height="60">, but this discounted reward is replaced in A2C with an estimate of the advantage, <img src="https://github.com/lockwo/RL-Implementation/blob/master/equations/a2c.PNG" width="200" height="50">. Remember from dueling networks that Q(s,a) = V(s) + A(s,a). Thus A(s,a) = Q(s,a) - V(s). This means that a single network can predict V(s) and the advtange can be dervied from this. Thus the policy and value are different heads of the same network. Although there are asynchronous versions of this, they offer few improvements, thus I implement it synchronously.

## Proximal Policy Optimization (PPO):

PPO is one of the most popular and successfuly reinforcement learning algorithms. It has achieved amazing things, such as superhuman Dota 2 performance, and remains straightforward to understand and implement. There a multiple kinds of PPO, the one implemented here is PPO-Clip but PPO-penalty (KL) is another common version. As an actor critic method, the critic is trained like any other (i.e. with a value network), but the policy is updated with the loss <img src="https://github.com/lockwo/RL-Implementations/blob/master/equations/loss.PNG" width="200" height="50">. This looks confusing but it is relatively straight forward. The policy is updated with the smallest step (via the min operator) to similar to the mechanism of a trust region, preventing too large of step sizes. The first term is basically the same as vanilla policy gradient, the difference in policies times the advantage. The second term is clipping the policy difference such that if the first term wants to take a large step size it cannot. In this case, the value and policy networks are completely separated and such do not share a loss function. Also there is commonly an entropy term on the loss function, which I will add at a later point. 

## Alpha-Beta Pruning:

Alpha-Beta pruning is a tree search algorithm for turn based 2 player games, in the case of my implementation: chess. It is very similar to minimax, except that it "prunes" certain nodes to increase computational efficiency. There is a maximizing player and a minimizing player, and you go through a normal tree search, but keep track of alpha and beta. Alpha begins at -infinity and represents the best score you can attain as a maximizng player (and the opposite for beta). This then gets adjusted as you go through the search. If a path cannot get above alpha, then it is pruned (for the max player). 

Because of the depth complexity of chess, a complete tree search is not feasible (i.e. we cannot tree search to the end of the game). Thus an evaluation function is needed. I did not write mine, I pieced it together from different sources. It is not very good and as such you will often find that the computer does not make very good moves because it values things that it shouldn't. Improvements to the heuristic evaluation are left as an exercise for the user.

# Reinforcement Learning for the World

<a href="https://info.flagcounter.com/lhWB"><img src="https://s11.flagcounter.com/countxl/lhWB/bg_FFFFFF/txt_000000/border_CCCCCC/columns_3/maxflags_12/viewers_0/labels_0/pageviews_1/flags_0/percent_0/" alt="Flag Counter" border="0"></a>
