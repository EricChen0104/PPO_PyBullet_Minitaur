# PPO OpenAI Gym Minitaur
This project implements a Proximal Policy Optimization (PPO) reinforcement learning agent to train the Minitaur robot to walk in the `MinitaurBulletEnv-v0` environment using PyBullet. The agent uses a multilayer perceptron (MLP) to model the policy and value networks and learns to control the robot in a continuous action space.

## DEMO
![](https://github.com/user-attachments/assets/8c6f88ac-d396-4f09-8316-79b777b29441)

## HOW TO RUN THE CODE
### Requirements
- Python 3.10+
- PyTorch
- gym
- pybullet
- matplotlib
- numpy

### Installation
```bash
git clone https://github.com/EricChen0104/PPO_PyBullet_Minitaur.git
cd PPO_PyBullet_Minitaur
```

## Algorithm Details
The PPO agent uses:
- Shared MLP backbone with 2 hidden layers 
- Gaussian action distribution with learned mean and log_std
- Tanh to bound actions in [-1, 1]
- GAE for advantage estimation
- Clipped surrogate objective for policy update

### Policy Network
- Input: 28-dim observation (Minitaur state)
- Actor: Two hidden layers: [128, 89], ReLU activations
- Critic: Two hidden layers: [89, 55], ReLU activations
- GAE calculation: <br/>
  ![](https://github.com/user-attachments/assets/bf26b6eb-4614-4a08-9471-eae84892a9e4)

### Reward
![](https://github.com/EricChen0104/PPO_PyBullet_Minitaur/blob/master/plot/ppo_training_curve.png?raw=true)

### Hyperparameters
| Parameter              | Value     |
|------------------------|-----------|
| Total steps            | 1,000     |
| Steps per rollout      | 4096      |
| PPO epochs             | 10        |
| Minibatch size         | 128       |
| Learning rate          | 3e-5      |
| γ (discount factor)    | 0.99      |
| λ (GAE lambda)         | 0.95      |
| Clip range (ε)         | 0.2       |
| Value loss coeff       | 0.5       |
| Entropy coeff          | 0.04      |

## Future Work
- [ ] Add observation normalization (e.g. running mean/std)
- [ ] Implement reward normalization
- [ ] Test with LSTM-based recurrent policies
- [ ] Add curriculum learning (e.g. with terrain or perturbation)

## References
- Tan, J., Zhang, T., Coumans, E., Iscen, A., Bai, Y., Hafner, D., ... & Vanhoucke, V. (2018). Sim-to-real: Learning agile locomotion for quadruped robots. arXiv preprint arXiv:1804.10332.
- Jadoon, N. A. K., & Ekpanyapong, M. (2025). Quadruped Robot Simulation Using Deep Reinforcement Learning--A step towards locomotion policy. arXiv preprint arXiv:2502.16401.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

