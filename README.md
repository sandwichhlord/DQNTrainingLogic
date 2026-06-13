# Hero Killer: Neural Combat AI 
An advanced 2D fighting game environment with an integrated Deep Q-Network (DQN) and Behavioral Cloning pipeline built in PyTorch and Pygame.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)
![Pygame](https://img.shields.io/badge/Pygame-Environment-green)


<video src="demo/model_gameplay.mp4" controls="controls" muted="muted" autoplay="autoplay" width="800"></video>
*(Gameplay of me against the trained model)*

## System Architecture
* **Custom Environment:** Built a custom gymnasium-style 2D fighting engine from scratch with pixel-perfect collision, frame-data limits, and relative state-space normalization.
* **League Training Matchmaking:** Implemented a dynamic training curriculum where the agent fights against a mixed pool of hardcoded bots, chaotic random agents, and historical snapshots of itself (Ghosts) to prevent policy cycling.
* **Behavioral Cloning:** Bypassed reward sparsity by utilizing supervised Imitation Learning, mapping expert human/bot states to discrete actions.

<video src="demo/model_training_against_opponent.mp4" controls="controls" muted="muted" autoplay="autoplay" width="800"></video>
*(The model learning how to play in the environment)*

<video src="demo/model_training_on_opponent.mp4" controls="controls" muted="muted" autoplay="autoplay" width="800"></video>
*(The model learning how to play based on an expert bots movements, i.e., behavioral cloning)*

## Quick Start
**1. Install Dependencies**
`pip install -r requirements.txt`

**2. Play Against the Boss AI**
Want to test your skills against the trained neural network?
`python -m scripts.play`

**3. Train a New Agent**
For an RL agent you want to teach from scratch against bots:-
`python -m scripts.train_against_opponent`
For behavioral cloning:-
`python -m scripts.train_on_opponent`