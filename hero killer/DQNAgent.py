import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# 1. THE BRAIN (Neural Network)
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        # A simple Feed-Forward Network
        # Input Layer (10 neurons) -> Hidden Layer (128 neurons)
        self.fc1 = nn.Linear(state_size, 128)
        self.relu = nn.ReLU() # Activation function (The "firing" logic)
        
        # Hidden Layer (128) -> Hidden Layer (128)
        self.fc2 = nn.Linear(128, 128)
        
        # Hidden Layer (128) -> Output Layer (6 Actions)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        # Pass the data through the layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x) # Returns raw Q-values (scores) for each action

# 2. THE AGENT (The Learner)
class DQNAgent:
    def __init__(self, state_size, action_size, training_mode=True):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters (The "Tuning Knobs")
        self.memory = deque(maxlen=2000) # Replay Buffer capacity
        self.gamma = 0.95    # Discount rate (How much we care about future rewards)
        if training_mode:
            self.epsilon = 1.0   # Exploration rate (1.0 = 100% Random at start)
        else:
            self.epsilon = 0.01   # Low exploration for watching the AI
        self.epsilon_min = 0.01 # Minimum exploration (1% random eventually)
        self.epsilon_decay = 0.995 # How fast we stop being random
        self.learning_rate = 0.0001
        self.batch_size = 64 # How many memories to learn from at once
 
        # Initialize the Brain
        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss() # Loss function (Mean Squared Error)

    def act(self, state):
        # EXPLORATION (Try random things)
        if random.random() <= self.epsilon:
            # WEIGHTED RANDOM LOGIC
            # We assign "Weights" to actions. Higher number = picked more often.
            # Actions: [Idle, Left, Right, Jump, Attack, Block]
            
            # We give 'Jump' (Index 3) a weight of 1.
            # We give everything else a weight of 2.
            # Total Weight = 11. Jump Probability = 1/11 (~9%)
            
            weights = [3, 3, 3, 1, 3, 3]
            action_indices = [0, 1, 2, 3, 4, 5]
            
            # random.choices returns a list, so we take [0]
            return random.choices(action_indices, weights=weights, k=1)[0]
        
        # EXPLOITATION (Use the Brain)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        
        return torch.argmax(q_values).item()
        
    def remember(self, state, action, reward, next_state, done):
        # Store the experience in the "Photo Album"
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        # TRAINING STEP (The "Dreaming" Phase)
        if len(self.memory) < self.batch_size:
            return None # Not enough memories yet

        # 1. Sample a random batch of memories
        minibatch = random.sample(self.memory, self.batch_size)
        
        # 2. Prepare the data
        # We process the whole batch at once for speed
        states = torch.FloatTensor(np.array([i[0] for i in minibatch])).to(self.device)
        actions = torch.LongTensor([i[1] for i in minibatch]).to(self.device)
        rewards = torch.FloatTensor([i[2] for i in minibatch]).to(self.device)
        next_states = torch.FloatTensor(np.array([i[3] for i in minibatch])).to(self.device)
        dones = torch.FloatTensor([i[4] for i in minibatch]).to(self.device)

        # 3. Calculate Target Q-Values (The "Truth")
        # Formula: Reward + (Gamma * Max_Q(next_state))
        # If done, target is just Reward.
        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # 4. Calculate Current Q-Values (The "Prediction")
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()

        # 5. Backpropagation (Learning)
        loss = self.criterion(current_q_values, target_q_values)
        
        self.optimizer.zero_grad() # Clear old gradients
        loss.backward()            # Calculate new gradients
        self.optimizer.step()      # Update weights

        # Return the loss value to track it in main.py
        return loss.item()
        
    def update_epsilon(self):
        # We call this ONLY at the end of a match
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        # Load the weights from the file into the Neural Network
        self.model.load_state_dict(torch.load(name, map_location=self.device))
        print(f"Brain loaded from {name}")