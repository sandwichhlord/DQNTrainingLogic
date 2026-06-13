import pygame
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from src.HeroKillerEnv import HeroKillerEnv
from src.DQNAgent import DQNAgent 
import random
import matplotlib.pyplot as plt
from collections import deque
import time

# =================================================================
# EXPERT BOT LOGIC
# =================================================================
def get_expert_action_p2(env):
    """
    Reads the environment state and outputs discrete actions (0-5) 
    for Player 2 by prioritizing combat states correctly.
    """
    distance = abs(env.x1 - env.x2)
    p_atk1 = env.atk_active1 
    
    threat_detected = p_atk1 and distance < 150
    
    # defend
    if threat_detected:
        if not env.block_cd_active2: 
            return 5  # block
        else: 
            return 3  # panic jump

    # attack
    elif distance <= 1.4 * env.width and not p_atk1:
        return 4  # attack

    # approach
    elif distance > 1.1 * env.width:
        if env.x2 > env.x1: 
            return 1  # left
        else: 
            return 2  # right
            
    else:
        return 0  # idle

# =================================================================
# HELPER FUNCTION FOR GRAPHING
# =================================================================
def smooth_curve(points, factor=0.9):
    """
    Applies an exponential moving average to a list of points
    so the resulting graph is a clean, readable curve instead of noise.
    """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

# =================================================================
# MAIN TRAINING LOOP
# =================================================================
def main():
    RENDER = True
    TRAINING = True 
    EPISODES = 2000
    MODEL_PATH = "herokiller_bc_cloned.pth" 
    BATCH_SIZE = 64

    print("\n" + "="*50)
    print("      HERO KILLER: BEHAVIORAL CLONING TRAINING      ")
    print("="*50 + "\n")
    
    env = HeroKillerEnv(training_mode=TRAINING, render_mode=RENDER, difficulty=1)
    
    state_size = 10
    action_size = 6
    
    agent = DQNAgent(state_size, action_size, training_mode=TRAINING)
    
    opponent_agent = DQNAgent(state_size, action_size, training_mode=False)
    opponent_agent.epsilon = 0.05 # tiny exploration floor for ghosts
    
    # BEHAVIORAL CLONING SETUP
    # 0: Idle, 1: Left, 2: Right, 3: Jump, 4: Attack, 5: Block
    class_weights = torch.FloatTensor([1.0, 5.0, 5.0, 2.0, 1.0, 2.0]) 
    bc_criterion = nn.CrossEntropyLoss(weight=class_weights)
    bc_optimizer = optim.Adam(agent.model.parameters(), lr=0.001)
    bc_memory = deque(maxlen=10000)
    
    # 3. GHOST POOL TRACKING
    historical_checkpoints = []
    
    scores_per_difficulty = {0: [], 1: [], 2: [], "GHOST": []}
    cloning_loss_history = []

    clock = pygame.time.Clock()
    FPS = 1200 if RENDER else 1000
    start_time = time.time()

    for e in range(EPISODES):
        
        # --- MATCHMAKING ROUTINE ---
        # 50% chance to fight a Ghost (if any exist), else fight a standard bot
        is_ghost_match = random.random() < 0.50 and len(historical_checkpoints) > 0
        
        if is_ghost_match:
            current_difficulty = "GHOST"
            ghost_model = random.choice(historical_checkpoints)
            opponent_agent.load(ghost_model)
        else:
            dice = random.random()
            if dice < 0.33:
                current_difficulty = 0
            elif dice < 0.66:
                current_difficulty = 1
            else:
                current_difficulty = 2
            env.difficulty = current_difficulty
        
        state = env.reset()

        #screen flipping so agent learns to fight from both sides
        if random.random() < 0.50:
            env.x1 = 800  # opponent on the Right
            env.x2 = 200  # AI on the Left
            
            env.prevKey1 = "LEFT"
            env.prevKey2 = "RIGHT"
            
            state = env.get_state()
            
        total_reward = 0
        done = False
        episode_loss = 0
        steps = 0
        
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            expert_action = get_expert_action_p2(env)
            
            if is_ghost_match:
                opp_state = env.get_state_p1()
                raw_opp_action = opponent_agent.act(opp_state)
                
                if raw_opp_action == 1:
                    opp_action = 2
                elif raw_opp_action == 2:
                    opp_action = 1
                else:
                    opp_action = raw_opp_action
            else:
                opp_action = None
            
            next_state, reward, done, _ = env.step(expert_action, opp_action=opp_action)
            
            if TRAINING:
                bc_memory.append((state, expert_action))
                
                if len(bc_memory) >= BATCH_SIZE:
                    minibatch = random.sample(bc_memory, BATCH_SIZE)
                    
                    states = torch.FloatTensor(np.array([m[0] for m in minibatch]))
                    expert_actions = torch.LongTensor([m[1] for m in minibatch])
                    
                    # Forward pass
                    predicted_q_values = agent.model(states)
                    
                    # Calculate error
                    loss = bc_criterion(predicted_q_values, expert_actions)
                    
                    # Backpropagate
                    bc_optimizer.zero_grad()
                    loss.backward()
                    bc_optimizer.step()
                    
                    episode_loss += loss.item()
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if env.should_render:
                env.render()
                clock.tick(FPS)
        
        avg_loss = episode_loss / steps if steps > 0 else 0
        cloning_loss_history.append(avg_loss)
        scores_per_difficulty[current_difficulty].append(total_reward)
        
        # Terminal Metrics Display
        print(f"[EPISODE {e+1:04d}/{EPISODES}] "
              f"Opponent: {str(current_difficulty):<5} | "
              f"Steps: {steps:03d} | "
              f"Expert Score: {total_reward:+.2f} | "
              f"Loss: {avg_loss:.5f}")

        if (e + 1) % 50 == 0:
            save_name = f"herokiller_bc_ghost_{e+1}.pth"
            torch.save(agent.model.state_dict(), save_name)
            
            # Update the main model path to the latest weights
            torch.save(agent.model.state_dict(), MODEL_PATH)
            
            if save_name not in historical_checkpoints:
                historical_checkpoints.append(save_name)
            
            print(f"--> Checkpoint Saved & Ghost Added: {save_name}")
    
    # --- METRICS & GRAPHS ---
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60.0
    print(f"\nTraining Finished. Total Time: {elapsed_time:.2f} minutes.")
    print("Generating Metrics...")
    
    plt.figure(figsize=(15, 6))
    
    # Graph 1: Expert Performance (Smoothed)
    plt.subplot(1, 2, 1)
    for diff, scores in scores_per_difficulty.items():
        if len(scores) > 0:
            smoothed_scores = smooth_curve(scores, factor=0.85)
            plt.plot(smoothed_scores, label=f'Vs {diff}')
    plt.xlabel('Episodes')
    plt.ylabel('Expert Score (Moving Average)')
    plt.title('Expert Bot Performance')
    plt.legend()
    
    # Graph 2: Cloning Loss (Smoothed)
    plt.subplot(1, 2, 2)
    smoothed_loss = smooth_curve(cloning_loss_history, factor=0.95)
    plt.plot(cloning_loss_history, color='pink', alpha=0.3, label='Raw Loss') 
    plt.plot(smoothed_loss, color='red', linewidth=2, label='Smoothed Loss')
    plt.xlabel('Episodes')
    plt.ylabel('CrossEntropy Loss')
    plt.title('Neural Network Cloning Error')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Clean up ghost files (optional, keeps directory clean)
    for ghost in historical_checkpoints:
        if os.path.exists(ghost):
            os.remove(ghost)
            
    # Save scores to file
    with open("training_scores.txt", "w") as f:
        for diff, scores in scores_per_difficulty.items():
            f.write(f"Opponent {diff} Scores:\n")
            for score in scores:
                f.write(f"{score}\n")
            f.write("\n")
            
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()