# this file is made for training on lightning.ai 

import pygame
import sys
import torch
import numpy as np
from DQNTrainingLogic.src.HeroKillerEnv import HeroKillerEnv
from DQNTrainingLogic.src.DQNAgent import DQNAgent 
import random
import matplotlib.pyplot as plt
import os
os.environ["SDL_VIDEODRIVER"] = "dummy" # prevents pygame from crashing on headless cloud servers

def main():

    DIFFICULTY = 1
    
    selfplayProbability = 0.5
    RENDER = False

    if RENDER:
        TRAINING = False
    else:
        TRAINING = True

    LOAD_MODEL = False 
    MODEL_PATH = "dMIXED_s7_herokiller_dqn_5000.pth" 
    
    EPISODES = 1200

    # =================================================================
    
    print(f"--- HERO KILLER: TRAINING STARTED [Difficulty: {DIFFICULTY}] ---")
    
    env = HeroKillerEnv(training_mode=TRAINING, render_mode=RENDER, difficulty=DIFFICULTY)
    
    state_size = 10
    action_size = 6
    agent = DQNAgent(state_size, action_size, training_mode=TRAINING)
    
    scores_per_difficulty = {0: [], 1: [], 2: [], "SELF-PLAY": []}
    epsilons_history = []
    losses_history = []

    opponent_agent = DQNAgent(state_size, action_size, training_mode=False) 
    opponent_agent.epsilon = 0.05 
    
    if LOAD_MODEL:
        try:
            agent.load(MODEL_PATH)
            print(f"Loaded Brain: {MODEL_PATH}")
            opponent_agent.load(MODEL_PATH)

            if TRAINING:
                agent.epsilon = 0.6
            else:
                agent.epsilon = 0
            print("Adaptation Phase")
        except FileNotFoundError:
            print(f"Could not find {MODEL_PATH}. Starting from scratch.")
    else:
        print("Starting Training from Scratch (Epsilon 1.0)")

    # fps
    clock = pygame.time.Clock()
    if RENDER:
        FPS = 120
    else:
        FPS = 300

    for e in range(EPISODES):

        is_self_play = random.random() < selfplayProbability

        if not is_self_play:
            # mma
            dice = random.random()
            if dice < 0.60:
                current_difficulty = 1
            elif dice < 0.80:
                current_difficulty = 0
            else:
                current_difficulty = 2

            env.difficulty = current_difficulty
        else:
            current_difficulty = "SELF-PLAY"
        
        state = env.reset()
        total_reward = 0
        done = False
        episode_losses = []
        
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            action = agent.act(state)
            
            # OPPONENT DECISION if selfplay
            if is_self_play:
                opp_state = env.get_state_p1() 
                opp_action = opponent_agent.act(opp_state)
            else:
                opp_action = None
            
            next_state, reward, done, _ = env.step(action, opp_action=opp_action)
            
            agent.remember(state, action, reward, next_state, done)
            
            # Capture the loss returned by replay()
            loss = agent.replay()
            if loss is not None:
                episode_losses.append(loss)
            
            state = next_state
            total_reward += reward
            
            # for headed streaming
            if env.should_render:
                env.render()
                clock.tick(FPS)
        

        agent.update_epsilon()

        avg_loss = np.mean(episode_losses) if episode_losses else 0

        # metric logging
        scores_per_difficulty[current_difficulty].append(total_reward)
        epsilons_history.append(agent.epsilon)
        losses_history.append(avg_loss)

        print(f"Episode: {e+1}/{EPISODES} | Score: {total_reward:.2f} | Loss: {avg_loss:.4f} | Epsilon: {agent.epsilon:.2f} | Difficulty: {current_difficulty}")

        if (e + 1) % 100 == 0:
            print("Updating Self-Play Opponent to latest model weights...")
            opponent_agent.model.load_state_dict(agent.model.state_dict())

        if (e + 1) % 200 == 0: # less logging than on local
            save_name = f"herokiller_dqn_{e+1}.pth"
            torch.save(agent.model.state_dict(), save_name)
            print(f"Model Saved: {save_name}")
    
    # --- VISUALIZATION & LOGGING ---
    print("\nGenerating Training Graphs...")

    # Figure 1: Scores over time
    plt.figure(figsize=(10, 5))
    for diff, scores in scores_per_difficulty.items():
        if scores:
            plt.plot(scores, label=f'Difficulty {diff}', alpha=0.6)
    plt.xlabel('Episodes')
    plt.ylabel('Scores')
    plt.title('Agent Performance: Scores per Difficulty')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_scores.png')
    plt.show()

    # Figure 2: Loss and Epsilon Decay over time
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:red'
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Loss (Cost)', color=color)
    ax1.plot(losses_history, color=color, alpha=0.7, label='Average Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Epsilon (Exploration Rate)', color=color)  
    ax2.plot(epsilons_history, color=color, linestyle='dashed', label='Epsilon Decay')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  
    plt.title('Training Metrics: Cost Function & Epsilon Decay')
    plt.grid(True, alpha=0.3)
    plt.savefig('training_metrics.png')
    plt.show()
    
    # Save scores and metrics to file
    with open("training_log.txt", "w") as f:
        f.write("--- Final Training Metrics ---\n\n")
        for diff, scores in scores_per_difficulty.items():
            if scores:
                f.write(f"Average Score at Difficulty {diff}: {np.mean(scores):.2f}\n")
        f.write(f"\nFinal Epsilon: {epsilons_history[-1]:.4f}\n")
        f.write(f"Final Average Loss (Last 50 eps): {np.mean(losses_history[-50:]):.4f}\n")
            
    print("Training Finished. Metrics saved to disk.")
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()