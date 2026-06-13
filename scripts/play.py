import pygame
import sys
import torch
import numpy as np
import os
from src.HeroKillerEnv import HeroKillerEnv
from src.DQNAgent import DQNAgent 

def main():
    MODEL_PATH = "models/herokiller_bc_cloned.pth" 
    FPS = 120
    
    print("\n" + "="*50)
    print("        HERO KILLER: MAN VS. MACHINE        ")
    print("="*50)
    print("CONTROLS:")
    print(" - ARROWS: Left / Right to Move")
    print(" - UP ARROW: Jump")
    print(" - DOWN ARROW: Attack")
    print(" - 'M' KEY: Block")
    print("="*50 + "\n")

    env = HeroKillerEnv(training_mode=False, render_mode=True, difficulty=1)
    # set training_mode = true for bot to learn how to play against you
    
    state_size = 10
    action_size = 6
    agent = DQNAgent(state_size, action_size, training_mode=False)

    if os.path.exists(MODEL_PATH):
        agent.load(MODEL_PATH)
        print(f"Boss AI Loaded Successfully from: {MODEL_PATH}")
    else:
        print(f"ERROR: Could not find {MODEL_PATH}.")
        print("Did you finish training? Make sure the file is in this folder.")
        sys.exit()

    agent.epsilon = 0.0

    clock = pygame.time.Clock()
    
    # 4. INFINITE MATCH LOOP
    match_count = 1
    p1_wins = 0
    p2_wins = 0

    while True:
        state = env.reset()
        done = False
        print(f"\n--- MATCH {match_count} START ---")
        print(f"Scoreboard -> HUMAN: {p1_wins} | AI: {p2_wins}")

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("\nClosing Arena...")
                    pygame.quit()
                    sys.exit()
            

            ai_action = agent.act(state)
            

            next_state, reward, done, _ = env.step(ai_action, opp_action=None)
            
            state = next_state
            
            env.render()
            clock.tick(FPS)
            
        if env.health1 <= 0:
            print("DEFEAT! The AI claims another victim.")
            p2_wins += 1
        elif env.health2 <= 0:
            print("VICTORY! You beat the machine.")
            p1_wins += 1
        else:
            print("DRAW! Time limit reached.")
            
        match_count += 1
        pygame.time.delay(1000)

if __name__ == "__main__":
    main()