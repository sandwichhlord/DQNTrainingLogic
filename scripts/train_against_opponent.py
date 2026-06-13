import pygame
import sys
import torch
import numpy as np
from src.HeroKillerEnv import HeroKillerEnv
from src.DQNAgent import DQNAgent 
import random
import matplotlib.pyplot as plt

# 4k episodes trained on mma + self
def main():

    DIFFICULTY = 1
    
    # False = Fast Training (Headless), True = Watch the AI (Windowed)
    selfplayProbability = 0.5
    RENDER = True

    if RENDER:
        TRAINING = False
    else:
        TRAINING = True
    # TRAINING = True

    LOAD_MODEL = True
    MODEL_PATH = "models/dMIXED_s7_herokiller_dqn_5000.pth" 
    
    EPISODES = 5000

    # =================================================================
    
    print(f"--- HERO KILLER: TRAINING STARTED [Difficulty: {DIFFICULTY}] ---")
    
    env = HeroKillerEnv(training_mode=TRAINING, render_mode=RENDER, difficulty=DIFFICULTY)
    
    state_size = 10
    action_size = 6
    agent = DQNAgent(state_size, action_size, training_mode=TRAINING)
    scores_per_difficulty = {0: [], 1: [], 2: [], "SELF-PLAY": []}

    opponent_agent = DQNAgent(state_size, action_size, training_mode=False) 
    opponent_agent.epsilon = 0.05 # opponent stays mostly consistent
    
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
        FPS = 1000
    else:
        FPS = 300
    # FPS = FPS

    for e in range(EPISODES):

        is_self_play = random.random() < selfplayProbability

        if not is_self_play:
            # mma
            dice = random.random()
            if dice < 0.60:
                current_difficulty = 0
            elif dice < 0.80:
                current_difficulty = 1
            else:
                current_difficulty = 2

            env.difficulty = current_difficulty
        else:
            current_difficulty = "SELF-PLAY"
        
        # rest of the loop

        state = env.reset()
        total_reward = 0
        done = False
        
        
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            # AI DECISION
            action = agent.act(state)
            # OPPONENT DECISION if selfplay
            if is_self_play:
                # need to flip obs for symmetry
                opp_state = env.get_state_p1() 
                opp_action = opponent_agent.act(opp_state)
            else:
                opp_action = None
            
            # EXECUTE ACTION
            next_state, reward, done, _ = env.step(action, opp_action=opp_action)
            
            # MEMORIZE & LEARN
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            state = next_state
            total_reward += reward
            
            # for headed streaming
            if env.should_render:
                env.render()
                clock.tick(FPS)
        
        agent.update_epsilon()


        
        print(f"Episode: {e+1}/{EPISODES} | Score: {total_reward:.2f} | Epsilon: {agent.epsilon:.2f} | Difficulty: {current_difficulty}")
        scores_per_difficulty[current_difficulty].append(total_reward)

        if (e + 1) % 100 == 0:
            print("Updating Self-Play Opponent to latest model weights...")
            opponent_agent.model.load_state_dict(agent.model.state_dict())

        # SAVE CHECKPOINT every 50 episodes
        if (e + 1) % 50 == 0:
            save_name = f"herokiller_dqn_{e+1}.pth"
            torch.save(agent.model.state_dict(), save_name)
            print(f"Model Saved: {save_name}")
    
    # print average scores per difficulty
    for diff, scores in scores_per_difficulty.items():
        if scores:
            avg_score = np.mean(scores)
            print(f"Average Score at Difficulty {diff}: {avg_score:.2f}")
        else:
            print(f"No episodes played at Difficulty {diff}.")
    # make graph of scores per difficulty over time
    for diff, scores in scores_per_difficulty.items():
        plt.plot(scores, label=f'Difficulty {diff}')
    plt.xlabel('Episodes')
    plt.ylabel('Scores')
    plt.title('Scores per Difficulty over Episodes')
    plt.legend()
    plt.show()
    
    #save scores to file
    with open("training_scores.txt", "w") as f:
        for diff, scores in scores_per_difficulty.items():
            f.write(f"Difficulty {diff} Scores:\n")
            for score in scores:
                f.write(f"{score}\n")
            f.write("\n")
            
    # EXIT
    print("Training Finished.")
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()