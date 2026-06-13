import pygame
import numpy as np
import random
from collections import deque

class HeroKillerEnv:
    def __init__(self, training_mode=True, render_mode=False, difficulty=0):
        self.training_mode = training_mode
        self.should_render = render_mode
        self.difficulty = difficulty # 0 = Sandbag, 1 = Aggressive
        
        # PHYSICAL CONSTANTS
        self.width = 50
        self.height = 50
        self.vel = 4
        self.screenW = 1000
        self.screenH = 600
        self.MAX_STEPS = 2000 # End game after 2000 frames (~30 seconds)
        
        # AI BRAIN CONSTANTS
        self.REACTION_DELAY = 15
        self.ERROR_RATE = 0.20
        self.observation_buffer = deque(maxlen=self.REACTION_DELAY)
        
        # RL SPACES
        self.action_space_n = 6 
        self.state_space_n = 10 

        # PYGAME SETUP
        pygame.init()
        if self.should_render:
            self.screen = pygame.display.set_mode((self.screenW, self.screenH))
            pygame.display.set_caption("Hero Killer: Training Dojo")
        else:
            self.screen = pygame.Surface((self.screenW, self.screenH))

    def reset(self):
        self.current_step = 0 # Reset timer
        
        # Positions (Start closer together to force interaction)
        self.x1 = 200 
        self.y1 = self.screenH - (self.height * 1.75)
        self.x2 = 800 
        self.y2 = self.screenH - (self.height * 1.75)
        
        self.health1 = 5
        self.health2 = 5
        
        # Physics State Variables (Reset all)
        self.vel1 = 0
        self.isJumping1 = False; self.jumpFrame1 = 24
        self.atk_active1 = False; self.atk_frame1 = 0
        self.atk_cooldown1 = False; self.atk_cd_frame1 = 0
        self.block_active1 = False; self.block_frame1 = 0
        self.block_cd_active1 = False; self.block_cd_frame1 = 0
        self.inv_frames1 = 0; self.prevKey1 = "RIGHT"

        self.vel2 = 0
        self.isJumping2 = False; self.jumpFrame2 = 24
        self.atk_active2 = False; self.atk_frame2 = 0
        self.atk_cooldown2 = False; self.atk_cd_frame2 = 0
        self.block_active2 = False; self.block_frame2 = 0
        self.block_cd_active2 = False; self.block_cd_frame2 = 0
        self.inv_frames2 = 0; self.prevKey2 = "LEFT"

        self.observation_buffer.clear()
        self.bot_timer = 0
        self.bot_state = "IDLE"

        return self.get_state()

    def get_state(self):
        # OPTIMIZED STATE VECTOR (Relative Coordinates)
        # Instead of giving absolute positions, we give the DISTANCE.
        # This makes it much easier for the AI to learn spacing.
        
        dx = (self.x1 - self.x2) / self.screenW # Distance X
        dy = (self.y1 - self.y2) / self.screenH # Distance Y
        
        state = np.array([
            dx,                                      # 1. Relative Distance X
            dy,                                      # 2. Relative Distance Y
            self.x2 / self.screenW,                  # 3. My X (To know if I'm in a corner)
            self.x1 / self.screenW,                  # 4. Enemy X
            self.health2 / 5.0,                      # 5. My HP
            self.health1 / 5.0,                      # 6. Enemy HP
            1.0 if self.atk_cooldown2 else 0.0,      # 7. My Atk CD
            1.0 if self.block_cd_active2 else 0.0,   # 8. My Blk CD
            1.0 if self.atk_cooldown1 else 0.0,      # 9. Enemy Atk CD
            1.0 if self.block_cd_active1 else 0.0    # 10. Enemy Blk CD
        ], dtype=np.float32)
        return state
    def get_state_p1(self):
        # dx was (x1 - x2). For P1, the relative distance is (x2 - x1)
        dx = (self.x2 - self.x1) / self.screenW 
        dy = (self.y2 - self.y1) / self.screenH 
        
        state = np.array([
            dx,                                      # 1. Distance to P2
            dy,                                      # 2. Relative Height
            self.x1 / self.screenW,                  # 3. My X (P1 now sees its own X)
            self.x2 / self.screenW,                  # 4. Enemy X
            self.health1 / 5.0,                      # 5. My HP (P1's health)
            self.health2 / 5.0,                      # 6. Enemy HP
            1.0 if self.atk_cooldown1 else 0.0,      # 7. My Atk CD
            1.0 if self.block_cd_active1 else 0.0,   # 8. My Blk CD
            1.0 if self.atk_cooldown2 else 0.0,      # 9. Enemy Atk CD
            1.0 if self.block_cd_active2 else 0.0    # 10. Enemy Blk CD
        ], dtype=np.float32)
        return state

    def step(self, action_index, opp_action=None):
        self.current_step += 1
        
        self._apply_action_p2(action_index)


        if opp_action is not None:
            # SELF-PLAY MODE: Use the action passed from the script
            self._apply_action_p1(opp_action)
        elif self.training_mode:
            # BOT MODE: Use Baby Algo
            self._move_hero_bot() 
        else:
            # HUMAN MODE
            self._handle_human_input() 
            
        self._update_physics()
        reward = self._resolve_combat()
        
        done = False
        
        if self.health1 <= 0:
            reward += 500 
            done = True
        elif self.health2 <= 0:
            reward -= 50
            done = True
            
        # TIMEOUT LOGIC (Prevents -2000 scores)
        if self.current_step >= self.MAX_STEPS:
            done = True
            # No huge penalty, just end the suffering
            
        # Living penalty (Encourages speed)
        reward -= 0.1
        
        dx = abs(self.x1 - self.x2)
        
        if dx > 100: 
            # Apply a tiny penalty every frame to force movement
            reward -= 0.1
        return self.get_state(), reward, done, {}
    def _apply_action_p1(self, action):
        if self.block_active1: return 

        if action == 1 and self.x1 < self.screenW - self.width:
            self.x1 += self.vel
            self.prevKey1 = "RIGHT"
        elif action == 2 and self.x1 > self.vel :
            self.x1 -= self.vel
            self.prevKey1 = "LEFT"
        
        if action == 3 and not self.isJumping1:
            self.isJumping1 = True
            
        if action == 4 and not self.atk_cooldown1:
            self.atk_active1 = True
            
        if action == 5 and not self.block_cd_active1:
            self.block_active1 = True

    # ---------------------------------------------------------
    # INTERNAL LOGIC
    # ---------------------------------------------------------

    def _apply_action_p2(self, action):
        if self.block_active2: return 

        if action == 1 and self.x2 > self.vel:
            self.x2 -= self.vel
            self.prevKey2 = "LEFT"
        elif action == 2 and self.x2 < self.screenW - self.width:
            self.x2 += self.vel
            self.prevKey2 = "RIGHT"
        
        if action == 3 and not self.isJumping2:
            self.isJumping2 = True
            
        if action == 4 and not self.atk_cooldown2:
            self.atk_active2 = True
            
        if action == 5 and not self.block_cd_active2:
            self.block_active2 = True

    def _move_hero_bot(self):
        """
        The 'Baby Algo' Logic.
        DIFFICULTY 0: Sandbag (Stands still, blocks if you spam, attacks only if touched)
        DIFFICULTY 1: Aggressive (Chases you)
        """
        current_reality = (self.x2, self.y2, self.atk_active2, self.block_active2)
        self.observation_buffer.append(current_reality)

        if len(self.observation_buffer) == self.REACTION_DELAY:
            p_x2, p_y2, p_atk2, p_block2 = self.observation_buffer[0]
        else:
            p_x2, p_y2, p_atk2, p_block2 = current_reality

        dx = p_x2 - self.x1 
        distance = abs(dx)

        if self.bot_timer > 0:
            self.bot_timer -= 1
        else:
            # --- SANDBAG LOGIC (Difficulty 0) ---
            if self.difficulty == 0:
                if distance > self.width * 4:
                    self.bot_state = "APPROACH" 
                else: 
                    self.bot_state = "IDLE"

            # --- SANDBAG LOGIC (Difficulty 1) ---
            elif self.difficulty == 1:
                threat_detected = p_atk2 and distance < 120
                if distance > self.width * 4:
                    self.bot_state = "APPROACH" 
                elif threat_detected:
                     self.bot_state = "DEFEND"
                elif distance < 60: # Self defense only
                     self.bot_state = "ATTACK"
                else:
                     self.bot_state = "IDLE"
            
            # --- AGGRESSIVE LOGIC (Difficulty 2) ---
            else:
                threat_detected = p_atk2 and distance < 150
                if threat_detected and random.random() > self.ERROR_RATE:
                    if not self.block_cd_active1: self.bot_state = "DEFEND"
                    else: self.bot_state = "PANIC_JUMP"
                elif distance > 1.2*self.width:
                    self.bot_state = "APPROACH"
                elif distance < 1.5 * self.width and not p_atk2:
                    if random.random() > self.ERROR_RATE: self.bot_state = "ATTACK"
                    else: self.bot_state = "HESITATE"
                else:
                    self.bot_state = "IDLE"
            
            self.bot_timer = random.randint(5, 15)

        # EXECUTION
        if self.bot_state == "DEFEND":
            if not self.block_active1 and not self.block_cd_active1: self.block_active1 = True     
        elif self.bot_state == "PANIC_JUMP":
             if not self.isJumping1: self.isJumping1 = True
        elif self.bot_state == "APPROACH":
            if dx > 0 and self.x1 < self.screenW - self.width:
                self.x1 += self.vel
                self.prevKey1 = "RIGHT"
            elif dx < 0 and self.x1 > self.vel:
                self.x1 -= self.vel
                self.prevKey1 = "LEFT"
        elif self.bot_state == "ATTACK":
            # Micro-spacing (Only in aggressive mode)
            if self.difficulty == 1 and distance > self.width + 10:
                 if dx > 0: self.x1 += self.vel
                 else: self.x1 -= self.vel
            if not self.atk_cooldown1 and not self.atk_active1:
                self.atk_active1 = True

    def _handle_human_input(self):
        keys = pygame.key.get_pressed()
        if not self.block_active1:
            if keys[pygame.K_LEFT] and self.x1 > self.vel:
                self.x1 -= self.vel
                self.prevKey1 = "LEFT"
            if keys[pygame.K_RIGHT] and self.x1 < self.screenW - self.width:
                self.x1 += self.vel
                self.prevKey1 = "RIGHT"
            if keys[pygame.K_UP] and not self.isJumping1:
                self.isJumping1 = True
            if keys[pygame.K_DOWN] and not self.atk_cooldown1:
                self.atk_active1 = True
            if keys[pygame.K_m] and not self.block_cd_active1:
                self.block_active1 = True

    def _update_physics(self):
        # BODY COLLISION
        body_p1 = (self.x1, self.y1, self.width, self.height)
        body_p2 = (self.x2, self.y2, self.width, self.height)
        
        if self._check_rect_collision(body_p1, body_p2):
            if self.x2 < self.x1:
                self.x2 -= 2 * self.vel
                self.x1 += 2 * self.vel
            else:
                self.x2 += 2 * self.vel
                self.x1 -= 2 * self.vel
            self.x1 = max(0, min(self.x1, self.screenW - self.width))
            self.x2 = max(0, min(self.x2, self.screenW - self.width))

        # P1 PHYSICS
        if self.isJumping1:
            if self.jumpFrame1 >= -24:
                neg = 1 if self.jumpFrame1 >= 0 else -1
                self.y1 -= (0.5 * (self.jumpFrame1 ** 2)) * 0.1 * neg
                self.jumpFrame1 -= 1
            else:
                self.isJumping1 = False; self.jumpFrame1 = 24
                
        if self.atk_active1:
            self.atk_frame1 += 1
            if self.atk_frame1 >= 10: self.atk_active1 = False; self.atk_frame1 = 0; self.atk_cooldown1 = True
        if self.atk_cooldown1:
            self.atk_cd_frame1 += 1
            if self.atk_cd_frame1 >= 30: self.atk_cooldown1 = False; self.atk_cd_frame1 = 0
        if self.block_active1:
            self.block_frame1 += 1
            if self.block_frame1 >= 35: self.block_active1 = False; self.block_frame1 = 0; self.block_cd_active1 = True
        if self.block_cd_active1:
            self.block_cd_frame1 += 1
            if self.block_cd_frame1 >= 50: self.block_cd_active1 = False; self.block_cd_frame1 = 0

        # P2 PHYSICS
        if self.isJumping2:
            if self.jumpFrame2 >= -24:
                neg = 1 if self.jumpFrame2 >= 0 else -1
                self.y2 -= (0.5 * (self.jumpFrame2 ** 2)) * 0.1 * neg
                self.jumpFrame2 -= 1
            else:
                self.isJumping2 = False; self.jumpFrame2 = 24
        
        if self.atk_active2:
            self.atk_frame2 += 1
            if self.atk_frame2 >= 10: self.atk_active2 = False; self.atk_frame2 = 0; self.atk_cooldown2 = True
        if self.atk_cooldown2:
            self.atk_cd_frame2 += 1
            if self.atk_cd_frame2 >= 30: self.atk_cooldown2 = False; self.atk_cd_frame2 = 0
        if self.block_active2:
            self.block_frame2 += 1
            if self.block_frame2 >= 35: self.block_active2 = False; self.block_frame2 = 0; self.block_cd_active2 = True
        if self.block_cd_active2:
            self.block_cd_frame2 += 1
            if self.block_cd_frame2 >= 50: self.block_cd_active2 = False; self.block_cd_frame2 = 0

        if self.inv_frames1 > 0: self.inv_frames1 -= 1
        if self.inv_frames2 > 0: self.inv_frames2 -= 1

    def _resolve_combat(self):
        reward = 0
        if self.prevKey1 == "LEFT": atk_rect_p1 = (self.x1 - self.width/2, self.y1 + self.height/4, self.width/2, self.height/2)
        else: atk_rect_p1 = (self.x1 + self.width, self.y1 + self.height/4, self.width/2, self.height/2)
        
        if self.prevKey2 == "LEFT": atk_rect_p2 = (self.x2 - self.width/2, self.y2 + self.height/4, self.width/2, self.height/2)
        else: atk_rect_p2 = (self.x2 + self.width, self.y2 + self.height/4, self.width/2, self.height/2)

        body_p1 = (self.x1, self.y1, self.width, self.height)
        body_p2 = (self.x2, self.y2, self.width, self.height)

        if self.atk_active2 and self.inv_frames1 == 0 and not self.block_active1:
            if self._check_rect_collision(atk_rect_p2, body_p1):
                self.health1 -= 1; self.inv_frames1 = 30; reward += 25 

        if self.atk_active1 and self.inv_frames2 == 0 and not self.block_active2:
            if self._check_rect_collision(atk_rect_p1, body_p2):
                self.health2 -= 1; self.inv_frames2 = 30; reward -= 15 

        return reward

    def _check_rect_collision(self, rect1, rect2):
        return (rect1[0] < rect2[0] + rect2[2] and rect1[0] + rect1[2] > rect2[0] and
                rect1[1] < rect2[1] + rect2[3] and rect1[1] + rect1[3] > rect2[1])

    def render(self):
        if not self.should_render: return
        self.screen.fill((20, 20, 20)) 
        
        c1 = (200, 200, 255) if self.block_active1 else ((100, 100, 0) if self.inv_frames1 > 0 else (250, 250, 0))
        pygame.draw.rect(self.screen, c1, (self.x1, self.y1, self.width, self.height))
        
        c2 = (200, 200, 255) if self.block_active2 else ((0, 100, 0) if self.inv_frames2 > 0 else (0, 250, 0))
        pygame.draw.rect(self.screen, c2, (self.x2, self.y2, self.width, self.height))

        if self.atk_active1:
            offset = -self.width/2 if self.prevKey1 == "LEFT" else self.width
            pygame.draw.rect(self.screen, (255,0,0), (self.x1+offset, self.y1+self.height//4, self.width//2, self.height//2))

        if self.atk_active2:
            offset = -self.width/2 if self.prevKey2 == "LEFT" else self.width
            pygame.draw.rect(self.screen, (255,0,0), (self.x2+offset, self.y2+self.height//4, self.width//2, self.height//2))
            
        pygame.display.update()