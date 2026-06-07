import cv2
import mediapipe as mp
import pygame
import sys
import random
import numpy as np
from pygame.locals import *

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, 
                       max_num_hands=2, 
                       min_detection_confidence=0.5, 
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def find_hands(img, draw=True):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            if draw:
                mp_draw.draw_landmarks(
                    img, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
    return img, results

def find_position(img, results, hand_no=0, draw=True):
    lm_list = []
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > hand_no:
        my_hand = results.multi_hand_landmarks[hand_no]
        for id, lm in enumerate(my_hand.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lm_list.append([id, cx, cy])
            if draw:
                cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
    return lm_list

# Define the tip landmarks for each finger
finger_tips = [4, 8, 12, 16, 20]

def get_finger_count(landmarks):
    if len(landmarks) == 0:
        return 0
    
    fingers_status = []
    
    # Detect thumb: compare x-coordinates for open/closed status
    if landmarks[finger_tips[0]][1] > landmarks[finger_tips[0] - 1][1]:
        fingers_status.append(1)
    else:
        fingers_status.append(0)
    
    # Detect the rest of the fingers (index, middle, ring, pinky)
    for finger in range(1, 5):
        if landmarks[finger_tips[finger]][2] < landmarks[finger_tips[finger] - 2][2]:
            fingers_status.append(1)
        else:
            fingers_status.append(0)
    
    return fingers_status.count(1)

class Dino:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 40
        self.height = 50
        self.vel_y = 0
        self.jumping = False
        self.gravity = 1
        self.jump_power = -18
        self.color = (50, 200, 50)
        
    def jump(self):
        if not self.jumping:
            self.jumping = True
            self.vel_y = self.jump_power
            
    def update(self):
        if self.jumping:
            self.vel_y += self.gravity
            self.y += self.vel_y
            
            # Ground collision
            if self.y >= 400 - self.height:
                self.y = 400 - self.height
                self.jumping = False
                self.vel_y = 0
                
    def draw(self, screen):
        # Draw dino body
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))
        # Draw eye
        pygame.draw.circle(screen, (255, 255, 255), (self.x + self.width - 10, self.y + 15), 5)
        pygame.draw.circle(screen, (0, 0, 0), (self.x + self.width - 8, self.y + 13), 2)
        # Draw leg when on ground
        if not self.jumping:
            pygame.draw.rect(screen, self.color, (self.x + 10, self.y + self.height, 8, 15))
            pygame.draw.rect(screen, self.color, (self.x + 25, self.y + self.height, 8, 15))
        
    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

class Obstacle:
    def __init__(self, x, obstacle_type="cactus"):
        self.x = x
        self.obstacle_type = obstacle_type
        self.width = 20
        self.height = 40
        self.color = (139, 69, 19)
        self.y = 400 - self.height
        
        # Different obstacle variations
        if obstacle_type == "small_cactus":
            self.width = 15
            self.height = 35
            self.y = 400 - self.height
        elif obstacle_type == "large_cactus":
            self.width = 25
            self.height = 50
            self.y = 400 - self.height
        elif obstacle_type == "bird":
            self.width = 30
            self.height = 20
            self.y = 350
            self.color = (100, 100, 100)
            
    def update(self, speed):
        self.x -= speed
        
    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))
        
        # Add details based on obstacle type
        if self.obstacle_type == "bird":
            # Draw wings
            pygame.draw.ellipse(screen, (80, 80, 80), (self.x - 10, self.y + 5, 15, 10))
            pygame.draw.ellipse(screen, (80, 80, 80), (self.x + self.width - 5, self.y + 5, 15, 10))
        elif "cactus" in self.obstacle_type:
            # Add cactus spikes
            pygame.draw.line(screen, (100, 50, 10), (self.x + self.width//2, self.y), 
                           (self.x + self.width//2, self.y - 8), 3)
            
    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

class Game:
    def __init__(self):
        # Initialize Pygame
        pygame.init()
        self.screen_width = 800
        self.screen_height = 500
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Finger Control Dino Game")
        self.clock = pygame.time.Clock()
        
        # Game variables
        self.dino = Dino(100, 400 - 50)
        self.obstacles = []
        self.score = 0
        self.high_score = 0
        self.game_speed = 5
        self.game_over = False
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Colors
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.gray = (100, 100, 100)
        self.sky_blue = (135, 206, 235)
        self.ground_color = (139, 69, 19)
        
        # Obstacle generation timer
        self.obstacle_timer = 0
        self.obstacle_delay = 90  # Frames between obstacles
        
        # Background
        self.ground_y = 400
        self.clouds = [(random.randint(0, 800), random.randint(50, 200)) for _ in range(3)]
        
        # Finger control variables
        self.last_jump_command = False
        self.jump_cooldown = 0
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Game Started! Raise your hand and open your fingers to make the dino jump!")
        print("Game controls: Open hand (fingers up) = Jump")
        
    def draw_background(self):
        # Sky
        self.screen.fill(self.sky_blue)
        
        # Ground
        pygame.draw.rect(self.screen, self.ground_color, (0, self.ground_y, self.screen_width, 100))
        pygame.draw.rect(self.screen, (100, 50, 10), (0, self.ground_y, self.screen_width, 5))
        
        # Ground details
        for i in range(0, self.screen_width, 30):
            pygame.draw.rect(self.screen, (100, 80, 20), (i, self.ground_y + 5, 15, 3))
            
        # Clouds
        for cloud in self.clouds:
            pygame.draw.ellipse(self.screen, (255, 255, 255), (cloud[0], cloud[1], 50, 30))
            pygame.draw.ellipse(self.screen, (255, 255, 255), (cloud[0] + 20, cloud[1] - 10, 40, 30))
            pygame.draw.ellipse(self.screen, (255, 255, 255), (cloud[0] - 20, cloud[1] - 10, 40, 30))
            
        # Move clouds slowly
        for i in range(len(self.clouds)):
            self.clouds[i] = (self.clouds[i][0] - 0.5, self.clouds[i][1])
            if self.clouds[i][0] < -100:
                self.clouds[i] = (self.screen_width + 100, random.randint(50, 200))
                
    def draw_ui(self):
        # Score
        score_text = self.font.render(f"Score: {self.score}", True, self.black)
        self.screen.blit(score_text, (10, 10))
        
        # High score
        if self.score > self.high_score:
            self.high_score = self.score
        high_score_text = self.small_font.render(f"High Score: {self.high_score}", True, self.black)
        self.screen.blit(high_score_text, (10, 50))
        
        # Game speed indicator
        speed_text = self.small_font.render(f"Speed: {self.game_speed:.1f}", True, self.black)
        self.screen.blit(speed_text, (10, 80))
        
        # Instructions
        inst_text = self.small_font.render("Raise your hand and open fingers to jump!", True, self.black)
        self.screen.blit(inst_text, (self.screen_width - 300, 10))
        
    def draw_game_over(self):
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        game_over_text = self.font.render("GAME OVER", True, self.white)
        score_text = self.font.render(f"Score: {self.score}", True, self.white)
        restart_text = self.small_font.render("Open your hand to restart the game!", True, self.white)
        
        text_rect = game_over_text.get_rect(center=(self.screen_width//2, self.screen_height//2 - 50))
        score_rect = score_text.get_rect(center=(self.screen_width//2, self.screen_height//2))
        restart_rect = restart_text.get_rect(center=(self.screen_width//2, self.screen_height//2 + 50))
        
        self.screen.blit(game_over_text, text_rect)
        self.screen.blit(score_text, score_rect)
        self.screen.blit(restart_text, restart_rect)
        
    def restart_game(self):
        self.dino = Dino(100, 400 - 50)
        self.obstacles = []
        self.score = 0
        self.game_speed = 5
        self.game_over = False
        self.obstacle_timer = 0
        
    def update_obstacles(self):
        # Generate new obstacles
        if not self.game_over:
            if self.obstacle_timer <= 0:
                # Random obstacle type
                obstacle_type = random.choice(["small_cactus", "cactus", "large_cactus", "bird"])
                if obstacle_type == "bird" and random.random() > 0.7:  # Birds less frequent
                    self.obstacles.append(Obstacle(self.screen_width, "bird"))
                else:
                    self.obstacles.append(Obstacle(self.screen_width, obstacle_type))
                self.obstacle_timer = random.randint(60, 120)
            else:
                self.obstacle_timer -= 1
                
        # Update existing obstacles
        for obstacle in self.obstacles[:]:
            obstacle.update(self.game_speed)
            if obstacle.x + obstacle.width < 0:
                self.obstacles.remove(obstacle)
                
    def check_collisions(self):
        dino_rect = self.dino.get_rect()
        for obstacle in self.obstacles:
            if dino_rect.colliderect(obstacle.get_rect()):
                self.game_over = True
                return True
        return False
        
    def run(self):
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False
                        
            # Process camera feed for finger control
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                frame, results = find_hands(frame)
                
                # Get finger count
                landmarks = find_position(frame, results, draw=False)
                fingers_up = get_finger_count(landmarks)
                
                # Control dino with fingers
                if not self.game_over:
                    # Jump when fingers are up (any fingers raised)
                    if fingers_up > 0 and not self.last_jump_command:
                        self.dino.jump()
                        self.last_jump_command = True
                    elif fingers_up == 0:
                        self.last_jump_command = False
                else:
                    # Restart game when hand is raised
                    if fingers_up > 0:
                        self.restart_game()
                
                # Draw finger count on camera feed
                if fingers_up > 0:
                    cv2.putText(frame, f"JUMP! ({fingers_up} fingers)", (20, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No hand detected", (20, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Show camera feed in a smaller window
                camera_display = cv2.resize(frame, (320, 240))
                cv2.imshow("Hand Control", camera_display)
                
                # Check for quit in camera window
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    running = False
                    
            # Game logic
            if not self.game_over:
                self.dino.update()
                self.update_obstacles()
                self.check_collisions()
                self.score += 1
                
                # Increase difficulty over time
                if self.score % 500 == 0 and self.game_speed < 15:
                    self.game_speed += 0.5
                    self.obstacle_delay = max(40, self.obstacle_delay - 5)
                    
            # Drawing
            self.draw_background()
            self.dino.draw(self.screen)
            for obstacle in self.obstacles:
                obstacle.draw(self.screen)
            self.draw_ui()
            
            if self.game_over:
                self.draw_game_over()
                
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS
            
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = Game()
    game.run()