import numpy as np
import pygame # Import Pygame
from homeostatic_crafter.env import Env

# --- Pygame Visualization Settings ---
SCREEN_WIDTH = 700  # Increased width to accommodate text
SCREEN_HEIGHT = 600
STATS_AREA_WIDTH = 200 # Width for the stats panel
GAME_AREA_WIDTH = SCREEN_WIDTH - STATS_AREA_WIDTH

TEXT_COLOR = (200, 200, 200)
BAR_COLOR_FULL = (0, 200, 0)
BAR_COLOR_EMPTY = (200, 0, 0)
BAR_HEIGHT = 20
BAR_WIDTH = 150
TEXT_OFFSET_X = 10
TEXT_OFFSET_Y = 10
LINE_SPACING = 25
# --- End Pygame Visualization Settings ---

def main():
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Homeostatic Crafter Test")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24) # Default system font, size 24

    env = Env() # No render_mode needed here, env.render() will give np array

    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    num_episodes = 3
    max_steps_per_episode = 200
    running = True

    homeostatic_var_names = ['Health', 'Food', 'Drink', 'Energy']

    for episode in range(num_episodes):
        if not running: break
        print(f"\n--- Episode {episode + 1} ---")
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        
        for step in range(max_steps_per_episode):
            if not running: break

            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
            if not running: break

            # Take a random action
            action = env.action_space.sample()
            # print(f"Step {step + 1}: Action: {env.action_names[action]} ({action})")

            # Perform the action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # --- Rendering --- 
            # Get environment image (H, W, C)
            env_img_hwc = env.render(size=(GAME_AREA_WIDTH, SCREEN_HEIGHT)) 
            # Pygame expects (W, H, C) for surfarray.make_surface
            env_img_whc = env_img_hwc.transpose((1, 0, 2))
            game_surface = pygame.surfarray.make_surface(env_img_whc)
            screen.blit(game_surface, (0, 0))

            # --- Homeostatic Variables Visualization ---
            stats_surface = pygame.Surface((STATS_AREA_WIDTH, SCREEN_HEIGHT))
            stats_surface.fill((30, 30, 30)) # Dark background for stats panel
            
            measurements = obs['measurements'] # Normalized [0, 1]
            current_y = TEXT_OFFSET_Y

            title_text = font.render("Homeostatic Vars:", True, TEXT_COLOR)
            stats_surface.blit(title_text, (TEXT_OFFSET_X, current_y))
            current_y += LINE_SPACING * 1.5

            for i, name in enumerate(homeostatic_var_names):
                value = measurements[i]
                
                # Variable Name
                text_surface = font.render(f"{name}: {value:.2f}", True, TEXT_COLOR)
                stats_surface.blit(text_surface, (TEXT_OFFSET_X, current_y))
                current_y += LINE_SPACING

                # Bar Background (empty part)
                pygame.draw.rect(stats_surface, BAR_COLOR_EMPTY, (TEXT_OFFSET_X, current_y, BAR_WIDTH, BAR_HEIGHT))
                # Bar Foreground (filled part)
                fill_width = int(BAR_WIDTH * value)
                pygame.draw.rect(stats_surface, BAR_COLOR_FULL, (TEXT_OFFSET_X, current_y, fill_width, BAR_HEIGHT))
                current_y += BAR_HEIGHT + LINE_SPACING // 2 # Add some space after the bar

            screen.blit(stats_surface, (GAME_AREA_WIDTH, 0))
            # --- End Homeostatic Visualization ---

            pygame.display.flip()
            clock.tick(30) # Target FPS
            # --- End Rendering ---
            
            # Print results to console (optional)
            # print(f"  Reward: {reward}")
            # print(f"  Done: {done}")
            
            total_reward += reward

            if done:
                print(f"Episode finished after {step + 1} steps.")
                break
        
        print(f"Total reward for episode {episode + 1}: {total_reward}")

    pygame.quit()
    print("\nTesting complete.")

if __name__ == '__main__':
    main() 
