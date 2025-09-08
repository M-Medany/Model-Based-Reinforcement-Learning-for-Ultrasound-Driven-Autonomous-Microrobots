import pygame
import sys
import numpy as np

# Initialize Pygame
pygame.init()

# Define screen dimensions and colors
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 800
BACKGROUND_COLOR = (255, 255, 255)
CIRCLE_COLOR = (0, 0, 255)

# Load your binary image for the environment
ENVIRONMENT_IMAGE = "binary_images/closing1.png"

# Initialize the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Simple 2D Game")

font = pygame.font.Font(None, 36)
score = 0

# Load the environment image
environment = pygame.image.load(ENVIRONMENT_IMAGE)
environment = pygame.transform.scale(environment, (SCREEN_WIDTH, SCREEN_HEIGHT))

# Set up the player circle
player_radius = 20
player_x, player_y = 140, 660
player_speed = 0.2

#sample random point and check if it is white or black
def find_target_point():
    x = np.random.randint(0, SCREEN_WIDTH)
    y = np.random.randint(0, SCREEN_HEIGHT)
    pixel_color = environment.get_at((int(x), int(y)))
    if pixel_color == (255, 255, 255, 255):
        return x, y
    else:
        return find_target_point()

def check_collision(player_x, player_y, radius):
    for x in range(int(player_x - radius), int(player_x + radius)):
        for y in range(int(player_y - radius), int(player_y + radius)):
            pixel_color = environment.get_at((x, y))
            if pixel_color == (0, 0, 0, 255):  # Check for black (obstacle) color
                return True
    return False

def check_goal_reached(player_x, player_y, radius):
    for x in range(int(player_x - radius), int(player_x + radius)):
        for y in range(int(player_y - radius), int(player_y + radius)):
            pixel_color = screen.get_at((x, y))
            if pixel_color == (255, 0, 0, 255):  # Check for red (target) color
                return True
    return False


new_target_point = find_target_point()


# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Handle user input
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        player_y -= player_speed
    if keys[pygame.K_DOWN]:
        player_y += player_speed
    if keys[pygame.K_LEFT]:
        player_x -= player_speed
    if keys[pygame.K_RIGHT]:
        player_x += player_speed

    # Check for collisions with the environment
    #player_rect = pygame.Rect(player_x - player_radius, player_y - player_radius, 2 * player_radius, 2 * player_radius)
    #pixel_color = environment.get_at((int(player_x), int(player_y)))
    score_text = font.render(f"Score: {score}", True, (255, 255, 255))

    # if pixel_color == (0, 0, 0, 255):  # Check for black (obstacle) color
    #     print("Game Over!")
    #     running = False
    # elif pixel_color == (255, 0, 0, 255):  # Check for red (target) color
    #     score += 1
    #     new_target_point = find_target_point()

    if check_collision(player_x, player_y, player_radius):
        print("Game Over!")
        running = False
    if check_goal_reached(player_x, player_y, player_radius):
        score += 1
        new_target_point = find_target_point()



    # Clear the screen
    screen.fill(BACKGROUND_COLOR)

    # Draw the environment and player
    screen.blit(environment, (0, 0))
    screen.blit(score_text, (10, 10))
    pygame.draw.circle(screen, CIRCLE_COLOR, (player_x, player_y), player_radius)
    pygame.draw.circle(screen, (255, 0, 0), new_target_point, player_radius)

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()