'''
Preparing Gym Environment
'''
import gym
from snake_learning import loop

env_dict = gym.envs.registration.registry.env_specs.copy()


for env in env_dict:
    if 'snake-plural-v0' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]
    if 'snake-v0' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]
 

import Gym_Snake_master.gym_snake
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd 
import random

game_on = True
testing = False

if game_on:
    from pyautogui import press

tf.executing_eagerly()

# Configuration paramaters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 100000000000

num_actions = 4
dir_to_key = {0:'w', 1:'d',2:'s',3:'a'}

def create_q_model():
    # Network defined by the Deepmind paper, modified heavily
    inputs = layers.Input(shape=(2, 5, 3,))

    layer1 = layers.Flatten()(inputs)
    layer2 = layers.Dense(120, activation="relu")(layer1)
    layer3 = layers.Dense(60, activation="relu")(layer2)
    layer4 = layers.Dense(60, activation="relu")(layer3)
    layer5 = layers.Dense(30, activation="relu")(layer4)
    action = layers.Dense(num_actions, activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=action)

def create_qRL_model():
    # Network defined by the Deepmind paper, modified heavily
    inputs = layers.Input(shape=(2, 5, 3,))

    layer1 = layers.Flatten()(inputs)
    layer2 = layers.Dense(30, activation="relu")(layer1)
    layer3 = layers.Dense(60, activation="relu")(layer2)
    layer4 = layers.Dense(60, activation="relu")(layer3)
    layer5 = layers.Dense(30, activation="relu")(layer4)
    action = layers.Dense(num_actions, activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=action)

def avoid_collision(state,snake,action):
    
    saved_a = action
    
    p_action = list(range(4))
                   
    if action == 3:
        action = 1
    elif action == 0:
        action = 3
    elif action == 1:
        action = 0
   
    while state[snake][action][2] == 1:
        p_action.remove(action)
        if(len(p_action) == 0):
                return saved_a
        action = np.random.choice(p_action)
        #print("Preventing death ", snake, "with action ", action)
   
    if action == 0:
        action = 1
    elif action == 1:
        action = 3
    elif action == 3:
        action = 0
    
    
    return action


# The first model makes the predictions for Q-values which are used to
# make a action.
models = [create_qRL_model(),create_qRL_model()]
# Build a target model for the prediction of future rewards.
# The weights of a target model get updated every 10000 steps thus when the
# loss between the Q-values is calculated the target Q-value is stable.
model_targets = [create_qRL_model(),create_qRL_model()]


# Construct Environment
env = gym.make('snake-plural-v0')
env.grid_size = [20,20]
env.unit_size = 10
env.unit_gap = 0
env.snake_size = 5
env.n_snakes = 2
env.n_foods = 1
observation = env.reset() # Constructs an instance of the game

# Controller
game_controller = env.controller

# Grid
grid_object = game_controller.grid
grid_pixels = grid_object.grid

# Snake(s)
snakes_array = game_controller.snakes
snake_object1 = snakes_array[0]

# Training

# In the Deepmind paper they use RMSProp however then Adam optimizer
# improves training time
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = [[],[]]
running_reward = [[],[]]
episode_count = 0
frame_count = 0
# Number of frames to take random action and observe output
epsilon_random_frames = 50000
# Number of frames for exploration
epsilon_greedy_frames = 1000000.0
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_length = 10000
# Train the model after 4 actions
update_after_actions = 4
# How often to update the target network
update_target_network = 500
# Using huber loss for stability
loss_function = keras.losses.Huber()


epsilon_random_frames = 0
epsilon_greedy_frames = 1
epsilon = 0.1
epsilon_min = 0.1
models[1] = keras.models.load_model("snakeRL_5040_FINAL.keras")
models[0] = keras.models.load_model("snakeRL_5040_FINAL.keras")
model_targets[0].set_weights(models[0].get_weights())
model_targets[1].set_weights(models[1].get_weights())
max_memory_length = 100

state = np.array(env.reset())
game_controller = env.controller
episode_reward = [0,0]


"""
Running the game
"""

import pygame
import random
pygame.init()

bg = (150, 150, 150)
head = (255, 150, 20)
body = (0, 200, 255)
food = (0, 200, 255)
score = (255, 150, 20)
red = (213, 50, 80)
outer = (100, 100, 200)

display_width = 600
display_hwidth = 300
display_height = 600
display_hheight = 200
display = pygame.display.set_mode((display_width, display_height))
clock = pygame.time.Clock()
font_style = pygame.font.SysFont("banschrift", 25)
score_font = pygame.font.SysFont("freesans", 35)
snake_block = 30
snake_speed = 15
hum_input = 0   # human input to be read by model


player_one_keys = [pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d]   # AI
player_two_keys = [pygame.K_UP, pygame.K_LEFT, pygame.K_DOWN, pygame.K_RIGHT]  # Human


class Snake:
    def __init__(self, ai, keys):
        self.snake_list = []
        self.is_ai = ai
        self.length = 1
        self.key_list = keys    # w/up, a/left, s/down, d/right


def update_move(player, event, is_hum = False):
    # If the player is human, then update the global human input as well
    if event == player.key_list[0]:    # w/UP
        if is_hum: hum_input = 0
        return 0, -snake_block              # x_change, y_change
    elif event == player.key_list[1]:  # a/LEFT
        if is_hum: hum_input = 3
        return -snake_block, 0
    elif event == player.key_list[2]:  # s/DOWN
        if is_hum: hum_input = 2
        return 0, snake_block
    elif event == player.key_list[3]:  # d/RIGHT
        if is_hum: hum_input = 1
        return snake_block, 0


def the_score(p1_score, p2_score):
    value = score_font.render("Player 1 Score: " + str(p1_score) + " Player 2 Score: " + str(p2_score), True, p1_score)
    display.blit(value, [0, 0])


def snake(block, snake_list):
    pygame.draw.rect(display, outer, [int(snake_list[-1][0]), int(snake_list[-1][1]), block, block])
    pygame.draw.rect(display, head, [int(snake_list[-1][0]), int(snake_list[-1][1]), block - 2, block - 2])

    for x in snake_list[0:-1]:
        pygame.draw.rect(display, outer, [int(x[0]), int(x[1]), block, block])
        pygame.draw.rect(display, body, [int(x[0]), int(x[1]), block - 2, block - 2])


def message(msg, color):
    mesg = font_style.render(msg, True, color)
    display.blit(mesg, [int(display_width / 6), int(display_height / 3)])


def game_loop():
    player_one = Snake(True, player_one_keys)
    player_two = Snake(False, player_two_keys)
    game_over = False
    game_close = False
    p1_x = display_width * (1/3)
    p2_x = display_width * (2/3)
    p1_y = display_height / 2
    p2_y = display_height / 2
    p1_x_change = 0
    p2_x_change = 0
    p1_y_change = 0
    p2_y_change = 0
    foodx = snake_block * random.randint(0, (display_width / snake_block) - 1)
    foody = snake_block * random.randint(0, (display_height / snake_block) - 1)

    while not game_over:
        while game_close:
            display.fill(bg)
            # Placeholder to get single player snake working
            message("You lost, press P to play again or Q to quit", red)
            the_score(player_one.length, player_two.length)
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game_over = True
                    game_close = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        game_over = True
                        game_close = False
                    if event.key == pygame.K_p:
                        game_loop()

        loop(hum_input)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key in player_one_keys:
                    p1_x_change, p1_y_change = update_move(player_one, event.key)
                elif event.key in player_two_keys:
                    p2_x_change, p2_y_change = update_move(player_two, event.key, True)

        if p1_x >= display_width or p1_x < 0 or p1_y >= display_height or p1_y < 0:
            game_close = True
        if p2_x >= display_width or p2_x < 0 or p2_y >= display_height or p2_y < 0:
            game_close = True

        p1_x += p1_x_change
        p1_y += p1_y_change
        p2_x += p2_x_change
        p2_y += p2_y_change

        display.fill(bg)
        pygame.draw.rect(display, food, [foodx, foody, snake_block, snake_block])
        p1_snake_head = [p1_x, p1_y]
        player_one.snake_list.append(p1_snake_head)

        p2_snake_head = [p2_x, p2_y]
        player_two.snake_list.append(p2_snake_head)

        if len(player_one.snake_list) > player_one.length:
            del player_one.snake_list[0]

        if len(player_two.snake_list) > player_two.length:
            del player_two.snake_list[0]

        for snake_list in [player_one.snake_list, player_two.snake_list]:
            for x in snake_list[:-1]:
                if x == p1_snake_head:
                    game_close = True

        snake(snake_block, player_one.snake_list)
        snake(snake_block, player_two.snake_list)
        the_score(player_one.length-1, player_two.length-1)

        pygame.display.update()

        if p1_x == foodx and p1_y == foody:
            foodx = snake_block * random.randint(0, (display_width / snake_block) - 1)
            foody = snake_block * random.randint(0, (display_height / snake_block) - 1)
            player_one.length += 1
        elif p2_x == foodx and p2_y == foody:
            foodx = snake_block * random.randint(0, (display_width / snake_block) - 1)
            foody = snake_block * random.randint(0, (display_height / snake_block) - 1)
            player_two.length += 1

        clock.tick(snake_speed)
    pygame.quit()


game_loop()
