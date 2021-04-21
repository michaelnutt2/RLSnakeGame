'''
Preparing Gym Environment
'''
from time import time

import gym
#from snake_learning import loop

env_dict = gym.envs.registration.registry.env_specs.copy()


for env in env_dict:
    if 'snake-plural-v0' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]
    if 'snake-v0' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]


import gym_snake
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
env.snake_size = 2
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
models[1] = keras.models.load_model("snake/snakeRL_5040_FINAL.keras")
models[0] = keras.models.load_model("snake/snakeRL_5040_FINAL.keras")
model_targets[0].set_weights(models[0].get_weights())
model_targets[1].set_weights(models[1].get_weights())
max_memory_length = 100

state = np.array(env.reset())
game_controller = env.controller
episode_reward = [0,0]

########################################################################
def loop(hum_input,state,epsilon,epsilon_random_frames,epsilon_min):
    if not game_on and testing:
        env.render(); #Adding this line would show the attempts
    # of the agent in a pop up window.
    frame_count = 10
    if frame_count%1000 == 0:
        pass
        #print(frame_count)

    state_c = game_controller.get_snake_info()

    # Use epsilon-greedy for exploration for snake 0
    if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
        # Take random action

        action0 = np.random.choice(num_actions)


        if testing or game_on == True: #No more random wall/body collision
            action0 = avoid_collision(state_c,0,action0)

        if game_on == True:
            pass
            #press(dir_to_key[action0])

        #TODO: Input the action of the player 2 human


        #TODO: Add algorithm to avoid walls and direct towards food


    else:
        # Predict action Q-values
        # From environment state

        state = tf.cast(state, tf.float32)
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = models[0](state_tensor, training=False)
        # Take best action
        #print("Taking educated guess snake 0: ")
        #print(random.choice(pd.DataFrame(action_probs.numpy()[0]).nlargest(n=1,columns=[0],keep='all').index))


        action0 = random.choice(pd.DataFrame(action_probs.numpy()[0]).nlargest(n=1,columns=[0],keep='all').index)

        action0 = avoid_collision(state_c,0,action0)

        for p_a in range(4): #just grab the food lol
            if state_c[0][p_a][2] == 2:
                action0 = p_a
                if action0 == 0:
                    action0 = 1
                elif action0 == 1:
                    action0 = 3
                elif action0 == 3:
                    action0 = 0
                print("Forcing food 0 with action ", action0)

    # Use epsilon-greedy for exploration for snake 1
    if game_on == True:
        # Human player input = action 1
        action1 = hum_input
        #print(action1)
    else:
        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            # Take random action
            action1 = np.random.choice(num_actions)

            if testing or game_on == True: #No more random wall/body collision
                action1 = avoid_collision(state,1,action1)

        else:
            # Predict action Q-values
            # From environment state
            state1 = [[],[]]
            state1[0] = state[1]
            state1[1] = state[0]

            state1 = tf.cast(state1, tf.float32)
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = models[1](state_tensor, training=False)
            # Take best action
            #print("Taking educated guess snake 1: ")
            #print(random.choice(pd.DataFrame(action_probs.numpy()[0]).nlargest(n=1,columns=[0],keep='all').index))
            action1 = random.choice(pd.DataFrame(action_probs.numpy()[0]).nlargest(n=1,columns=[0],keep='all').index)

            action1 = avoid_collision(state_c,1,action1)

            for p_a in range(4): #just grab the food lol
                if state_c[1][p_a][2] == 2:
                    action1 = p_a
                    if action1 == 0:
                        action1 = 1
                    elif action1 == 1:
                        action1 = 3
                    elif action1 == 3:
                        action1 = 0
                    print("Forcing food 1 with action ", action1)




    # Decay probability of taking random action
    epsilon -= epsilon_interval / epsilon_greedy_frames
    epsilon = max(epsilon, epsilon_min)

    # Apply the sampled action in our environment
    state_next, reward, done, _ = env.step([action0,action1])
    #print(reward)
    state_next = np.array(state_next)

    if reward[0] >= 1.0 or reward[0] <= -1.0:
        episode_reward[0] += reward[0]
    if reward[1] >= 1.0 or reward[1] <= -1.0:
        episode_reward[1] += reward[1]


    # Save actions and states in replay buffer
    action_history.append([action0,action1])
    state_history.append(state)
    state_next_history.append(state_next)
    done_history.append(done)
    rewards_history.append(reward)
    state = state_next

    # Update every fourth frame and once batch size is over 32
    if not testing and (game_on == False and frame_count % update_after_actions == 0 and len(done_history) > batch_size):

        # Get indices of samples for replay buffers
        indices = np.random.choice(range(len(done_history)), size=batch_size)

        # Using list comprehension to sample from replay buffer snake 0
        state_sample0 = np.array([state_history[i] for i in indices])
        state_next_sample0 = np.array([state_next_history[i] for i in indices])
        rewards_sample0 = [rewards_history[i][0] for i in indices]
        action_sample0 = [action_history[i][0] for i in indices]
        done_sample0 = tf.convert_to_tensor(
            [float(done_history[i]) for i in indices]
        )

        # Build the updated Q-values for the sampled future states
        # Use the target model for stability
        future_rewards0 = model_targets[0].predict(state_next_sample0)
        #print(tf.reduce_max(future_rewards0, axis=1))
        # Q value = reward + discount factor * expected future reward
        updated_q_values0 = rewards_sample0 + gamma * tf.reduce_max(
            future_rewards0, axis=1
        )

        # If final frame set the last value to -1
        updated_q_values0 = updated_q_values0 * (1 - done_sample0) - done_sample0

        # Create a mask so we only calculate loss on the updated Q-values
        masks = tf.one_hot(action_sample0, num_actions)
        with tf.GradientTape() as tape:
            # Train the model on the states and updated Q-values
            state_sample0 = tf.cast(state_sample0, tf.float32)
            q_values = models[0](state_sample0,training=True)

            # Apply the masks to the Q-values to get the Q-value for action taken
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            # Calculate loss between new Q-value and old Q-value
            loss = loss_function(updated_q_values0, q_action)

        # Backpropagation
        grads = tape.gradient(loss, models[0].trainable_variables)
        optimizer.apply_gradients(zip(grads, models[0].trainable_variables))

        # Using list comprehension to sample from replay buffer snake 1
        state_sample1 = np.array([state_history[i] for i in indices])
        state_next_sample1 = np.array([state_next_history[i] for i in indices])
        rewards_sample1 = [rewards_history[i][1] for i in indices]
        action_sample1 = [action_history[i][1] for i in indices]
        done_sample1 = tf.convert_to_tensor(
            [float(done_history[i]) for i in indices]
        )

        # Build the updated Q-values for the sampled future states
        # Use the target model for stability
        future_rewards1 = model_targets[1].predict(state_next_sample1)
        # Q value = reward + discount factor * expected future reward
        updated_q_values1 = rewards_sample1 + gamma * tf.reduce_max(
            future_rewards1, axis=1
        )
        #print(tf.reduce_max(future_rewards1, axis=1))
        # If final frame set the last value to -1
        updated_q_values1 = updated_q_values1 * (1 - done_sample1) - done_sample1

        # Create a mask so we only calculate loss on the updated Q-values
        masks = tf.one_hot(action_sample1, num_actions)

        with tf.GradientTape() as tape:
            # Train the model on the states and updated Q-values
            state_sample1 = tf.cast(state_sample1, tf.float32)
            q_values = models[1](state_sample1,training=True)

            # Apply the masks to the Q-values to get the Q-value for action taken
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            # Calculate loss between new Q-value and old Q-value
            loss = loss_function(updated_q_values1, q_action)

        # Backpropagation
        grads = tape.gradient(loss, models[1].trainable_variables)
        optimizer.apply_gradients(zip(grads, models[1].trainable_variables))



    if not testing and game_on == False and frame_count % update_target_network == 0:
        # update the the target network with new weights
        model_targets[0].set_weights(models[0].get_weights())
        model_targets[1].set_weights(models[1].get_weights())


    # Limit the state and reward history
    if len(rewards_history) > max_memory_length:
        del rewards_history[:1]
        del state_history[:1]
        del state_next_history[:1]
        del action_history[:1]
        del done_history[:1]

    return action0

#####################################################


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
display_hwidth = 600
display_height = 600
display_hheight = 600
display = pygame.display.set_mode((display_width, display_height))
clock = pygame.time.Clock()
font_style = pygame.font.SysFont("banschrift", 25)
score_font = pygame.font.SysFont("freesans", 35)
snake_block = 30
snake_speed = 10

hum_input = -1   # human input to be read by model


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
    global hum_input
    if is_hum:
        if event == player.key_list[0]:    # w/UP
            if is_hum:
                hum_input = 0
            return 0, -snake_block              # x_change, y_change
        elif event == player.key_list[1]:  # a/LEFT
            if is_hum:
                hum_input = 3
            return -snake_block, 0
        elif event == player.key_list[2]:  # s/DOWN
            if is_hum:
                hum_input = 2
            return 0, snake_block
        elif event == player.key_list[3]:  # d/RIGHT
            if is_hum:
                hum_input = 1
            return snake_block, 0
    else :
        if event == 0:    # w/UP
            if is_hum:
                hum_input = 0
            return 0, -snake_block              # x_change, y_change
        elif event == 3:  # a/LEFT
            if is_hum:
                hum_input = 3
            return -snake_block, 0
        elif event == 2:  # s/DOWN
            if is_hum:
                hum_input = 2
            return 0, snake_block
        elif event == 1:  # d/RIGHT
            if is_hum:
                hum_input = 1
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


def taunt(taunts, color):
    select = random.randint(0,len(taunts)-1)
    selected = taunts[select]
    if selected[-1] == 'G':
        if color == 'blue':
            color = 'purple'
        elif color == 'red':
            color = 'pink'
    print(selected)
    return selected, color

def game_loop():



    global game_controller
    global state
    global hum_input

    player_one = Snake(True, player_one_keys)
    player_two = Snake(False, player_two_keys)
    game_over = False
    game_close = False
    p1_x = snake_block * game_controller.snakes[0].head[0]
    p2_x = snake_block * game_controller.snakes[1].head[0]
    p1_y = snake_block * game_controller.snakes[0].head[1]
    p2_y = snake_block * game_controller.snakes[1].head[1]
    p1_x_change = 0
    p2_x_change = 0
    p1_y_change = 0
    p2_y_change = 0
    taunting = False
    foodx = game_controller.grid.food_coord[0] * snake_block
    foody = game_controller.grid.food_coord[1] * snake_block
    print("food_coord ", foody)

    gTaunts = bTaunts = nTaunts = []
    with open('snake/GptBotTaunt.csv', 'r+') as i_f:
        while True:
            line = i_f.readline()
            if line == "" :
                    break
            bTaunts.append(line)
            
    """with open("NeutralTaunts.csv", 'r+') as i_f:
        while True:
            line = i_f.readline()
            if line == "" :
                    break
            nTaunts.append(line)"""

    with open("snake/GptPlayerTaunt.csv", 'r+') as i_f:
        while True:
            line = i_f.readline()
            if line == "" :
                    break
            gTaunts.append(line)

    # Remove duplicates
    _gTaunts = set(gTaunts)
    gTaunts = list(_gTaunts)
    _bTaunts = set(bTaunts)
    bTaunts = list(_bTaunts)

    while not game_over:
        while game_close:
            display.fill(bg)
            # Placeholder to get single player snake working
            message("Game over, press P to play again or Q to quit", red)
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
                        state = np.array(env.reset())
                        hum_input = -1
                        game_controller = env.controller
                        game_loop()

        foodx = game_controller.grid.food_coord[0] * snake_block
        foody = game_controller.grid.food_coord[1] * snake_block



        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key in player_one_keys:
                    #p1_x_change, p1_y_change = update_move(player_one, event.key)
                    pass
                elif event.key in player_two_keys:
                    #print(hum_input)
                    p2_x_change, p2_y_change = update_move(player_two, event.key, True)

        if hum_input != -1:
            action = loop(hum_input,state,epsilon,epsilon_random_frames,epsilon_min)
            p1_x_change, p1_y_change = update_move(player_one, action, False)



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
            game_controller.grid.new_food()
            foodx = game_controller.grid.food_coord[0] * snake_block
            foody = game_controller.grid.food_coord[1] * snake_block

            player_one.length += 1
            # Hannah
            """
            if player_one.length == player_two.length and not taunting:
                taunt_msg = taunt(nTaunts, 'blue')
                taunting = True
                start_time = time()"""
            if not taunting:
                taunt_msg = taunt(gTaunts, 'blue')
                taunting = True
                start_time = time()
            ####
        elif p2_x == foodx and p2_y == foody:
            game_controller.grid.new_food()
            foodx = game_controller.grid.food_coord[0] * snake_block
            foody = game_controller.grid.food_coord[1] * snake_block
            player_two.length += 1
            # Hannah
            """if player_one.length == player_two.length and not taunting:
                taunt_msg = taunt(nTaunts, 'yellow')
                taunting = True
                start_time = time()"""
            if not taunting:
                taunt_msg = taunt(bTaunts, 'red')
                taunting = True
                start_time = time()
            ####

        if taunting:
            curr_time = time()
            if (curr_time - start_time) < 3:
                message(taunt_msg[0], taunt_msg[1])
                pygame.display.update()
            else:
                taunting = False
        clock.tick(snake_speed)
    pygame.quit()


game_loop()
