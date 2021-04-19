import gym

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

game_on = False
testing = True

if game_on:
    from pyautogui import press
    from two_headed_snake import hum_input

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
models = [create_q_model(),create_q_model()]
# Build a target model for the prediction of future rewards.
# The weights of a target model get updated every 10000 steps thus when the
# loss between the Q-values is calculated the target Q-value is stable.
model_targets = [create_q_model(),create_q_model()]


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

if testing or game_on == True:
    epsilon_random_frames = 0
    epsilon_greedy_frames = 1
    epsilon = 0.1
    epsilon_min = 0.1
    models[0] = keras.models.load_model("snake0_28500_03.keras")
    models[1] = keras.models.load_model("snake1_28500_03.keras")
    model_targets[0].set_weights(models[0].get_weights())
    model_targets[1].set_weights(models[1].get_weights())
    max_memory_length = 100

while True:  # Run until solved
    state = np.array(env.reset())
    game_controller = env.controller
    episode_reward = [0,0]
    print("On episode: ", episode_count)
    for timestep in range(1, max_steps_per_episode):
        if not game_on and testing:
            env.render(); #Adding this line would show the attempts
        # of the agent in a pop up window.
        frame_count += 1
        if frame_count%1000 == 0:
            print(frame_count)
        
        state = game_controller.get_snake_info()

        # Use epsilon-greedy for exploration for snake 0
        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            # Take random action
            
            action0 = np.random.choice(num_actions)
            
            
            
            if testing or game_on == True: #No more random wall/body collision
               action0 = avoid_collision(state,0,action0)
            
            if game_on == True:
                press(dir_to_key[action0])

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
            
            action0 = avoid_collision(state,0,action0)
            
            for p_a in range(4): #just grab the food lol
                if state[0][p_a][2] == 2:
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
        else:
            if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                # Take random action
                action1 = np.random.choice(num_actions)
                
                if testing or game_on == True: #No more random wall/body collision
                    action1 = avoid_collision(state,1,action1)
                   
            else:
                # Predict action Q-values
                # From environment state
                state = tf.cast(state, tf.float32)
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = models[1](state_tensor, training=False)
                # Take best action
                #print("Taking educated guess snake 1: ")
                #print(random.choice(pd.DataFrame(action_probs.numpy()[0]).nlargest(n=1,columns=[0],keep='all').index))
                action1 = random.choice(pd.DataFrame(action_probs.numpy()[0]).nlargest(n=1,columns=[0],keep='all').index)
                
                action1 = avoid_collision(state,1,action1)
                
                for p_a in range(4): #just grab the food lol
                    if state[1][p_a][2] == 2:
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

        if done:
            break

    # Update running reward to check condition for solving
    episode_reward_history[0].append(episode_reward[0])
    episode_reward_history[1].append(episode_reward[1])
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward0 = np.mean(episode_reward_history[0])
    running_reward1 = np.mean(episode_reward_history[1])

    if not testing and game_on == False and episode_count%10 == 0:   
        models[0].save("snake0_"+str(episode_count)+"_03.keras")
        models[1].save("snake1_"+str(episode_count)+"_03.keras")

    episode_count += 1

    if not testing and game_on == False and (running_reward0 > 10 * 20):  # Condition to consider the task solved
        print("Solved at episode {} with snake 0!".format(episode_count))
        models[0].save("snake0_"+str(episode_count)+"_03_done.keras")
        models[1].save("snake1_"+str(episode_count)+"_03_done.keras")
        break
    
    if not testing and game_on == False and (running_reward1 > 10 * 20):  # Condition to consider the task solved
        print("Solved at episode {} with snake 1!".format(episode_count))
        models[0].save("snake0_"+str(episode_count)+"_03_done.keras")
        models[1].save("snake1_"+str(episode_count)+"_03_done.keras")
        break

