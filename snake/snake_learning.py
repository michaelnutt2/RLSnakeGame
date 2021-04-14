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
max_steps_per_episode = 100000

num_actions = 4


def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(200, 200, 3,))

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(num_actions, activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=action)


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
epsilon_random_frames = 500
# Number of frames for exploration
epsilon_greedy_frames = 10000.0
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_length = 10000
# Train the model after 4 actions
update_after_actions = 4
# How often to update the target network
update_target_network = 100
# Using huber loss for stability
loss_function = keras.losses.Huber()

while True:  # Run until solved
    state = np.array(env.reset())
    episode_reward = [0,0]
    print("On episode: ", episode_count)
    for timestep in range(1, max_steps_per_episode):
        env.render(); #Adding this line would show the attempts
        # of the agent in a pop up window.
        frame_count += 1

        # Use epsilon-greedy for exploration for snake 0
        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            # Take random action
            
            action0 = np.random.choice(num_actions)
        else:
            # Predict action Q-values
            # From environment state
            print("Taking educated guess: ")
            state = tf.cast(state, tf.float32)
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = models[0](state_tensor, training=False)
            # Take best action
            print(tf.argmax(action_probs).numpy())
            action0 = tf.argmax(action_probs).numpy()
            
        # Use epsilon-greedy for exploration for snake 1
        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            # Take random action
            action1 = np.random.choice(num_actions)
        else:
            # Predict action Q-values
            # From environment state
            state = tf.cast(state, tf.float32)
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = models[1](state_tensor, training=False)
            # Take best action
            print("Taking educated guess: ")
            print(tf.argmax(action_probs).numpy())
            action1 = tf.argmax(action_probs).numpy()

        # Decay probability of taking random action
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)

        # Apply the sampled action in our environment
        state_next, reward, done, _ = env.step([action0,action1])
        state_next = np.array(state_next)

        episode_reward[0] += reward[0]
        episode_reward[1] += reward[1]
        

        # Save actions and states in replay buffer
        action_history.append([action0,action1])
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(done)
        rewards_history.append(reward)
        state = state_next

        # Update every fourth frame and once batch size is over 32
        if frame_count % update_after_actions == 0 and len(done_history) > batch_size:

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
                q_values = models[0](state_sample0)

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

            # If final frame set the last value to -1
            updated_q_values1 = updated_q_values1 * (1 - done_sample1) - done_sample1

            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample1, num_actions)
            
            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                state_sample1 = tf.cast(state_sample1, tf.float32)
                q_values = models[1](state_sample1)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values1, q_action)

            # Backpropagation
            grads = tape.gradient(loss, models[1].trainable_variables)
            optimizer.apply_gradients(zip(grads, models[1].trainable_variables))



        if frame_count % update_target_network == 0:
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

    if episode_count%10 == 0:   
        models[0].save("snake0_"+str(episode_count))
        models[1].save("snake1_"+str(episode_count))

    episode_count += 1

    if running_reward0 > 40 or running_reward1 > 40:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break

