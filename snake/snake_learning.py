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

epochs = 5
episodes = 50
espilon = 0.2
for i in range(episodes):
    env.render()
    #print(observation)
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    #print(observation, reward, done, info)
    if done:
        print("Finished after {} timesteps".format(i+1))
        break


