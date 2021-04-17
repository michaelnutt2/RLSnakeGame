from gym_snake.envs.snake import Snake
from gym_snake.envs.snake import Grid
import numpy as np

class Controller():
    """
    This class combines the Snake, Food, and Grid classes to handle the game logic.
    """

    def __init__(self, grid_size=[30,30], unit_size=10, unit_gap=1, snake_size=3, n_snakes=1, n_foods=1, random_init=True):

        assert n_snakes < grid_size[0]//3
        assert n_snakes < 25
        assert snake_size < grid_size[1]//2
        assert unit_gap >= 0 and unit_gap < unit_size

        self.snakes_remaining = n_snakes
        self.grid = Grid(grid_size, unit_size, unit_gap)

        self.snakes = []
        self.dead_snakes = []
        for i in range(1,n_snakes+1):
            start_coord = [i*grid_size[0]//(n_snakes+1), snake_size+1]
            self.snakes.append(Snake(start_coord, snake_size))
            color = [self.grid.HEAD_COLOR[0], i*10, 0]
            self.snakes[-1].head_color = color
            self.grid.draw_snake(self.snakes[-1], color)
            self.dead_snakes.append(None)

        if not random_init:
            for i in range(2,n_foods+2):
                start_coord = [i*grid_size[0]//(n_foods+3), grid_size[1]-5]
                self.grid.place_food(start_coord)
        else:
            for i in range(n_foods):
                self.grid.new_food()

    def move_snake(self, direction, snake_idx):
        """
        Moves the specified snake according to the game's rules dependent on the direction.
        Does not draw head and does not check for reward scenarios. See move_result for these
        functionalities.
        """

        snake = self.snakes[snake_idx]
        if type(snake) == type(None):
            return

        # Cover old head position with body
        self.grid.cover(snake.head, self.grid.BODY_COLOR)
        # Erase tail without popping so as to redraw if food eaten
        self.grid.erase(snake.body[0])
        # Find and set next head position conditioned on direction
        snake.action(direction)

    def move_result(self, direction, snake_idx=0):
        """
        Checks for food and death collisions after moving snake. Draws head of snake if
        no death scenarios.
        """

        snake = self.snakes[snake_idx]
        if type(snake) == type(None):
            return 0

        # Check for death of snake
        if self.grid.check_death(snake.head):
            self.dead_snakes[snake_idx] = self.snakes[snake_idx]
            self.snakes[snake_idx] = None
            self.grid.cover(snake.head, snake.head_color) # Avoid miscount of grid.open_space
            self.grid.connect(snake.body.popleft(), snake.body[0], self.grid.SPACE_COLOR)
            reward = -1.0
        # Check for reward
        elif self.grid.food_space(snake.head):
            self.grid.draw(snake.body[0], self.grid.BODY_COLOR) # Redraw tail
            self.grid.connect(snake.body[0], snake.body[1], self.grid.BODY_COLOR)
            self.grid.cover(snake.head, snake.head_color) # Avoid miscount of grid.open_space
            reward = 1.0
            self.grid.new_food()
        else:
            reward = 1/np.linalg.norm(self.grid.food_coord - snake.head)
            empty_coord = snake.body.popleft()
            self.grid.connect(empty_coord, snake.body[0], self.grid.SPACE_COLOR)
            self.grid.draw(snake.head, snake.head_color)

        self.grid.connect(snake.body[-1], snake.head, self.grid.BODY_COLOR)

        return reward

    def kill_snake(self, snake_idx):
        """
        Deletes snake from game and subtracts from the snake_count 
        """
        
        assert self.dead_snakes[snake_idx] is not None
        self.grid.erase(self.dead_snakes[snake_idx].head)
        self.grid.erase_snake_body(self.dead_snakes[snake_idx])
        self.dead_snakes[snake_idx] = None
        self.snakes_remaining -= 1
    
    def get_snake_info(self):
        snake_info = []
        
        for snake in range(len(self.snakes)):
            agent = self.snakes[snake]
            if agent is not None :
                H = agent.head
                surr_coord = ( (H[0] + 1, H[1]), (H[0] - 1, H[1]), (H[0], H[1] + 1), (H[0], H[1] - 1) )
                surr_info = []
            
                for i in range(4):
                    if self.grid.check_death(surr_coord[i]):
                        surr_info.append(np.array(surr_coord[i] + (1,)))
                    elif self.grid.food_space(surr_coord[i]):
                        surr_info.append(np.array(surr_coord[i] + (2,)))
                    else:
                        surr_info.append(np.array(surr_coord[i] + (0,)))
            
                # Return the euclidean distance
                food_dist = np.linalg.norm(self.grid.food_coord - agent.head)
                
                food_diff = [self.grid.food_coord[0] - H[0] , self.grid.food_coord[1] - H[1]]
                food_diff.append(food_dist)
                
                surr_info.append(food_diff)
                snake_info.append(surr_info)
            else:
                neg_1s = [-1,-1,-1]
                surr_info = []
                surr_info.append(neg_1s)
                surr_info.append(neg_1s)
                surr_info.append(neg_1s)
                surr_info.append(neg_1s)
                surr_info.append(neg_1s)
                snake_info.append(surr_info)
            
        return snake_info

    def step(self, directions):
        """
        Takes an action for each snake in the specified direction and collects their rewards
        and dones.

        directions - tuple, list, or ndarray of directions corresponding to each snake.
        """

        # Fawad's Code #

        # Return a list of the spaces surrounding the head of the snake [SPACE = 0, BODY/HEAD/WALL = 1, FOOD = 2]. 
        # A wall will be considered a Body 
        snake_info = self.get_snake_info()

        ###############################


        # Ensure no more play until reset
        if self.snakes_remaining < 1 or self.grid.open_space < 1:
            if type(directions) == type(int()) or len(directions) is 1:
                return self.grid.grid.copy(), 0, True, {"snakes_remaining":self.snakes_remaining}
            else:
                return self.grid.grid.copy(), [0]*len(directions), True, {"snakes_remaining":self.snakes_remaining}

        rewards = []

        if type(directions) == type(int()):
            directions = [directions]

        for i, direction in enumerate(directions):
            if self.snakes[i] is None and self.dead_snakes[i] is not None:
                self.kill_snake(i)
            self.move_snake(direction,i)
            rewards.append(self.move_result(direction, i))

        done = self.snakes_remaining < 1 or self.grid.open_space < 1
        if len(rewards) is 1:
            return snake_info, rewards[0], done, {"snakes_remaining":self.snakes_remaining}
        else:
            return snake_info, rewards, done, {"snakes_remaining":self.snakes_remaining}

    
