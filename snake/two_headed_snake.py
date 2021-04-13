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
snake_block = 20
snake_speed = 15

player_one_keys = [pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d]
player_two_keys = [pygame.K_UP, pygame.K_LEFT, pygame.K_DOWN, pygame.K_RIGHT]


class Snake:
    def __init__(self, ai, keys):
        self.snake_list = []
        self.is_ai = ai
        self.length = 1
        self.key_list = keys    # w/up, a/left, s/down, d/right


def update_move(player, event):
    if event == player.key_list[0]:    # w/UP
        return 0, -snake_block              # x_change, y_change
    elif event == player.key_list[1]:  # a/LEFT
        return -snake_block, 0
    elif event == player.key_list[2]:  # s/DOWN
        return 0, snake_block
    elif event == player.key_list[3]:  # d/RIGHT
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
    # TODO rework to take input to determine if players or ai
    player_one = Snake(False, player_one_keys)
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

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key in player_one_keys:
                    p1_x_change, p1_y_change = update_move(player_one, event.key)
                elif event.key in player_two_keys:
                    p2_x_change, p2_y_change = update_move(player_two, event.key)

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
