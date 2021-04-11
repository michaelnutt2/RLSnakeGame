"""
Running the game
"""

import pygame
import random
pygame.init()

bg = (150, 150, 150)
h = (255, 150, 20)
body = (0, 200, 255)
fo = (0, 200, 255)
sc = (255, 150, 20)
red = (213, 50, 80)
outer = (100, 100, 200)

display_width = 600
display_hwidth = 300
display_height = 400
display_hheight = 200
display = pygame.display.set_mode((display_width, display_height))
clock = pygame.time.Clock()
snake_block = 10
snake_speed = 15
font_style = pygame.font.SysFont("banschrift", 25)
score_font = pygame.font.SysFont("freesans", 35)


def the_score(score):
    value = score_font.render("Your Score: " + str(score), True, sc)
    display.blit(value, [0, 0])


def player_snake(snake_block, snake_list):
    pygame.draw.rect(display, outer, [int(snake_list[-1][0]), int(snake_list[-1][1]), snake_block, snake_block])
    pygame.draw.rect(display, h, [int(snake_list[-1][0]), int(snake_list[-1][1]), snake_block-2, snake_block-2])

    for x in snake_list[0:-1]:
        pygame.draw.rect(display, outer, [int(x[0]), int(x[1]), snake_block, snake_block])
        pygame.draw.rect(display, body, [int(x[0]), int(x[1]), snake_block-2, snake_block-2])


def message(msg, color):
    mesg = font_style.render(msg, True, color)
    display.blit(mesg, [int(display_width / 6), int(display_height / 3)])


def game_loop():
    game_over = False
    game_close = False
    x1 = display_width / 2
    y1 = display_height / 2
    x1_change = 0
    y1_change = 0
    snake_list = []
    length_of_snake = 1
    foodx = snake_block * random.randint(0, (display_width / snake_block) - 1)
    foody = snake_block * random.randint(0, (display_height / snake_block) - 1)

    while not game_over:
        while game_close:
            display.fill(bg)
            message("You lost, press P to play again or Q to quit", red) # Placeholder to get single player snake working
            the_score(length_of_snake - 1)
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
                if event.key == pygame.K_a: # left
                    x1_change = -snake_block
                    y1_change = 0
                elif event.key == pygame.K_d:   # right
                    x1_change = snake_block
                    y1_change = 0
                elif event.key == pygame.K_w:   # up
                    x1_change = 0
                    y1_change = -snake_block
                elif event.key == pygame.K_s:   # down
                    x1_change = 0
                    y1_change = snake_block
        if x1 >= display_width or x1 < 0 or y1 >= display_height or y1 < 0:
            game_close = True
        x1 += x1_change
        y1 += y1_change
        display.fill(bg)
        pygame.draw.rect(display, fo, [foodx, foody, snake_block, snake_block])
        snake_head = []
        snake_head.append(x1)
        snake_head.append(y1)
        snake_list.append(snake_head)

        if len(snake_list) > length_of_snake:
            del snake_list[0]

        for x in snake_list[:-1]:
            if x == snake_head:
                game_close = True

        player_snake(snake_block, snake_list)
        the_score(length_of_snake-1)

        pygame.display.update()

        if x1 == foodx and y1 == foody:
            foodx = snake_block * random.randint(0, (display_width / snake_block) - 1)
            foody = snake_block * random.randint(0, (display_height / snake_block) - 1)
            length_of_snake += 1
        clock.tick(snake_speed)
    pygame.quit()


game_loop()
