import os
import pickle
import random
import time
from xmlrpc.client import MAXINT

import numpy as np

from MonteCarloTreeSearch import mcts_parent_function,mcts_decisive_parent_function
from SetupAndConfig import columns,rows,players_token_colour,block_size,size_of_screen,width,height
from Minimax import min_max,iterative_deepening_minimax
from EvaluationFunctions import win_check_position,score_board_simplified
from HandleMoveFunction import handle_move
from RLModelContainerClass import RlModelsContainer
import pygame



# This ensures that the who goes first, as P1 is a random 50/50
def select_first_player():
    if random.randint(0, 100) >= 000:
        return True
    return False

# This function is used to get the players move, this is slightly complex as this is done using PyGame opposed to the console for convenience
def get_player_move(cols_valid_moves, current_player):
    col_selected = 99 #The col selected can start at any value not in the valid col range of 0 -> 6
    previous_anticipated_col = -1 #this is used to tell the animate_anticipated_col function what the previous anticipated col was so the old token can be removed

    while col_selected not in cols_valid_moves: # The while loop forces a user to make a valid action
        #animate_anticipated_col is responsible for the hovering animation when the user is deciding their move
        previous_anticipated_col=animate_anticipated_col(previous_anticipated_col, players_token_colour[current_player])
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                col_selected = event.pos[0] // 125
                if col_selected in cols_valid_moves:
                    return handle_move(col_selected, cols_valid_moves)

# This function handles the users move
def player_move_handler(cols_valid_moves, player):
    insertion_position = get_player_move(cols_valid_moves, player)
    return insertion_position

# This function is where we handle getting the correct AI to make a move,
# Keep in mind the handle move function must be called when a move is decided as this keeps the cols_valid_moves up to date
def ai_move_handler(board, cols_valid_moves, opponent, player_token):
    if opponent == "lg" or opponent =="local game":
        return player_move_handler(cols_valid_moves, player_token)
    elif opponent == "rnd" or opponent =="random":
        return handle_move(random.choice(list(cols_valid_moves.keys())), cols_valid_moves)
    elif opponent=="grd" or opponent =="greedy":
        greedy = min_max(board, cols_valid_moves, (0, 0), True, player_token, 1, -MAXINT, MAXINT)[1]
        return handle_move(greedy, cols_valid_moves)
    elif opponent == "min" or opponent =="minmax":
        move = min_max(board, cols_valid_moves, (0, 0), True, player_token, 12, -MAXINT, MAXINT)[1]
        return handle_move(move, cols_valid_moves)
    elif opponent== "minid" or opponent =="minmax iterative deepening":
        move= iterative_deepening_minimax(board, cols_valid_moves, player_token, 15, 2)
        return handle_move(move, cols_valid_moves)
    elif opponent == "mcts" or opponent =="monte carlo tree search":
        move = mcts_parent_function(board, cols_valid_moves, player_token, 2)
        return handle_move(move, cols_valid_moves)
    elif opponent == "mctsd" or opponent =="monte carlo tree search decisive":
        move = mcts_decisive_parent_function(board, cols_valid_moves, player_token, 2)
        return handle_move(move, cols_valid_moves)
    elif  opponent == "ql" or opponent =="q-learning":
        state=models.get_state(board,player_token)
        return handle_move(models.q_learning_action_selector(state, cols_valid_moves), cols_valid_moves)
    elif opponent== "dqn" or opponent =="deep q network":
        return handle_move(models.dqn_model.select_best_action(board, cols_valid_moves, player_token), cols_valid_moves)
    else:
        return player_move_handler(cols_valid_moves, player_token)

# The game_play function is where we control the flow of the current game, correctly handling each players turn,
# Handling and checking for wins.
def game_play(cols_valid_moves, board, opponent, ai_turn):
    # Track the time it takes the AI to move
    ai_total_execution_time=0
    player_total_execution_time=0
    draw_board(board)
    counter_count = 0
    current_player=1
    while cols_valid_moves:
        if ai_turn:
            start_time = time.time()
            insertion_position = ai_move_handler(board, cols_valid_moves, opponent, current_player)
            end = time.time()
            ai_total_execution_time += ((end - start_time) * 10 ** 3)
            print("MS to run code", (end - start_time) * 10 ** 3)

        else:
            start_time = time.time()
            insertion_position = player_move_handler(cols_valid_moves, current_player)
            end = time.time()
            player_total_execution_time+= ((end - start_time) * 10 ** 3)


        board[insertion_position[0]][insertion_position[1]] = current_player
        draw_board(board)
        reward = score_board_simplified(board, current_player)
        if ai_turn:
            reward=0-reward

        print(f"Current score = {reward}")
        counter_count += 1
        if counter_count > 6 and win_check_position(insertion_position[0],insertion_position[1], board, current_player):
            if ai_turn:
                player_id = "AI"
            else:
                player_id = "User"
            print("|--------------------------------------------------------------------------------------------------|\n"
                  "|=========================================== GAME OVER ============================================|\n"
                  f"|-----------                          The winner was {player_id}\n"
                  f"|-----------                        total execution time of AI {ai_total_execution_time}\n"
                  f"|-----------                       total execution time of player {player_total_execution_time}\n"
                  "|--------------------------------------------------------------------------------------------------|\n")

            time.sleep(3)
            break
        current_player = 3 - current_player
        ai_turn=not ai_turn

    start_game()

def animate_anticipated_col(current_anticipated_col, colour):
    hovering_col = (pygame.mouse.get_pos()[0]) // 125
    if hovering_col != current_anticipated_col:
        new_hovering_pos = (hovering_col * 125, block_size / 2)
        pygame.draw.circle(display, (0, 0, 0), ((current_anticipated_col * 125) + block_size / 2, new_hovering_pos[1]),
                           block_size / 2.5)
        pygame.draw.circle(display, colour, ((new_hovering_pos[0]) + block_size / 2, new_hovering_pos[1]),
                           block_size / 2.5)
        pygame.display.update()
    return hovering_col

def draw_board(board):
    pygame.draw.rect(display, (0, 100, 250), (0, 0 + block_size, width, height - block_size))
    for c in range(columns):
        for r in range(rows):
            if board[r][c] == 1:
                pygame.draw.circle(display, (200, 0, 0),(c * block_size + block_size / 2, r * block_size + block_size * 1.5),block_size / 2.5)
            elif board[r][c] == 2:
                pygame.draw.circle(display, (200, 200, 0),(c * block_size + block_size / 2, r * block_size + block_size * 1.5),block_size / 2.5)
            else:
                pygame.draw.circle(display, (0, 0, 0),(c * block_size + block_size / 2, r * block_size + block_size * 1.5), block_size / 2.5)
    pygame.display.update()
    print_board(board)

def print_board(board):
    print(board, 0)

def create_board():
    return np.zeros((rows, columns), dtype=np.uint8)

def start_game():
    opponent = input("What game-mode would you like to play? \n"
                    "Local Two Player Game = 'lg' \n"
                    "random = 'rnd' | 0 / 10 \n"
                    "minimax = 'min' | 8 / 10 \n"
                    "monte carlo tree search = 'mcts' | 8 / 10  \n"
                    "q learning = 'ql' | 4/10 \n"
                    "deep q network = 'dqn' | 4/10 \n"
                    "monte carlo tree search decisive = 'mctsd' | 8 / 10 \n"
                    "minimax iterative deepening = 'minid' | 9/10 \n"
                     "or simply enter quit or q if you wish to close the application.")
    if opponent!="q" and opponent!="quit":
        board = create_board()
        ai_turn = select_first_player()
        cols_valid_moves = {i: 5 for i in range(columns)}
        game_play(cols_valid_moves, board, opponent, ai_turn)

models=RlModelsContainer()
tutorials=False
if __name__ == "__main__":
    tutorial_response = input("Do you wish to see hints on how to play the game and more information about the game ?")
    if (tutorial_response=="yes" or tutorial_response=="ye" or tutorial_response=="YE" or tutorial_response=="Yeah"
            or tutorial_response=="yeah" or tutorial_response=="Yes"):
        tutorials=True
        print("--------------------------------------------------------------------------------------------------\n"
              "======================================== About this game =========================================\n"
              "--------------------------------------------------------------------------------------------------\n"
              "Following this message, you will be shown a collection of Connect Four Opponents. \n"
              "Some will be examples of AI, some will be more simplistic algorithms, some will be very hard to be beat, other very easy.\n"
              "I encourage you to play around and have fun trying to compete against the algorithms.\n"
              "The value to the right is how strong I believe the algorithm to be at Connect Four but you may have a different opinion. \n"
              "After each move you will be shown a score of how well the application deems you to be playing!"
              "\n the greater the value, the more it thinks its winning, "
              "\n the lower the value the more it thinks its losing\n"
              "--------------------------------------------------------------------------------------------------\n"
              "======================================== About this game =========================================\n"
              "--------------------------------------------------------------------------------------------------\n")
    display = pygame.display.set_mode(size_of_screen)
    start_game()