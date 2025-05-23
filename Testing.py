import time

from Minimax import min_max, iterative_deepening_minimax
from MonteCarloTreeSearch import mcts_parent_function, mcts_decisive_parent_function
from SetupAndConfig import columns
from MainApplication import create_board
import random
from HandleMoveFunction import handle_move
from EvaluationFunctions import win_check_position
from QLearning import action_selector,get_state
from DeepQNetwork import DQNAgent
from AIFunctions import (function_call_list, function_name_list, low_mcts_handler, low_minmax_handler, low_mcts_dec_handler,
                         low_iterative_deepening_minimax, high_minmax_handler, high_mcts_handler, hig_mcts_dec_handler,
                         high_iterative_deepening_minimax, random_handler, heuristic_handler, deep_q_network_handler, q_learning_handler)


player_names={}

for i in range(0,12):
    player_names[function_call_list[i]]=function_name_list[i]

player_A = q_learning_handler
player_B = low_minmax_handler
game_conclusion_tracker = [0, 0, 0]
player_A_time_taken=0
player_B_time_taken=0
player_A_move_count=0
player_B_move_count=0
print("working")
for i in range(1, 10):
    cols_valid_move = {i: 5 for i in range(columns)}
    board = create_board()
    no_winner = True
    current_player=1
    while cols_valid_move:
        if current_player == 1:
            start_time = time.time()
            move = player_A(board, cols_valid_move, current_player)
            end_time = time.time()
            player_A_time_taken+= end_time - start_time
            player_A_move_count+=1
        else:
            start_time = time.time()
            move = player_B(board, cols_valid_move, current_player)
            end_time = time.time()
            player_B_time_taken += end_time - start_time
            player_B_move_count += 1

        insertion_position = handle_move(move, cols_valid_move)
        board[insertion_position[0]][insertion_position[1]] = current_player

        if win_check_position(insertion_position[0],insertion_position[1], board, current_player):
            no_winner = False
            break
        current_player = 3 - current_player

    if no_winner:
        game_conclusion_tracker[0] += 1
    else:
        game_conclusion_tracker[current_player] += 1


with open("Documentation_&_notes/Testing_model_configs", "a") as log_file:
    log_file.write(f"Player A={player_names[player_A]}(P1) vs Player B={player_names[player_B]} (P2)\n")
    log_file.write(f"player A average move time ={player_A_time_taken / player_A_move_count} \n")
    log_file.write(f"player B average move time ={player_B_time_taken / player_B_move_count} \n")
    log_file.write(f"Draws | {player_names[player_A]} wins | {player_names[player_B]} wins = {game_conclusion_tracker} \n")
    log_file.write("-" * 30 + "\n")



game_conclusion_tracker = [0, 0, 0]
player_B_move_count=0
player_B_time_taken=0
player_A_move_count=0
player_A_time_taken=0

for i in range(1, 10):
    cols_valid_move = {i: 5 for i in range(columns)}
    board = create_board()
    no_winner = True
    current_player=1
    while cols_valid_move:
        if current_player == 1:
            start_time = time.time()
            move = player_B(board, cols_valid_move, current_player)
            end_time = time.time()
            player_B_time_taken += end_time - start_time
            player_B_move_count += 1
        else:
            start_time = time.time()
            move = player_A(board, cols_valid_move, current_player)
            end_time = time.time()
            player_A_time_taken += end_time - start_time
            player_A_move_count += 1

        insertion_position = handle_move(move, cols_valid_move)
        board[insertion_position[0]][insertion_position[1]] = current_player

        if win_check_position(insertion_position[0],insertion_position[1], board, current_player):
            no_winner = False
            break
        current_player = 3 - current_player


    if no_winner:
        game_conclusion_tracker[0] += 1
    else:
        game_conclusion_tracker[3-current_player] += 1


with open("Documentation_&_notes/Testing_model_configs", "a") as log_file:
    log_file.write(f"Player B={player_names[player_B]}(P1) vs Player A={player_names[player_A]} (P2)\n")
    log_file.write(f"player A average move time ={player_A_time_taken / player_A_move_count} \n")
    log_file.write(f"player B average move time ={player_B_time_taken / player_B_move_count} \n")
    log_file.write(f"Draws | {player_names[player_A]} wins | {player_names[player_B]} wins = {game_conclusion_tracker} \n")
    log_file.write("-" * 30 + "\n")


