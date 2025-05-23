import time
import matplotlib.pyplot as plt
import numpy as np
from SetupAndConfig import columns
from MainApplication import create_board
from AIFunctions import function_call_list,function_name_list
from HandleMoveFunction import handle_move
from EvaluationFunctions import win_check_position

def run_player_round(player_one_index,quantity_of_opponents,current_AI):
    #Setup lists for graph data
    opponents=[]
    wins=[]
    draws=[]
    losses=[]
    player_time = []
    opp_time = []
    #Establish what function player A is
    player_A = function_call_list[player_one_index]
    for player_two_index in range(quantity_of_opponents):
        print(f"current opponent= {function_name_list[player_two_index]}")
        player_B = function_call_list[player_two_index]
        #If player A==player B we do NOT run compete them
        if player_one_index == player_two_index:
            continue

        game_conclusion_tracker = [0, 0, 0]
        player_A_total_execution_time = 0
        player_B_total_execution_time = 0
        player_A_move_count = 0
        player_B_move_count = 0
        # Conduct the duals between player A and B, alternating who is P1 and P2
        for i in range(0, quantity_of_games):
            cols_valid_move = {i: 5 for i in range(columns)}
            board = create_board()
            no_winner = True
            current_player = 1
            player_A_turn = True if i % 2 == 0 else False
            while cols_valid_move:
                #Conduct the cur players move and track time taken
                if player_A_turn == 1:
                    start_time = time.time()
                    move = player_A(board, cols_valid_move, current_player)
                    end_time = time.time()
                    player_A_total_execution_time += end_time - start_time
                    player_A_move_count += 1
                else:
                    start_time = time.time()
                    move = player_B(board, cols_valid_move, current_player)
                    end_time = time.time()
                    player_B_total_execution_time += end_time - start_time
                    player_B_move_count += 1

                insertion_position = handle_move(move, cols_valid_move)
                board[insertion_position[0]][insertion_position[1]] = current_player

                if win_check_position(insertion_position[0],insertion_position[1], board, current_player):
                    no_winner = False
                    break
                current_player = 3 - current_player
                player_A_turn= not player_A_turn

            print("Current pos = ", i)
            #Track wins losses and draws
            if not no_winner:
                if player_A_turn:
                    game_conclusion_tracker[1] += 1
                else:
                    game_conclusion_tracker[2] += 1
            else:
                game_conclusion_tracker[0] += 1
        # Calcualte avg time taken for each player
        avg_player_time = player_A_total_execution_time / player_A_move_count
        avg_opp_time = player_B_total_execution_time / player_B_move_count

        print(f"here is the total number of outcomes for DRAW |{current_AI}|{function_name_list[player_two_index]} "
              f"\n {game_conclusion_tracker}")
        print(f"{current_AI} average time to move {avg_player_time}")
        print(f"{function_name_list[player_two_index]} average time to move {avg_opp_time}")

        #Track each players performance for graph data
        opponents.append(function_name_list[player_two_index])
        draws.append(game_conclusion_tracker[0])
        wins.append(game_conclusion_tracker[1])
        losses.append(game_conclusion_tracker[2])
        player_time.append(avg_player_time)
        opp_time.append(avg_opp_time)

        #log the data to a file for validation of graph etc
        with open("plots/plot_data", "a") as log_file:
            log_file.write(f"Current Player = {current_AI}| Opponent = {function_name_list[player_two_index]}\n")
            log_file.write(f"Draws= {game_conclusion_tracker[0]}| Wins = {game_conclusion_tracker[1]} | Losses = {game_conclusion_tracker[2]} \n")
            log_file.write(f"Player avg move time = {avg_player_time}| opponents avg move time = {avg_opp_time}\n")
            log_file.write("-" * 30 + "\n")

    #plot the performance graph for player A
    fig_bar, ax_bar = plt.subplots(figsize=(11, 6))
    y_positions = np.arange(len(opponents))
    bar_height = 0.2

    ax_bar.barh(y_positions, draws, height=bar_height, color='green', label='Draws')
    ax_bar.barh(y_positions + bar_height, losses, height=bar_height, color='orange', label='Losses')
    ax_bar.barh(y_positions + 2 * bar_height, wins, height=bar_height, color='blue', label='Wins')

    ax_bar.set_xticks(np.arange(0, 100 + 1, 5))
    ax_bar.set_yticks(y_positions + bar_height)
    ax_bar.set_yticklabels(opponents)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel('Number of Games')
    ax_bar.set_ylabel('Opponents')
    ax_bar.set_title(f'Performance of {current_AI} vs. All Other AIs')
    ax_bar.legend()
    plt.tight_layout()
    filename = f"plots/performancePlots/{current_AI}_performance.png"
    plt.savefig(filename, dpi=300, format='png')

    #plot the time consumption graph for player A
    fig_line, ax_line = plt.subplots(figsize=(9, 5))
    x = np.arange(len(opponents))
    ax_line.plot(x, player_time, marker='o', label='Player Avg Time per Move')
    ax_line.plot(x, opp_time, marker='o', label='Opponent Avg Time per Move')

    ax_line.set_xticks(x)
    ax_line.set_xticklabels(opponents, rotation=0)
    ax_line.set_xticklabels(opponents, rotation=45, ha='right')
    ax_line.set_xlabel('Opponents')
    ax_line.set_ylabel('Time ( Seconds )')
    ax_line.set_title(f'Time Consumption of {current_AI}')
    ax_line.legend()
    plt.tight_layout()

    filename = f"plots/timeMovePlots/{current_AI}_time_move.png"
    plt.savefig(filename, dpi=300, format='png')

#the procedure used to conduct each player's RR tournament
def algorithm_dual():
    quantity_of_opponents = len(function_call_list)
    for player_one_index in range(6, quantity_of_player):
        current_ai=function_name_list[player_one_index]
        run_player_round(player_one_index,quantity_of_opponents,current_ai)

#tournament setup variables
quantity_of_games=100
quantity_of_configs_to_test=12
quantity_of_player=12
algorithm_dual()
