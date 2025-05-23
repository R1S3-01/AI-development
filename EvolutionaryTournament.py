import math
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
from SetupAndConfig import rows,columns
from HandleMoveFunction import handle_move
from EvaluationFunctions import win_check_position
from AIFunctions import function_call_list,function_name_list
from AIFunctions import function_call_list_low,function_name_list_low

# This is the function used to calculate changes to Elo,
# its very simple and ensures a standardised rating metric for the models
def update_elo(p1, p2, result, k=50):
    p1_elo = elo_ratings[p1]
    p2_elo = elo_ratings[p2]

    p1_win_proba = (1.0 / (1 + math.pow(10, (p1_elo - p2_elo) / 400.0)))
    p2_win_proba = (1 - p1_win_proba)

    elo_ratings[p1] = max(0, p1_elo + k * (result - p1_win_proba))
    elo_ratings[p2] = max(0, p2_elo + k *  ((1 - result) - p2_win_proba))

# Here is where the necessary variables are initialised:
# population data is used to store each algorithm population as it changes
population_data=[]
population_pool=[]
player_quantities={}
player_names={}
current_player_quant=0
print(len(function_call_list_low))
#Set up the player population dict and player names dict, using the functions as the key
for i in range(0,len(function_call_list_low)):

    player_quantities[function_call_list_low[i]]=10
    player_names[function_call_list_low[i]]=function_call_list_low[i]
    #Set up the population pool each function starts with 10 population
    for j in range(0, 10):
        population_pool.append(function_call_list_low[i])


#Elos all start at 1000, stored in a dict with the function as key
elo_ratings = {fn: 1000.0 for fn in function_call_list_low}
competition_count=0

length_of_pool=len(population_pool)

#Where the competition is conducted

while competition_count<250 and current_player_quant<length_of_pool:
    game_conclusions_tracker = [0, 0, 0]
    player_A_index=randint(0, length_of_pool)
    player_B_index=randint(0, length_of_pool)

    #Ensure the competing algorithms are distinct
    while population_pool[player_A_index]==population_pool[player_B_index]:
        player_A_index = randint(0, length_of_pool)
        player_B_index = randint(0, length_of_pool)


    player_A=population_pool[player_A_index]
    player_B=population_pool[player_B_index]

    algo_A_start_elo=elo_ratings[player_A]
    algo_B_start_elo=elo_ratings[player_B]

    print(f"Algo A={player_names[player_A]}| Algo B = {player_names[player_B]}")
    print(f"Starting Elo A= {algo_A_start_elo}, Elo B= {algo_B_start_elo}")
    #Simulate the two games they play, each having one game as P1
    for i in range(2):

        cols_valid_move = {i: 5 for i in range(columns)}
        board = np.zeros((rows, columns), dtype=np.uint8)
        no_winner = True
        player_A_turn= True if i == 0 else False
        result=0.5
        total_moves=0
        current_player = 1
        while cols_valid_move:
            if player_A_turn == 1:
                move = player_A(board, cols_valid_move, current_player)
            else:
                move = player_B(board, cols_valid_move, current_player)
            total_moves+=1
            insertion_position = handle_move(move, cols_valid_move)
            board[insertion_position[0]][insertion_position[1]] = current_player

            if win_check_position(insertion_position[0],insertion_position[1], board, current_player):
                no_winner = False
                break
            current_player = 3 - current_player
            player_A_turn=not player_A_turn

        # Handle the outcome of each game
        if not no_winner:
            if player_A_turn:
                game_conclusions_tracker[1] += 1
                result = 1
            else:
                game_conclusions_tracker[2] += 1
                result = 0
        else:
            game_conclusions_tracker[0]+=1
            result=0.5
        #Update both algorithms elo each game
        update_elo(player_A, player_B, result, 50 - total_moves)

    competition_count+=1
    #If there was a unanimous victor then update the population pool and player quants accordingly
    if game_conclusions_tracker[1]==2:
        player_quantities[player_A] += 1
        player_quantities[player_B] -= 1
        population_pool[player_B_index] = player_A
        current_player_quant = player_quantities[player_A]
    elif game_conclusions_tracker[2]==2:
        player_quantities[player_B] += 1
        player_quantities[player_A] -= 1
        population_pool[player_A_index] = player_B
        current_player_quant = player_quantities[player_B]
    #Add the current player quantities to the population data list
    population_data.append(list(player_quantities.values()))

    print(game_conclusions_tracker)
    print(f"Elo rating of algo A= {elo_ratings[player_A]}, Elo rating of Algo B= {elo_ratings[player_B]}")
    print(f"Current game = {competition_count}")
    print("*"*30)

#establish the history of each algorithm's population
each_algo_pops = list(zip(*population_data))
plt.figure(figsize=(14, 7))
#Plot the data onto the EvolutionGraph
for algo_pop, label in zip(each_algo_pops, function_name_list_low):
    # establish if and where each algorithm dropped to 0 population, this is the cutoff point
    cutoff = next((i for i, val in enumerate(algo_pop) if val == 0), len(algo_pop))
    if cutoff < len(algo_pop):
        plt.plot(range(cutoff + 1), algo_pop[:cutoff + 1], label=label)
        plt.plot(cutoff, 0, 'x', color='black')
    else:
        plt.plot(range(len(algo_pop)), algo_pop, label=label)

#Creat the labels and legend and save the graph
plt.xlabel("Generations")
plt.ylabel("Population")
plt.title("Program's Population Over Generations")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.0)
plt.grid(True)
plt.tight_layout()
filename = "plots/EvolutionPlots/EvoLow.png"
plt.savefig(filename, dpi=500, format='png')

#Plot the elo bar chart graph
plt.figure(figsize=(12, 6))
names = [function_name_list_low[i] for i in range(len(function_name_list_low))]
ratings = [elo_ratings[fn] for fn in function_call_list_low]
plt.bar(names, ratings)
plt.xticks(rotation=45, ha='right')
plt.ylabel("Elo Rating")
plt.title("Final Elo Ratings")
plt.tight_layout()
filename = "plots/EvolutionPlots/EloLow.png"
plt.savefig(filename, dpi=300, format='png')






