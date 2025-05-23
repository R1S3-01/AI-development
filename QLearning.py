import time
from xmlrpc.client import MAXINT

import numpy as np
import random
import pickle
from SetupAndConfig import columns, rows, player_one_token, player_two_token
from EvaluationFunctions import win_check_position
from HandleMoveFunction import handle_move
from MonteCarloTreeSearch import mcts_parent_function
from Minimax import min_max



# We cant hash the np array board as it is mutable and we also need to hash the board with the current player ID,
# This is because the current board is viewd completely different from each players' perspective.
# So what is it we are actually returning, we pass in the board and the current player and we return,
# A tuple containing each row as a tuple and the current player so
# (((0,0,0,0,0,0,0),(0,0,0,0,0,0,0),...)),1)
def get_state(current_board,current_player):
    return tuple(map(tuple, current_board)), current_player

# The action selector function is what selects what move to make from a given state
def action_selector(cur_state, valid_moves, epsilon=0.0):
    # If the state isnt yet in the Q-table, we must first add it, this means were in an unexplored position.
    if cur_state not in Q_table:
        # Because te postion is unexplored, we assigned all actions following this position with a score of 0.
        Q_table[cur_state] = {move: 0 for move in valid_moves}

    # now we isolate and copy the dictionary section of the current state and get all its following available moves and their scores
    valid_q_values = {move: Q_table[cur_state][move] for move in Q_table[cur_state]}

    # Exploration vs. Exploitation e-greedy
    if random.uniform(0, 1) < epsilon:
        # Exploration random move
        return random.choice(list(valid_moves.keys()))
    else:
        # Exploitation best Q-value move
        return max(valid_q_values, key=valid_q_values.get)


# Notice that unlike in MCTS or Minimax, we only backpropagate to the immediately proceeding old state.
# This means that changes to further proceeding q-table entries take many iterations to update
def update_q_table(old_state, action, reward, new_state,valid_cols):
    if new_state not in Q_table:
        #if new state not in the q table, initialise it and store all its moves and their q values as 0
        Q_table[new_state] = {move: 0 for move in valid_cols}
    # Here we obtain the max future q value from the new state
    max_future_q = max(Q_table[new_state].values()) if Q_table[new_state] else 0
    # now we use the Bellman equation to effective backpropogate a portion of this max q value to the proceeding Q-Table entry.
    Q_table[old_state][action] += alpha * (reward + gamma * max_future_q - Q_table[old_state][action])




# So below here is the training code and im sure there is a way to smartly convert all these functions into a single function using
# conditional logic and loops etc, but this is unecessary complication to the code, instead the untidy approach has been opted for.
# Thus there are 4 functions with lots of repeated code, where the could be one, but we get increased clarity and efficiency in practice!

#Here as the name suggests, we are training Q learning against MCTS, with Q learning being player 1
def train_table_mcts_p1(quantity, epsilon_start=1.0, epsilon_end=0.10, epsilon_decay_episodes=10000, mcts_lim=0.01):
    # These are the values we want to track, in order to track training data.
    minimax_wins = 0
    q_wins = 0
    move_count = 0
    training_time = time.time()
    # we only need a single loop due to having the 4 separate functions
    for episode in range(quantity):
        # This is how we calculate the decaying epsilon value, this allows us to lower epsilon as training goes on.
        # This is very important and effecitve as we need a progessively lower epsilon value.
        progress = min(1.0, episode / float(epsilon_decay_episodes))
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * progress

        print("Episode =", episode, "| Epsilon =", round(epsilon, 4))

        # Reset and clear the board, cols valid moves and game move count
        board.fill(0)
        cols_valid_move = {i: rows - 1 for i in range(columns)}
        game_move_count=0
        # we can use a while true loop as there are checks throughout the loop which break out when needed.
        while True:
            # THIS IS THE MOST IMPORTANT THING IN RL. CONSISTENCY, we must make a decision on what the previous state is,
            # is it before our opponents turn? after their turn? it doesnt really matter what we choose as long as we stick to our choice
            previous_state = get_state(board, 1)
            # here we call the action selector, getting the Q-Learning models move
            action = action_selector(previous_state, cols_valid_move, epsilon)

            #carry out the move
            row, col = handle_move(action, cols_valid_move)
            board[row][col] = 1
            player_win = win_check_position(row, col, board, 1)
            game_move_count += 1
            #if the game is now terminally, we must update this in the Q network, despite it the opponent not being able to move fist.
            # This is vital as game conclusions are the most impactful on the Q-Table's Q values and will ensure correct training.
            if player_win:
                current_state = get_state(board, 1)
                # Notice wins are greater the quicker they are
                update_q_table(previous_state, action, 100-game_move_count, current_state, cols_valid_move)
                q_wins += 1
                break
            if not cols_valid_move:
                current_state = get_state(board, 1)
                #No reward for losses
                update_q_table(previous_state, action, 0, current_state, cols_valid_move)
                break

            # Now the opposing AI plays
            move = mcts_parent_function(board,cols_valid_move,2,mcts_lim)

            #carry out the move
            row, col = handle_move(move, cols_valid_move)
            board[row][col] = 2
            current_state = get_state(board, 1)
            opponent_win = win_check_position(row, col, board, 2)
            game_move_count += 1

            # Once again, if the opponent won we must now update the Q-Table despite Q-Learning not getting another turn.
            if opponent_win:
                minimax_wins += 1

                update_q_table(previous_state, action, -50+game_move_count, current_state, cols_valid_move)
                break
            if not cols_valid_move:
                update_q_table(previous_state, action, 0, current_state, cols_valid_move)
                break
            # If no win, there is no reward, we dont need to have a small negative, as we have the sacling terminal rewards.
            update_q_table(previous_state, action, 0, current_state, cols_valid_move)

        #update the total move count
        move_count+=game_move_count

        #Now for clarity and efficiency, we only document training and update the q-table file every 5000 episodes.
        if (episode + 1) % 5000 == 0:
            with open("Training logs/Q_Learning/Q_Network_mcts_p1", "a") as log_file:
                log_file.write(f" Q-Learning (P1) VS MCTS Roll={mcts_lim}(P2)\n")
                log_file.write(f"Progress Update [{episode + 1}/{quantity}]:\n")
                log_file.write(f"  1- Q-TABLE Wins: {q_wins} | MCTS Wins: {minimax_wins}\n")
                log_file.write(f"  2- Avg moves per game: {move_count / 5000:.2f}\n")
                log_file.write(f"  3- Time to complete batch: {time.time() - training_time:.4f}\n")
                log_file.write("-" * 30 + "\n")
                training_time = time.time()
            with open(q_table_filename, "wb") as f:
                pickle.dump(Q_table, f)
            print("Q-table saved successfully.")
            q_wins = 0
            minimax_wins = 0
            move_count = 0
#Only commenting the train function the rest are similar in nature only order and opposing AI which changes.
def train_table_mcts_p2(quantity, epsilon_start=1.0, epsilon_end=0.10, epsilon_decay_episodes=10000, mcts_lim=0.01):
    minimax_wins = 0
    q_wins = 0
    move_count = 0
    training_time = time.time()
    for episode in range(quantity):
        progress = min(1.0, episode / float(epsilon_decay_episodes))
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * progress
        print("Episode =", episode, "| Epsilon =", round(epsilon, 4))

        board.fill(0)
        cols_valid_move = {i: rows - 1 for i in range(columns)}
        game_move_count = 0

        move = mcts_parent_function(board,cols_valid_move,1,mcts_lim)
        row, col = handle_move(move, cols_valid_move)
        board[row][col] = 1
        game_move_count += 1
        while True:
            previous_state = get_state(board, 2)
            action = action_selector(previous_state, cols_valid_move, epsilon)
            row, col = handle_move(action, cols_valid_move)
            board[row][col] = 2
            game_move_count += 1
            player_win = win_check_position(row, col, board, 2)
            if player_win:
                current_state = get_state(board, 2)
                update_q_table(previous_state, action, 100 - game_move_count, current_state, cols_valid_move)
                q_wins += 1
                break
            if not cols_valid_move:
                current_state = get_state(board, 2)
                update_q_table(previous_state, action, 0, current_state, cols_valid_move)
                break


            move = mcts_parent_function(board,cols_valid_move,1,mcts_lim)
            row, col = handle_move(move, cols_valid_move)
            board[row][col] = 1
            game_move_count += 1
            opponent_win = win_check_position(row, col, board, 1)
            current_state = get_state(board, 2)
            if opponent_win:
                minimax_wins += 1
                update_q_table(previous_state, action, -50 + game_move_count, current_state, cols_valid_move)
                break

            if not cols_valid_move:
                update_q_table(previous_state, action, 0, current_state, cols_valid_move)
                break

            update_q_table(previous_state, action, 0, current_state, cols_valid_move)

        move_count += game_move_count


        if (episode + 1) % 5000 == 0:
            with open("Training logs/Q_Learning/Q_Network_mcts_p2", "a") as log_file:
                log_file.write(f"MCTS Roll={mcts_lim} (P1) VS Q-Learning (P2)\n")
                log_file.write(f"Progress Update [{episode + 1}/{quantity}]:\n")
                log_file.write(f"  1- Q-TABLE Wins: {q_wins} | MCTS Wins: {minimax_wins}\n")
                log_file.write(f"  2- Avg moves per game: {move_count / 5000:.2f}\n")
                log_file.write(f"  3- Time to complete batch: {time.time() - training_time:.4f}\n")
                log_file.write("-" * 30 + "\n")

            training_time = time.time()

            with open(q_table_filename, "wb") as f:
                pickle.dump(Q_table, f)
            print("Q-table saved successfully.")
            q_wins = 0
            minimax_wins = 0
            move_count = 0

#Only commenting the train function the rest are similar in nature only order and opposing AI which changes.
def train_table_mini_p1(quantity, epsilon_start=1.0, epsilon_end=0.10, epsilon_decay_episodes=10000,mini_depth=5):
    minimax_wins = 0
    q_wins = 0
    move_count = 0
    training_time=time.time()


    for episode in range(quantity):
        progress = min(1.0, episode / float(epsilon_decay_episodes))
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * progress
        print("Episode =", episode, "| Epsilon =", round(epsilon, 4))
        board.fill(0)
        cols_valid_move = {i: rows - 1 for i in range(columns)}
        game_move_count=0
        while True:

            previous_state = get_state(board, 1)
            action = action_selector(previous_state, cols_valid_move, epsilon)
            row, col = handle_move(action, cols_valid_move)
            board[row][col] = 1
            player_win = win_check_position(row, col, board, 1)
            game_move_count += 1

            if player_win:
                current_state = get_state(board, 1)
                update_q_table(previous_state, action, 100-game_move_count, current_state, cols_valid_move)
                q_wins += 1
                break
            if not cols_valid_move:
                current_state = get_state(board, 1)
                update_q_table(previous_state, action, 0, current_state, cols_valid_move)
                break

            move = min_max(board,cols_valid_move,(0,0),True,2,mini_depth,float('-inf'),float('inf'))[1]
            row, col = handle_move(move, cols_valid_move)
            board[row][col] = 2
            current_state = get_state(board, 1)
            opponent_win = win_check_position(row, col, board, 2)
            game_move_count += 1
            if opponent_win:
                minimax_wins += 1
                update_q_table(previous_state, action, -50+game_move_count, current_state, cols_valid_move)
                break
            if not cols_valid_move:
                update_q_table(previous_state, action, 0, current_state, cols_valid_move)
                break

            update_q_table(previous_state, action, 0, current_state, cols_valid_move)
        move_count+=game_move_count

        if (episode + 1) % 5000 == 0:
            with open("Training logs/Q_Learning/Q_Network_mini_p1", "a") as log_file:
                log_file.write(f"Q-Learning (P1) VS Minimax D={mini_depth}(P2)\n")
                log_file.write(f"Progress Update [{episode + 1}/{quantity}]:\n")
                log_file.write(f"  1- QT P1 Wins: {q_wins} | Mini P2 Wins: {minimax_wins}\n")
                log_file.write(f"  2- Avg moves per game: {move_count / 5000:.2f}\n")
                log_file.write(f"  3- Time to complete batch: {time.time()-training_time:.4f}\n")
                log_file.write("-" * 30 + "\n")

            training_time = time.time()

            with open(q_table_filename, "wb") as f:
                pickle.dump(Q_table, f)
            print("Q-table saved successfully.")
            q_wins = 0
            minimax_wins = 0
            move_count = 0

#Only commenting the train function the rest are similar in nature only order and opposing AI which changes.
def train_table_mini_p2(quantity, epsilon_start=1.0, epsilon_end=0.10, epsilon_decay_episodes=10000,mini_depth=5):
    minimax_wins = 0
    q_wins = 0
    move_count = 0
    training_time = time.time()
    for episode in range(quantity):
        progress = min(1.0, episode / float(epsilon_decay_episodes))
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * progress
        print("Episode =", episode, "| Epsilon =", round(epsilon, 4))

        board.fill(0)
        cols_valid_move = {i: rows - 1 for i in range(columns)}
        game_move_count = 0

        move = min_max(board, cols_valid_move, (0, 0), True, 1, mini_depth, float('-inf'), float('inf'))[1]
        row, col = handle_move(move, cols_valid_move)
        board[row][col] = 1
        game_move_count += 1
        while True:
            previous_state = get_state(board, 2)
            action = action_selector(previous_state, cols_valid_move, epsilon)
            row, col = handle_move(action, cols_valid_move)
            board[row][col] = 2
            game_move_count += 1
            player_win = win_check_position(row, col, board, 2)
            if player_win:
                current_state = get_state(board, 2)
                update_q_table(previous_state, action, 100 - game_move_count, current_state, cols_valid_move)
                q_wins += 1
                break
            if not cols_valid_move:
                current_state = get_state(board, 2)
                update_q_table(previous_state, action, 0, current_state, cols_valid_move)
                break


            move = min_max(board, cols_valid_move, (0, 0), True, 1, mini_depth, float('-inf'), float('inf'))[1]
            row, col = handle_move(move, cols_valid_move)
            board[row][col] = 1
            game_move_count += 1
            opponent_win = win_check_position(row, col, board, 1)
            current_state = get_state(board, 2)
            if opponent_win:
                minimax_wins += 1
                update_q_table(previous_state, action, -50 + game_move_count, current_state, cols_valid_move)
                break

            if not cols_valid_move:
                update_q_table(previous_state, action, 0, current_state, cols_valid_move)
                break

            update_q_table(previous_state, action, 0, current_state, cols_valid_move)

        move_count += game_move_count

        # Periodic logging and Q-table save
        if (episode + 1) % 5000 == 0:
            with open("Training logs/Q_Learning/Q_Network_mini_p2", "a") as log_file:
                log_file.write(f"Mini D={mini_depth}(P1) vs Q-Learning (P2)\n")
                log_file.write(f"Progress Update [{episode + 1}/{quantity}]:\n")
                log_file.write(f"  1- QT P2 Wins: {q_wins} | Mini P1 Wins: {minimax_wins}\n")
                log_file.write(f"  2- Avg moves per game: {move_count / 5000:.2f}\n")
                log_file.write(f"  3- Time to complete batch: {time.time() - training_time:.4f}\n")
                log_file.write("-" * 30 + "\n")

            training_time = time.time()

            with open(q_table_filename, "wb") as f:
                pickle.dump(Q_table, f)
            print("Q-table saved successfully.")
            q_wins = 0
            minimax_wins = 0
            move_count = 0





# This section is a small bit of setup
board = np.zeros((rows, columns), dtype=np.uint8)
q_table_filename = "Q_Table_Save_States/Q_Table_Official_Training.pkl"
# Initialise and assign these core variables with values:
alpha = 0.4  # Learning rate
gamma = 0.98  # Discount factor
episodes = 0  # Number of training games
try:
    with open(q_table_filename, "rb") as f:
        Q_table = pickle.load(f)
    print("Q-table loaded successfully!")
except FileNotFoundError:
    print("Q-table not found. Initializing a new one.")
    Q_table = {
        (tuple(map(tuple, board)), player_one_token): {move: 0 for move in range(columns)},
        (tuple(map(tuple, board)), player_two_token): {move: 0 for move in range(columns)}
    }

#In the main section is where training is coordinated, below i have left commented all the training i conducted which will support the data in the logs.
if __name__ == "__main__":
    # train batch one
    # train_table_mini_p1(50000, 1, 0.17, 45000,6)
    # train_table_mcts_p1(50000, 1, 0.17, 45000,0.005)
    # train_table_mini_p2(50000, 1, 0.17, 45000, 6)
    # train_table_mcts_p2(50000, 1, 0.17, 45000,0.005)
    #
    # train_table_mini_p1(50000, 1, 0.17, 45000,7)
    # train_table_mcts_p1(50000, 1, 0.17, 45000,0.01)
    # train_table_mini_p2(50000, 1, 0.17, 45000, 7)
    # train_table_mcts_p2(50000, 1, 0.17, 45000,0.01)

    # train_table_mini_p1(75000, 0.75, 0.20, 45000, 8)
    # train_table_mcts_p1(75000, 0.75, 0.20, 45000, 0.03)
    # train_table_mini_p2(75000, 0.75, 0.20, 45000, 8)
    # train_table_mcts_p2(75000, 0.75, 0.20, 45000, 0.03)

    # train_table_mini_p1(100000, 0.65, 0.15, 65000, 10)
    # train_table_mini_p1(50000, 0.45, 0.0,45000 , 10)
    # train_table_mini_p2(100000, 0.65, 0.15, 65000, 10)
    # train_table_mini_p2(50000, 0.45, 0.0,45000 , 10)

    # train_table_mcts_p1(100000, 0.65, 0.15, 60000, 0.03)
    # train_table_mcts_p2(100000, 0.65, 0.15, 65000, 0.03)
    # train_table_mcts_p1(50000, 0.45, 0.0, 60000, 0.03)
    # train_table_mcts_p2(50000, 0.45, 0.0, 60000, 0.03)

    #Finale training
    #The desire with this training is that we can improve MCTS
    # train_table_mcts_p1(35000, 0.50, 0.15, 25000, 0.03)
    # train_table_mcts_p2(35000, 0.50, 0.15, 25000, 0.03)
    # # This last 'training' block is where the finale results of the model can be seen,
    # # this will be important to baseline their performance and gain a glimpse of what to expect in future competitions!
    # train_table_mini_p1(10000, 0.25, 0.00, 5000, 9)
    # train_table_mini_p2(10000, 0.25, 0.00, 5000, 9)
    # train_table_mcts_p1(10000, 0.25, 0.00, 5000, 0.03)
    # train_table_mcts_p2(10000, 0.25, 0.00, 5000, 0.03)

    # No more training will be performed past this point as the Q-Table file is already far too large....
    # But running something like this doesnt increase the file size as there are no new states seen due to the nature of minimax !
    # I will leave this last bit of code uncommented so that there is no erros showing in my IDE and running it is useless
    train_table_mini_p1(5000, 0.001, 0.00, 1, 9)
    train_table_mini_p2(5000, 0.001, 0.00, 1, 9)





