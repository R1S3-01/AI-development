import os
import pickle
import random
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from SetupAndConfig import columns, rows
from HandleMoveFunction import handle_move
from EvaluationFunctions import win_check_position,score_board_simplified
from MonteCarloTreeSearch import mcts_parent_function
from Minimax import min_max


# This class is where the structure of the Deep Q Network is defined,
# As can be seen, there are 3 convolutional layers -> 3 fully connected linear layers.
# BatchNorm is used to stabilise the model
# ReLu is used to control what neurons are activated in the network, introducing non-linearity
class DQNConvModel(nn.Module):
    def __init__(self):
        # this line sets up everything that PyTorch needs before i implement custom layers
        super(DQNConvModel, self).__init__()

        # The convolutional layer is used for feature extraction, breaking the board down as it would an image and understanding what is being shown
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=4, stride=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

        )
        # The sequential layer is used for Q-Value estimation, this is the part which makes decisions on where is best to move.
        self.fc_layers = nn.Sequential(
            nn.Linear(3072 + 1, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, columns)
        )

    # The forward function is my models custom logic on how to translate inputted game boards into q values!
    def forward(self, board, current_player):
        x = self.conv_layers(board)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, current_player], dim=1)
        q_values = self.fc_layers(x)
        return q_values

# A fundamental preprocessing function used to prepare data for the neural net
def state_to_tensor(board_state, current_player):
    # cur_player_matrix is a binary matrix which only fills cells which contain a token== current player
    cur_player_matrix = (board_state == current_player).astype(np.float32)
    # opp_player_matrix is a binary matrix which only fills cells which contain a token== opposing player
    opp_player_matrix = ((board_state != 0) & (board_state != current_player)).astype(np.float32)
    # combines both planes.
    combo_player_matrix = np.stack([cur_player_matrix, opp_player_matrix], axis=0)
    #converts the board to a tensor board so the shape becomes [1, 2, rows, columns]
    board_tensor = torch.tensor(combo_player_matrix, dtype=torch.float32).unsqueeze(0)
    #Convert the current player int into a tensor board so it can be processed.
    cur_player_tensor = torch.tensor([[float(current_player)]], dtype=torch.float32)
    return board_tensor, cur_player_tensor


# This is the actual DQNAgent class, essentially here is where we control how the network functions and learns.
class DQNAgent:
    # All these values are important and have been tweaked and adjusted over many iterations of testing.
    def __init__(self,gamma=0.99,lr=0.0001,batch_size=256,replay_size=250000,target_update_freq=1000):
        self.rows = rows
        # How much future q values are valued
        self.gamma = gamma
        #how many experiences from memory are used at one time during training
        self.batch_size = batch_size
        #How frequently the target network updates
        self.target_update_freq = target_update_freq
        # The policy net is the network model which is trained and predicts what moves to make
        self.policy_net = DQNConvModel()
        # The target network is used to stabilise the Q targets
        self.target_net = DQNConvModel()
        #load state dict is used to load the policy net into the target net making them initally equal .
        self.target_net.load_state_dict(self.policy_net.state_dict())
        #eval freezes dropout and batch normalisation
        self.target_net.eval()
        # the optimiser applies gradient updates to the policy network we use.
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr, weight_decay=0.00001)
        #MSE is used to compare predicted Q values with the seperate target Q values.
        self.loss_fn = nn.MSELoss()
        # here we are establishing the replay buffer with a max size of replay size
        self.replay_buffer = deque(maxlen=replay_size)
        #steps done is used to track when the target network should be updated
        self.steps_done = 0

    #Select action is what is used during training to get the move from the DQN.
    def select_action(self, state, valid_actions, current_player, epsilon):
        # We see the usage of exploration depending on epsilon, as seen in the Q-Learning model.
        if random.random() < epsilon:
            return random.choice(list(valid_actions.keys()))
        else:
            #conver the board and player varaibles into appropriate tensor supported data
            board_tensor,cur_player_tensor= state_to_tensor(state, current_player)
            #Get the predicted values for each move
            with torch.no_grad():
                q_values = self.policy_net(board_tensor, cur_player_tensor)

            #convert the values to a np array
            q_values_np = q_values.detach().cpu().numpy().flatten()

            # Invalidate illegal actions
            for col in range(columns):
                if col not in valid_actions:
                    q_values_np[col] = -float('inf')

            #return best action
            return int(np.argmax(q_values_np))

    #Select best action is what is used when not training to get the DQN to pick the best move it can.
    # Largely the same as the prior function but notice the policy_net.eval and no exploration.
    def select_best_action(self, state, valid_actions, current_player):
        self.policy_net.eval()
        board_tensor, cur_player_tensor = state_to_tensor(state, current_player)

        with torch.no_grad():
            q_values = self.policy_net(board_tensor, cur_player_tensor)

        q_values_np = q_values.detach().cpu().numpy().flatten()

        # Invalidate illegal actions
        for col in range(columns):
            if col not in valid_actions:
                q_values_np[col] = -float('inf')

        return int(np.argmax(q_values_np))

    # The remember state function is used to add a new experience into the replay buffer.
    # An experience is a collection of data, being:
    # it is the old state, the action taken, the reward of the action, the new state created if the state is terminal and the current player
    def remember_state(self, state, action, reward, next_state, done, current_player):
        self.replay_buffer.append((state, action, reward, next_state, done, current_player))

    # The update last state function is used when we need to update the last saved experience.
    # This happens in order to keep continuity in the network, imagine the DQN
    # moves and then the opponent moves and wins, we cant store the opponents move in memory, instead we update the last state as a lost and terminal position.
    def update_last_state(self, new_reward, new_done):
        s, a, old_reward, next_s, old_done, p = self.replay_buffer[-1]
        self.replay_buffer[-1] = (s, a, new_reward, next_s, new_done, p)

    # The train back function is the largest function in the DQNAgent and perhaps the most important
    def train_batch(self):
        # No training is conducted if the replay buffer hasnt yet hit the batch size minimum
        if len(self.replay_buffer) < self.batch_size:
            return
        #increment steps done to correctly enforce target_update_freq
        self.steps_done += 1

        batch = random.sample(self.replay_buffer, self.batch_size)

        # Prepare batches
        board_batch = []
        cur_player_batch = []
        action_batch = []
        reward_batch = []
        next_board_batch = []
        next_cur_player_batch = []
        done_batch = []

        # Here we are going through each experience in the batch and converting the
        # data into tensor data and appending it to batch lists
        for s, a, r, s_next, d, p in batch:
            board_tensor, cur_player_tensor = state_to_tensor(s, p)
            board_batch.append(board_tensor)
            cur_player_batch.append(cur_player_tensor)
            action_batch.append(a)
            reward_batch.append(r)

            next_board_tensor, next_cur_player_tensor = state_to_tensor(s_next, p)
            next_board_batch.append(next_board_tensor)
            next_cur_player_batch.append(next_cur_player_tensor)
            done_batch.append(d)

        # Here we concatenate all the data in the lists into single batches
        board_batch_tensor = torch.cat(board_batch, dim=0)
        cur_player_batch_tensor = torch.cat(cur_player_batch, dim=0)
        next_board_batch_tensor = torch.cat(next_board_batch, dim=0)
        next_cp_batch_tensor = torch.cat(next_cur_player_batch, dim=0)

        # Convert the still unconverted data into the correct tensors and add a second column so they match input shapes
        action_batch_tensor = torch.LongTensor(action_batch).unsqueeze(1)
        reward_batch_tensor = torch.FloatTensor(reward_batch).unsqueeze(1)
        done_batch_tensor = torch.BoolTensor(done_batch).unsqueeze(1)


        #now we get predicted Q values from the actions taken in the action batch
        q_values = self.policy_net(board_batch_tensor, cur_player_batch_tensor).gather(1, action_batch_tensor)

        # Now utilising the target network, we estimate the best Q-Values from the next state
        with torch.no_grad():
            max_next_q = self.target_net(next_board_batch_tensor, next_cp_batch_tensor).max(dim=1, keepdim=True)[0]
            q_target = reward_batch_tensor + self.gamma * max_next_q * (~done_batch_tensor)

        #measure how far our predicted Q values where from the target Q values.
        loss = self.loss_fn(q_values, q_target)


        self.optimizer.zero_grad() # remove old gradients
        loss.backward() #Calculate how much impact each weight in model had on the loss
        self.optimizer.step() #This is where the model actually learns and we update the model weights accordingly.

        # Update the target network if correct time
        if self.steps_done % self.target_update_freq == 0:
            print("We updated the target net")
            with torch.no_grad():
                # We do not just make the target network the policy network, instead we make it 1% closer to the policy network !
                for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                    target_param.data.copy_(0.995 * target_param.data + 0.01 * policy_param.data)


    def save(self, path_to_model,path_to_buffer):
        with open(path_to_buffer, "wb") as f:
            pickle.dump(self.replay_buffer, f)

        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path_to_model)
        print("Model & replay buffer saved!")

    def load(self, path_to_model,path_to_buffer="Not Needed"):
        if not os.path.exists(path_to_model):
            print("No saved model found. Starting new training.")
            return

        checkpoint = torch.load(path_to_model)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if path_to_buffer!="Not Needed":
            if os.path.exists(path_to_buffer):
                with open(path_to_buffer, "rb") as f:
                    self.replay_buffer = pickle.load(f)
                print(f"Replay buffer loaded! Stored experiences: {len(self.replay_buffer)}")
            else:
                print("No replay buffer found at that path location, creating new buffer")
        else:
            print("Saving memory - No buffer loaded as not needed for optimal move selection")
        print("Model loaded successfully from", path_to_model)


def train_dqn_mcts_player1(dqn_agent,num_episodes=50000,epsilon_start=0.99,epsilon_end=0.1,epsilon_decay_episodes=12500,mcts_time=0.020):

    dqn_wins = 0
    mcts_wins = 0
    total_moves = 0
    dqn_player = 1
    mcts_player = 2
    start_time = time.time()
    for episode in range(num_episodes):

        board = np.zeros((rows, columns), dtype=np.uint8)
        valid_cols = {c: 5 for c in range(columns)}
        done = False
        move_count = 0
        progress = min(1.0, episode / float(epsilon_decay_episodes))
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * progress
        current_player = 1
        previous_state=board

        while not done and bool(valid_cols):

            action_col = dqn_agent.select_action(previous_state, valid_cols, dqn_player, epsilon)
            insertion_pos = handle_move(action_col, valid_cols)
            board[insertion_pos[0]][insertion_pos[1]] = dqn_player

            if win_check_position(insertion_pos[0],insertion_pos[1], board, dqn_player):
                move_count+=1
                dqn_wins += 1
                done = True
                reward = 1000 - (move_count * 3)
                dqn_agent.remember_state(previous_state, action_col, reward, board.copy(), done, dqn_player)
                break

            if not bool(valid_cols):
                dqn_agent.remember_state(previous_state, action_col, 50, board.copy(), True, dqn_player)
                break

            reward = score_board_simplified(board, dqn_player)

            dqn_agent.remember_state(previous_state, action_col, reward, board.copy(), False, dqn_player)
            mcts_col = mcts_parent_function(board, valid_cols, mcts_player, mcts_time)
            insertion_pos = handle_move(mcts_col, valid_cols)
            board[insertion_pos[0]][insertion_pos[1]] = mcts_player
            move_count += 2

            if win_check_position(insertion_pos[0],insertion_pos[1], board, mcts_player):
                mcts_wins += 1
                dqn_agent.update_last_state(new_reward=-750 + (move_count * 3), new_done=True)
                break

            previous_state = board.copy()
            current_player = 3 - current_player

        dqn_agent.train_batch()
        total_moves += move_count

        if (episode + 1) % 5000  == 0:
            with open(dqn_vs_mcts_p1_training_logs, "a") as log_file:
                log_file.write(f"DQN P1 | MCTS {mcts_time} P2\n")
                log_file.write(f"Progress Update [{episode + 1}/{num_episodes}]:\n")
                log_file.write(f"  1- DQN Wins: {dqn_wins} | MCTS Wins: {mcts_wins}\n")
                log_file.write(f"  2- Avg moves per game: {total_moves / 5000:.2f}\n")
                log_file.write(f"  3- Batch took: {time.time() - start_time}\n")
                log_file.write("=" * 30 + "\n")
            start_time=time.time()
            dqn_agent.save(ai_trained_model_path, ai_trained_buffer_path)
            dqn_wins=0
            mcts_wins=0
            total_moves=0

        print(f"Episode {episode} | ended in {move_count} moves | epsilon ={epsilon}")

    dqn_agent.save(ai_trained_model_path, ai_trained_buffer_path)
    print("DQN model saved as dqn_ai_trained_model.pth")


def train_dqn_mcts_player2(dqn_agent, num_episodes=50000, epsilon_start=0.99, epsilon_end=0.1, epsilon_decay_episodes=12500, mcts_time=0.020):

    dqn_wins = 0
    mcts_wins = 0
    total_moves = 0
    dqn_player = 2
    mcts_player = 1
    start_time=time.time()
    for episode in range(num_episodes):

        board = np.zeros((rows, columns), dtype=np.uint8)
        valid_cols = {c: 5 for c in range(columns)}
        done = False
        move_count = 0
        progress = min(1.0, episode / float(epsilon_decay_episodes))
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * progress
        current_player = 1
        mcts_col = mcts_parent_function(board, valid_cols, mcts_player, mcts_time)
        insertion_pos = handle_move(mcts_col, valid_cols)
        board[insertion_pos[0]][insertion_pos[1]] = mcts_player

        while not done and bool(valid_cols):

            previous_state = board.copy()
            action_col = dqn_agent.select_action(previous_state, valid_cols, dqn_player, epsilon)
            insertion_pos = handle_move(action_col, valid_cols)
            board[insertion_pos[0]][insertion_pos[1]] = dqn_player
            move_count += 2

            if win_check_position(insertion_pos[0],insertion_pos[1], board, dqn_player):
                # DQN won
                dqn_wins += 1
                done = True
                reward = 1000 - (move_count * 3)
                dqn_agent.remember_state(previous_state, action_col, reward, board.copy(), done, dqn_player)
                break

            reward = score_board_simplified(board, dqn_player)

            if not bool(valid_cols):
                dqn_agent.remember_state(previous_state, action_col, 50, board.copy(), True, dqn_player)
                break
            dqn_agent.remember_state(previous_state, action_col, reward, board.copy(), done, dqn_player)


            mcts_col = mcts_parent_function(board, valid_cols, mcts_player, mcts_time)
            insertion_pos = handle_move(mcts_col, valid_cols)
            board[insertion_pos[0]][insertion_pos[1]] = mcts_player

            if win_check_position(insertion_pos[0],insertion_pos[1], board, mcts_player):
                move_count += 1
                mcts_wins += 1
                dqn_agent.update_last_state(new_reward=-750 + (move_count * 3), new_done=True)
                break

            current_player = 3 - current_player

        dqn_agent.train_batch()
        total_moves += move_count

        if (episode + 1) % 5000 == 0:
            with open(dqn_vs_mcts_p2_training_logs, "a") as log_file:
                log_file.write(f"MCTS {mcts_time} P1 | DQN P2\n")
                log_file.write(f"Progress Update [{episode + 1}/{num_episodes}]:\n")
                log_file.write(f"  1- MCTS Wins: {mcts_wins}|DQN Wins: {dqn_wins}  \n")
                log_file.write(f"  2- Avg moves per game: {total_moves / 5000:.2f}\n")
                log_file.write(f"  3- Batch took: {time.time()-start_time}\n")
                log_file.write("-" * 30 + "\n")
            start_time=time.time()
            dqn_agent.save(ai_trained_model_path, ai_trained_buffer_path)
            dqn_wins = 0
            mcts_wins = 0
            total_moves = 0

        print(f"Episode {episode} | ended in {move_count} moves| epsilon ={epsilon}")

    dqn_agent.save(ai_trained_model_path, ai_trained_buffer_path)
    print("DQN model saved as dqn_ai_trained_model.pth")



def train_dqn_minimax_player1(dqn_agent,num_episodes=50000,epsilon_start=0.99,epsilon_end=0.1,epsilon_decay_episodes=12500,depth=20):

    dqn_wins = 0
    mini_wins = 0
    total_moves = 0
    dqn_player = 1
    mcts_player = 2
    start_time = time.time()
    for episode in range(num_episodes):

        board = np.zeros((rows, columns), dtype=np.uint8)
        valid_cols = {c: 5 for c in range(columns)}
        done = False
        move_count = 0
        progress = min(1.0, episode / float(epsilon_decay_episodes))
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * progress
        current_player = 1
        previous_state=board

        while not done and bool(valid_cols):

            action_col = dqn_agent.select_action(previous_state, valid_cols, dqn_player, epsilon)
            insertion_pos = handle_move(action_col, valid_cols)
            board[insertion_pos[0]][insertion_pos[1]] = dqn_player

            if win_check_position(insertion_pos[0],insertion_pos[1], board, dqn_player):
                move_count+=1
                dqn_wins += 1
                done = True
                reward = 1000 - (move_count * 3)
                dqn_agent.remember_state(previous_state, action_col, reward, board.copy(), done, dqn_player)
                break

            if not bool(valid_cols):
                dqn_agent.remember_state(previous_state, action_col, 50, board.copy(), True, dqn_player)
                break
            reward = score_board_simplified(board, dqn_player)

            dqn_agent.remember_state(previous_state, action_col, reward, board.copy(), False, dqn_player)
            minimax_col = min_max(board,valid_cols,(0,0),True,2,depth,float('-inf'), float('inf'))[1]
            insertion_pos = handle_move(minimax_col, valid_cols)
            board[insertion_pos[0]][insertion_pos[1]] = mcts_player
            move_count += 2

            if win_check_position(insertion_pos[0],insertion_pos[1], board, mcts_player):
                mini_wins += 1
                dqn_agent.update_last_state(new_reward=-750 + (move_count * 3), new_done=True)
                break

            previous_state = board.copy()
            current_player = 3 - current_player

        dqn_agent.train_batch()
        total_moves += move_count

        if (episode + 1) % 5000  == 0:
            with open(dqn_vs_mini_p1_training_logs, "a") as log_file:
                log_file.write(f"DQN P1  | MINI {depth} P2\n")
                log_file.write(f"Progress Update [{episode + 1}/{num_episodes}]:\n")
                log_file.write(f"  1- DQN Wins: {dqn_wins} | MINI Wins: {mini_wins}\n")
                log_file.write(f"  2- Avg moves per game: {total_moves / 5000:.2f}\n")
                log_file.write(f"  3- Batch took: {time.time() - start_time}\n")
                log_file.write("=" * 30 + "\n")
            start_time=time.time()
            dqn_agent.save(ai_trained_model_path, ai_trained_buffer_path)
            dqn_wins=0
            mini_wins=0
            total_moves=0

        print(f"Episode {episode} | ended in {move_count} moves| epsilon ={epsilon}")

    dqn_agent.save(ai_trained_model_path, ai_trained_buffer_path)
    print("DQN model saved as dqn_ai_trained_model.pth")


def train_dqn_minimax_player2(agent_ai_trained, num_episodes=50000, epsilon_start=0.99, epsilon_end=0.1, epsilon_decay_episodes=12500, depth=20):

    dqn_wins = 0
    mini_wins = 0
    total_moves = 0
    dqn_player = 2
    mcts_player = 1
    start_time=time.time()
    for episode in range(num_episodes):

        board = np.zeros((rows, columns), dtype=np.uint8)
        valid_cols = {c: 5 for c in range(columns)}
        done = False
        move_count = 0
        progress = min(1.0, episode / float(epsilon_decay_episodes))
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * progress
        current_player = 1
        minimax_col = min_max(board,valid_cols,(0,0),True,1,depth,float('-inf'), float('inf'))[1]
        insertion_pos = handle_move(minimax_col, valid_cols)
        board[insertion_pos[0]][insertion_pos[1]] = mcts_player

        while not done and bool(valid_cols):

            previous_state = board.copy()
            action_col = agent_ai_trained.select_action(previous_state, valid_cols, dqn_player, epsilon)
            insertion_pos = handle_move(action_col, valid_cols)
            board[insertion_pos[0]][insertion_pos[1]] = dqn_player
            move_count += 2

            if win_check_position(insertion_pos[0],insertion_pos[1], board, dqn_player):
                # DQN won
                dqn_wins += 1
                done = True
                reward = 1000 - (move_count * 3)
                agent_ai_trained.remember_state(previous_state, action_col, reward, board.copy(), done, dqn_player)
                break

            reward = score_board_simplified(board, dqn_player)

            if not bool(valid_cols):
                agent_ai_trained.remember_state(previous_state, action_col, 50, board.copy(), True, dqn_player)
                break
            agent_ai_trained.remember_state(previous_state, action_col, reward, board.copy(), done, dqn_player)


            minimax_col = min_max(board,valid_cols,(0,0),True,1,depth,float('-inf'), float('inf'))[1]
            insertion_pos = handle_move(minimax_col, valid_cols)
            board[insertion_pos[0]][insertion_pos[1]] = mcts_player

            if win_check_position(insertion_pos[0],insertion_pos[1], board, mcts_player):
                move_count += 1
                mini_wins += 1
                agent_ai_trained.update_last_state(new_reward=-750 + (move_count * 3), new_done=True)
                break

            current_player = 3 - current_player

        agent_ai_trained.train_batch()
        total_moves += move_count

        if (episode + 1) % 5000 == 0:
            with open(dqn_vs_mini_p2_training_logs, "a") as log_file:
                log_file.write(f"Mini {depth} P1 | DQN P2 \n")
                log_file.write(f"Progress Update [{episode + 1}/{num_episodes}]:\n")
                log_file.write(f"  1- Mini Wins: {mini_wins} | DQN Wins: {dqn_wins}\n")
                log_file.write(f"  2- Avg moves per game: {total_moves / 5000:.2f}\n")
                log_file.write(f"  3- Batch took: {time.time()-start_time}\n")
                log_file.write("-" * 30 + "\n")
            start_time=time.time()
            agent_ai_trained.save(ai_trained_model_path, ai_trained_buffer_path)
            dqn_wins = 0
            mini_wins = 0
            total_moves = 0

        print(f"Episode {episode} | ended in {move_count} moves | epsilon ={epsilon}")

    agent_ai_trained.save(ai_trained_model_path, ai_trained_buffer_path)
    print("DQN model saved as dqn_ai_trained_model.pth")

def train_dqn_self_play(agent_self_trained_one,one_model_path,one_buffer_path, agent_self_trained_two,two_model_path,two_buffer_path, num_episodes=50000, epsilon_start=0.99, epsilon_end=0.1, epsilon_decay_episodes=12500):

    total_moves = 0
    start_time=time.time()
    for episode in range(num_episodes):

        board = np.zeros((rows, columns), dtype=np.uint8)
        valid_cols = {c: 5 for c in range(columns)}
        done = False
        move_count = 0
        progress = min(1.0, episode / float(epsilon_decay_episodes))
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * progress
        current_player = 1


        while not done and bool(valid_cols):

            previous_state = board.copy()
            action_col = agent_self_trained_one.select_action(previous_state, valid_cols, 1, epsilon)
            insertion_pos = handle_move(action_col, valid_cols)
            board[insertion_pos[0]][insertion_pos[1]] = 1
            move_count += 2

            if win_check_position(insertion_pos[0],insertion_pos[1], board, 1):
                # DQN won
                reward = 1000 - (move_count * 3)
                agent_self_trained_one.remember_state(previous_state, action_col, reward, board.copy(), True, 1)
                break



            if not bool(valid_cols):
                agent_self_trained_one.remember_state(previous_state, action_col, -1, board.copy(), True, 1)
                break


            agent_self_trained_one.remember_state(previous_state, action_col, -0.5, board.copy(), False, 1)
            previous_state = board.copy()

            action_col = agent_self_trained_two.select_action(previous_state, valid_cols, 2, epsilon)
            insertion_pos = handle_move(action_col, valid_cols)
            board[insertion_pos[0]][insertion_pos[1]] = 2

            if win_check_position(insertion_pos[0],insertion_pos[1], board, 2):
                move_count += 1
                reward = 1000 - (move_count * 3)

                agent_self_trained_two.remember_state(previous_state, action_col, reward, board.copy(), True, 2)
                break

            if not bool(valid_cols):
                agent_self_trained_two.remember_state(previous_state, action_col, 2, board.copy(), True, 2)
                break


            agent_self_trained_two.remember_state(previous_state, action_col, 1, board.copy(), False, 2)
            current_player = 3 - current_player

        agent_self_trained_two.train_batch()
        agent_self_trained_one.train_batch()
        total_moves += move_count

        if episode%500==0:
            print(episode)

    with open(dqn_agent_one_vs_two_training_logs, "a") as log_file:
        log_file.write(f"DQN P1 | DQN P2 \n")
        log_file.write(f"  Avg moves per game: {total_moves / 5000:.2f}\n")
        log_file.write(f"  Batch took: {time.time() - start_time}\n")
        log_file.write("-" * 30 + "\n")

    agent_self_trained_one.save(one_model_path, one_buffer_path)
    agent_self_trained_two.save(two_model_path, two_buffer_path)


    agent_self_trained_one.save(one_model_path,one_buffer_path)
    print("DQN two model saved as dqn_agent_self_trained_one.pth")
    agent_self_trained_two.save(two_model_path,two_buffer_path)
    print("DQN two model saved as dqn_agent_self_trained_two.pth")


def test_dqn(dqn_agent,games,file_path):
    dqn_wins=0
    mcts_wins=0
    draws=0
    for episode in range(games):
        board = np.zeros((rows, columns), dtype=np.uint8)
        valid_cols = {c: 5 for c in range(columns)}
        player_win=False
        while valid_cols and not player_win:
            action_col=dqn_agent.select_action(board, valid_cols, 1, 0)
            insertion_pos = handle_move(action_col, valid_cols)
            board[insertion_pos[0]][insertion_pos[1]] = 1
            if win_check_position(insertion_pos[0],insertion_pos[1], board, 1):
                dqn_wins+=1
                player_win=True
                break

            if not valid_cols:
                break

            action_col = min_max(board,valid_cols,(0,0),True,2,8,float('-inf'), float('inf'))[1]
            insertion_pos= handle_move(action_col, valid_cols)
            board[insertion_pos[0]][insertion_pos[1]] = 2
            if win_check_position(insertion_pos[0],insertion_pos[1],board,2):
                mcts_wins+=1
                player_win=True
                break

        if not player_win:
            draws+=1

    with open(file_path, "a") as log_file:
        log_file.write(f"DQN P1 | MCTS P2 \n")
        log_file.write(f" Wins: {dqn_wins} | Losses: {mcts_wins}| Draws: {draws}\n")
        log_file.write("-" * 30 + "\n")

    dqn_wins=0
    mcts_wins=0
    draws=0
    for episode in range(games):
        board = np.zeros((rows, columns), dtype=np.uint8)
        valid_cols = {c: 5 for c in range(columns)}
        player_win = False
        while valid_cols and not player_win:
            action_col = min_max(board,valid_cols,(0,0),True,2,8,float('-inf'), float('inf'))[1]
            insertion_pos = handle_move(action_col, valid_cols)
            board[insertion_pos[0]][insertion_pos[1]] = 2
            if win_check_position(insertion_pos[0],insertion_pos[1], board, 2):
                mcts_wins += 1
                player_win = True
                break

            if not valid_cols:
                break

            action_col=dqn_agent.select_action(board, valid_cols, 2, 0)
            insertion_pos = handle_move(action_col, valid_cols)
            board[insertion_pos[0]][insertion_pos[1]] = 1
            if win_check_position(insertion_pos[0],insertion_pos[1], board, 1):
                dqn_wins += 1
                player_win = True
                break
        if not player_win:
            draws += 1

    with open(file_path, "a") as log_file:
        log_file.write(f"MCTS P1 | DQN P2 | \n")
        log_file.write(f" Wins: {dqn_wins} | Losses: {mcts_wins}| Draws: {draws}\n")
        log_file.write("-" * 30 + "\n")



#The DQN has lots of paths, the varaible names clear up what each path leads too
dqn_vs_mini_p1_training_logs= "Training logs/DQN/AI/DQN_VS_MINIMAX_3_3"
dqn_vs_mini_p2_training_logs= "Training logs/DQN/AI/MINIMAX_VS_DQN_3_3"
dqn_vs_mcts_p1_training_logs= "Training logs/DQN/AI/DQN_VS_MCTS_3_3"
dqn_vs_mcts_p2_training_logs= "Training logs/DQN/AI/MCTS_VS_DQN_3_3"

dqn_agent_one_vs_two_training_logs="Training logs/DQN/self_trained/agent_one_vs_two"
dqn_agent_two_vs_one_training_logs="Training logs/DQN/self_trained/agent_two_vs_one"

ai_trained_model_path= "DQN/dqn_ai_trained/dqn_ai_trained_model.pth"
ai_trained_buffer_path= "DQN/dqn_ai_trained/replay_buffer.pkl"

one_model_path= "DQN/dqn_self_trained_one/dqn_agent_self_trained_one.pth"
one_buffer_path= "DQN/dqn_self_trained_one/replay_buffer.pkl"

two_model_path= "DQN/dqn_self_trained_two/dqn_agent_self_trained_two.pth"
two_buffer_path= "DQN/dqn_self_trained_two/replay_buffer.pkl"

agent_one_vs_mcts_testing="Training logs/DQN/self_trained/agent_one_vs_mcts"
agent_two_vs_mcts_testing="Training logs/DQN/self_trained/agent_two_vs_mcts"

def main():
    # Here is where we load the model we are training
    agent_ai_trained = DQNAgent()
    if os.path.exists(ai_trained_model_path):
        print("✅ Loading saved model...")
        agent_ai_trained.load(ai_trained_model_path,ai_trained_buffer_path)
    else:
        print(" No saved model found, starting fresh training.")

    #finale consolidation attempt
    # train_dqn_minimax_player1(agent_ai_trained,35000,epsilon_start=0.70,epsilon_end=0.25,
    #                           epsilon_decay_episodes=30000,depth=8)
    # train_dqn_mcts_player1(agent_ai_trained, 35000, epsilon_start=0.70, epsilon_end=0.25,
    #                        epsilon_decay_episodes=30000, mcts_time=0.02)
    # train_dqn_minimax_player2(agent_ai_trained, 35000, epsilon_start=0.70, epsilon_end=0.25,
    #                           epsilon_decay_episodes=30000, depth=8)
    # train_dqn_mcts_player2(agent_ai_trained, 35000, epsilon_start=0.70, epsilon_end=0.25,
    #                        epsilon_decay_episodes=30000, mcts_time=0.02)
    #
    # train_dqn_minimax_player1(agent_ai_trained, 30000, epsilon_start=0.25, epsilon_end=0.05,
    #                           epsilon_decay_episodes=20000, depth=8)
    # train_dqn_mcts_player1(agent_ai_trained, 30000, epsilon_start=0.25, epsilon_end=0.05,
    #                        epsilon_decay_episodes=20000, mcts_time=0.02)
    # train_dqn_minimax_player2(agent_ai_trained, 30000, epsilon_start=0.25, epsilon_end=0.05,
    #                           epsilon_decay_episodes=20000, depth=8)
    # train_dqn_mcts_player2(agent_ai_trained, 30000, epsilon_start=0.25, epsilon_end=0.05,
    #                        epsilon_decay_episodes=20000, mcts_time=0.02)

    # train_dqn_minimax_player1(agent_ai_trained, 20000, epsilon_start=0.25, epsilon_end=0.05,
    #                           epsilon_decay_episodes=20000, depth=8)
    # train_dqn_mcts_player1(agent_ai_trained, 20000, epsilon_start=0.25, epsilon_end=0.05,
    #                        epsilon_decay_episodes=20000, mcts_time=0.02)
    # train_dqn_minimax_player2(agent_ai_trained, 20000, epsilon_start=0.25, epsilon_end=0.05,
    #                           epsilon_decay_episodes=20000, depth=8)
    # train_dqn_mcts_player2(agent_ai_trained, 20000, epsilon_start=0.25, epsilon_end=0.05,
    #                        epsilon_decay_episodes=20000, mcts_time=0.02)
    # test_dqn(agent_ai_trained,10000,agent_one_vs_mcts_testing)
    # The problem seems to be, that the model is unable to learn how to play against both algorithms and is instead stuck
    # perpetually being unable to play either to a good standard. I am not going to seek to correct this as i dont have time and its a valid concern with the model !
    print("Training completed and model saved!")





if __name__ == "__main__":
    main()