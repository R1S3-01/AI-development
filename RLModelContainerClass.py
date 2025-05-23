from DeepQNetwork import DQNAgent
import pickle
import os
# The RLmodelContainer class is used to house the re-enforcement learning algorithms.
# This is beneficial as it allows them to be utilised globally throughout the application without the need for inefficient and unclear global varaibles
# It also keeps the conde more concise and cleaner
class RlModelsContainer:
    q_table = {}
    dqn_model = DQNAgent()
    q_table_loaded=False
    dqn_loaded=False

    def load_q_table(self):
        if not self.q_table_loaded:
            self.q_table_loaded = True
            q_table_filename = "Q_Table_Save_States/Q_Table_Official_Training.pkl"
            if os.path.exists(q_table_filename):
                with open(q_table_filename, "rb") as f:
                    self.q_table = pickle.load(f)
                print("Q-table loaded successfully!")
            else:
                print("No trained Q-table found. The AI will play randomly.")
                self.q_table = {}

    def load_dqn(self):
        if not self.dqn_loaded:
            self.dqn_loaded = True
            dqn_ai_trained_model_load_path = "DQN/dqn_ai_trained/dqn_ai_trained_model.pth"
            self.dqn_model.load(path_to_model=dqn_ai_trained_model_load_path)

    def get_state(self,board,current_player):
        return tuple(map(tuple, board)), current_player

    def q_learning_action_selector(self,cur_state, valid_moves):
        self.load_q_table()
        # If the state isnt yet in the Q-table, we must first add it, this means were in an unexplored position.
        if cur_state not in self.q_table:
            # Because te postion is unexplored, we assigned all actions following this position with a score of 0.
            self.q_table[cur_state] = {move: 0 for move in valid_moves}
        # now we isolate and copy the values of the current state key in the dictionary and get all its following available moves and their scores
        valid_q_values = {move: self.q_table[cur_state][move] for move in self.q_table[cur_state]}
        return max(valid_q_values, key=valid_q_values.get)

    def dqn_action_selector(self,cur_board,valid_cols,cur_player):
        self.load_dqn()
        return self.dqn_model.select_best_action(state=cur_board,valid_actions=valid_cols,current_player=cur_player)