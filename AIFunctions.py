import random
from Minimax import min_max,iterative_deepening_minimax
from MonteCarloTreeSearch import mcts_parent_function,mcts_decisive_parent_function
from DeepQNetwork import DQNAgent
from RLModelContainerClass import RlModelsContainer

def random_handler(board,cols,player_token):
    return random.choice(list(cols.keys()))

def heuristic_handler(board,cols,player_token):
    return min_max(board, cols, (0, 0), True, player_token, 1, float('-inf'), float('inf'))[1]

def low_iterative_deepening_minimax(board,cols,player_token):
    return iterative_deepening_minimax(board, cols, player_token, 13, 0.5)

def high_iterative_deepening_minimax(board,cols,player_token):
    return iterative_deepening_minimax(board, cols, player_token, 14, 1)

def q_learning_handler(board,cols,player_token):
    state=models.get_state(board,player_token)
    return models.q_learning_action_selector(state,cols)

def deep_q_network_handler(board, cols, player_token):
    return models.dqn_model.select_best_action(board, cols, player_token)

def low_minmax_handler(board,cols,player_token):
    return min_max(board,cols,(0,0),True,player_token,9,float('-inf'), float('inf'))[1]


def high_minmax_handler(board, cols, player_token):
    return min_max(board, cols, (0, 0), True, player_token, 12, float('-inf'), float('inf'))[1]

def low_mcts_handler(board,cols,player_token):
    return mcts_parent_function(board,cols,player_token,0.25)


def high_mcts_handler(board, cols, player_token):
    return mcts_parent_function(board, cols, player_token, 1)

def low_mcts_dec_handler(board, cols, player_token):
    return mcts_decisive_parent_function(board,cols,player_token,0.2)

def hig_mcts_dec_handler(board, cols, player_token):
    return mcts_decisive_parent_function(board, cols, player_token, 1)

models=RlModelsContainer()
#The functions list is utilised repeatedly in testing, it allows functions to be treated like variables and called by their index in this list
function_call_list=[
    random_handler,heuristic_handler,
    low_mcts_handler,low_minmax_handler,low_mcts_dec_handler,low_iterative_deepening_minimax,
    q_learning_handler,deep_q_network_handler,
    high_minmax_handler,high_mcts_handler,hig_mcts_dec_handler,high_iterative_deepening_minimax
]

# Similarly, what can also be seen is the functions name list,this is handy, as say we are using a function;
# we can call the function by doing function_call_list[x]
# we can then get the name of the function being used by saying name= function_name_list[4]
function_name_list=[
    "Rand","Heuristic",
    "Low-MCTS","Low-Mini","low-MCTS-Dec","Low-Mini-Id",
    "Q-learn","DQN",
    "High-Mini","High-MCTS","High-MCTS-Dec","High-Mini-Id"
]

function_call_list_low=[
    random_handler,heuristic_handler,
    low_mcts_handler,low_minmax_handler,low_mcts_dec_handler,low_iterative_deepening_minimax,
    q_learning_handler,deep_q_network_handler
]

# Similarly, what can also be seen is the functions name list,this is handy, as say we are using a function;
# we can call the function by doing function_call_list[x]
# we can then get the name of the function being used by saying name= function_name_list[4]
function_name_list_low=[
    "Rand","Heuristic",
    "Low-MCTS","Low-Mini","low-MCTS-Dec","Low-Mini-Id",
    "Q-learn","DQN"
]