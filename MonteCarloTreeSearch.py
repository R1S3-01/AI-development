import math
import copy
import random
import time
from HandleMoveFunction import handle_move
from EvaluationFunctions import  win_check_position

# For MCTS, it is best to use a Node class, that will store key data about each node in the game tree
class MCTSNode:
    def __init__(self, board, cols_valid_moves, move=None, parent=None, player=None):
        self.board = board
        self.parent = parent
        self.children = []
        self.score = 0
        self.move = move
        self.visits = 0
        self.player_identity = player
        self.cols_valid_moves = cols_valid_moves

#The parent function is the main function which conducts the MCTS and is called to start MCTS
def mcts_parent_function(board, cols_valid_moves, player, limit):
    root = MCTSNode(board=board,cols_valid_moves=cols_valid_moves,  parent=None, player=player)
    start_time=time.time()

    # From the current root node, we add all possible child nodes, a child node stores the boards which are created from moves following the root board position.
    for col in cols_valid_moves.keys():
        # The root board must be copied to prevent it from being altered.
        new_board = copy.copy(root.board)
        new_col_valid_move = copy.copy(cols_valid_moves)
        insertion_position=handle_move(col,new_col_valid_move)
        new_board[insertion_position] = player
        child_node = MCTSNode(board=new_board, cols_valid_moves=new_col_valid_move, move=insertion_position, parent=root, player=player)
        root.children.append(child_node)

    # Funnily enough, this little loop is the where the main bulk of the algorithm is called and controlled
    while time.time()-start_time<limit:
        # Use the selection method to select a leaf/child node from the root.
        leaf = selection(root)
        # If it has been visited before, we expand the leaf
        if leaf.visits > 0:
            leaf = expansion(leaf)
        # To obtain the result from the given leaf we run a simulation
        result = simulation(leaf)
        # Now backpropogate the results up the tree to the root node
        back_propagation(leaf, result)



    # Here is optional code for showing more details to the user. comment these out if testing and dont want the spam.
    # print("Positional score = ", root.score)
    # print("Iterations ran = ", root.visits)
    # Pick the child node which had the highest score and return the input col which created it
    best_child = max(root.children, key=lambda c: c.score / (c.visits + 1e-6))
    return best_child.move[1]



#decisive is the exact same as normal except for a change in the simulation method, it is a separate function to ensure clarity and efficiency
def mcts_decisive_parent_function(board, cols_valid_moves, player, limit):
    root = MCTSNode(board=board,cols_valid_moves=cols_valid_moves,  parent=None, player=player)
    start_time=time.time()

    for col in cols_valid_moves.keys():
        new_board = copy.copy(root.board)
        new_col_valid_move = copy.copy(cols_valid_moves)
        insertion_position=handle_move(col,new_col_valid_move)
        new_board[insertion_position] = player
        child_node = MCTSNode(board=new_board, cols_valid_moves=new_col_valid_move, move=insertion_position, parent=root, player=player)
        root.children.append(child_node)


    while time.time()-start_time<limit:
        leaf = selection(root)
        if leaf.visits > 0:
            leaf = expansion(leaf)
        result = simulation_decisive(leaf)
        back_propagation(leaf, result)




    # print("Positional score = ", root.score)
    # print("Iterations ran = ", root.visits)
    best_child = max(root.children, key=lambda c: c.score / (c.visits + 1e-6))
    return best_child.move[1]

#The selection function is responsible for choosing what child node from the root to select next.
def selection(node):
    # This while node.child, ensures that we are selecting leaf nodes from the game tree.
    while node.children:
        unvisited_children = [child for child in node.children if child.visits == 0]
        if unvisited_children:
            return random.choice(unvisited_children)

        # the Upper Confidence Bounds of each child from the current node are stored
        ucb_scores = []
        for child in node.children:
            # Explotation is the average socre of the node
            exploitation = child.score / (child.visits + 1e-6)
            # Exploration ensures less traversed nodes are favored
            exploration = 2 * math.sqrt(math.log(node.visits + 1e-6) / (child.visits + 1e-6))
            # save the UCB score, exploitation + exploration
            ucb_scores.append(exploitation + exploration)

        # Move to the child with the highest UCB score
        best_index = ucb_scores.index(max(ucb_scores))
        node = node.children[best_index]

    # return the leaf node with the highest UCB1 score
    return node

# The expansion phase is where we expand a current leaf node to add its children into the MCTS game tree
def expansion(node):
    # If the node is a terminal node, it cant be expanded
    if is_terminal(node.board,node.move[0],node.move[1],node.cols_valid_moves):
        return node
    #get all valid moves in a list from the current node
    moves = list(node.cols_valid_moves.keys())
    #for each of these moves, create a child node to the current leaf node which stores the game state after the corresponding move has been made
    for move in moves:
        next_player=3-node.player_identity
        child_cols_valid_move = copy.deepcopy(node.cols_valid_moves)
        insertion_pos = handle_move(move, child_cols_valid_move)
        child_board = copy.deepcopy(node.board)
        child_board[insertion_pos] = next_player  # Place token

        child_node = MCTSNode(
            board=child_board,
            cols_valid_moves=child_cols_valid_move,
            move=insertion_pos,
            parent=node,
            player=next_player
        )
        node.children.append(child_node)

    return random.choice(node.children)

# Simulation is the function which carries out a random game from the given node.
def simulation(node):
    # First make copies of all the nodes current info as we dont want to change any of that
    sim_board = copy.deepcopy(node.board)
    sim_cols_valid_move = copy.deepcopy(node.cols_valid_moves)

    # Now we identify which player we are and get the nodes insertion position,
    # the token inserted into the node which followed on from its proceeding node
    sim_player = node.player_identity
    insertion_pos=(node.move[0],node.move[1])

    # We initially and consistently check if the simulated board is terminal, as this means simulation is completed
    while not is_terminal(sim_board, insertion_pos[0],insertion_pos[1], sim_cols_valid_move):
        #All that is happening is we select a random valid move - Insert the token and then change to the next player token
        col = random.choice(list(sim_cols_valid_move.keys()))
        insertion_pos = handle_move(col, sim_cols_valid_move)
        sim_board[insertion_pos] = sim_player
        sim_player = 3 - sim_player
    # Now simulation is completed, but we must return the correct response for backpropagation
    # More said in the game_outcome function
    return game_outcome(sim_board, insertion_pos)


#Simulation decsive is largely the same as normal simulation, with the only change being:
#If there is ever a move which results in a win, we choose that move.
#Alternatively, if there is a position that would allow the opposing player to win if left, we choose that position
def simulation_decisive(node):

    sim_board = copy.deepcopy(node.board)
    sim_cols_valid_move = copy.deepcopy(node.cols_valid_moves)
    sim_player = node.player_identity
    insertion_pos=(node.move[0],node.move[1])

    while not is_terminal(sim_board, insertion_pos[0],insertion_pos[1], sim_cols_valid_move):
        win_found=False
        random_col = random.choice(list(sim_cols_valid_move.keys()))
        for col in sim_cols_valid_move.keys():
            row=sim_cols_valid_move[col]
            if win_check_position(row,col,sim_board,1) or win_check_position(row,col,sim_board,2):
                insertion_pos = handle_move(col, sim_cols_valid_move)
                win_found=True
                break
        if not win_found:
            insertion_pos = handle_move(random_col, sim_cols_valid_move)

        sim_board[insertion_pos] = sim_player
        sim_player = 3 - sim_player


    return game_outcome(sim_board, insertion_pos)

# This checks who won the game from the insertion position and returns their token, IE 1 or 2.
# If no one won, then 0 is returned, 0 representing a draw
def game_outcome(board,insertion_position):
    if win_check_position(insertion_position[0],insertion_position[1],board,board[insertion_position]):
        return board[insertion_position]
    return 0

# Is terminal checks have been seen before in Minimax, simply checking if the board state is a finished game
def is_terminal(board, row,col,cols_valid_moves):
    if win_check_position(row, col, board, board[row][col]) or not bool(cols_valid_moves):
        return True
    return False

# Back propagation is a small but finicky section,
# In this function, we check if the result of the simulation, the winning token ID is equal to the current nodes identity,
# If it is there is an increase in score as that move path has lead to a simulated win. If not there is a decrease as it lead to a loss
# I have chosen to not score draws, I think this is fine practice for connect four
def back_propagation(node, result):
    while node is not None:
        node.visits += 1
        if result == node.player_identity:
            node.score += 1
        elif result==3-node.player_identity:
            node.score-= 0.25
        node = node.parent



