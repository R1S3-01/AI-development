import math

import time

import numpy as np

from EvaluationFunctions import score_board_simplified,score_position,win_check_position
from HandleMoveFunction import handle_move

transposition_table = {}
# Minimax performs best when moves are explored best -> worse.
# This move_order_maker, evaluates each valid token placement position and returns them in a best to worse order.
# It does this in a naive but efficient way, of scoring the positions only immediately around the placed token, using score_poistion.
# It is not scoring the whole board, as this would take too long and the gain in accuracy isnt worse the loss of efficiency and thus depth
def move_order_maker(board, cols_valid_move, maximising_player,origin_player):
    # moves and scores is a list which stores tuples, moves and their scores.
    moves_and_scores = []
    # Establish what player we are evaulating for
    token = origin_player  if maximising_player else 3-origin_player

    for col in cols_valid_move.keys():
        # Theoretically insert the token and score the board
        theoretical_row = cols_valid_move[col]
        theoretical_board=board.copy()
        theoretical_board[theoretical_row][col]=token
        score = score_position(theoretical_row,col,theoretical_board,token)
        moves_and_scores.append((col, score))
    #sort the list
    moves_and_scores.sort(key=lambda x: x[1], reverse=True)
    #only return the moves not the scores
    return [move[0] for move in moves_and_scores]

#This function checks if a node is terminal, meaning that in the given state the game is over
def is_terminal_node(board, cols_valid_move,insertion_pos, player_token):
    #check if a player has one from the insertion position, not the whole board as this is unnecessary.
    #If there is a win, it could only have happened in the last move as this check is applied to all prior moves as well.
    if win_check_position(insertion_pos[0],insertion_pos[1], board, player_token):
        return player_token
    #Is there any valid insertion positions? If not board is full
    if not bool(cols_valid_move):
        #Has to be a unint8 0 as otherwise it is detected as False causing errors.
        return np.uint8(0)
    else:
        #Else return a string for easy checking
        return ""

#Iterative deepening minimax uses normal minmax, but only expands the tree one depth at a time untill the time limit is reached.
#Building the Game Tree Incrementally.
def iterative_deepening_minimax(board, cols_valid_move, origin_player_token, max_depth, time_limit):
    start_time = time.time()
    best_move = None
    for depth in range(1, max_depth + 1):

        if time.time() - start_time >= time_limit or depth>max_depth:
            break

        score, move = min_max(board, cols_valid_move, (0, 0), True, origin_player_token, depth, -math.inf, math.inf)

        if move is not None:
            best_move = move
    return best_move


# Too much to comment how minimax works, but key sections and distinctive points will be commented
def min_max(board, cols_valid_move, insertion_pos, maximising_player, origin_player_token, depth, alpha, beta):
    # At the start of every new position being explored, it must be determined if it is a terminal node.
    # as if so we stop traversing the path and backpropagate the results.
    terminal_node = is_terminal_node(board, cols_valid_move, insertion_pos, board[insertion_pos])

    # If the cur node is a terminal node or we have reached the depth limit we will end the Minimax game tree traversal.
    if depth == 0 or isinstance(terminal_node, np.uint8):
        # The +- depth is done to encourage faster wins and longer loses!
        if terminal_node == origin_player_token:
            return 100000 + depth, None
        elif terminal_node == 3 - origin_player_token:
            return -100000 - depth, None
        elif terminal_node == 0:
            return 0, None
        else:
            return score_board_simplified(board, origin_player_token), None

    # If the node isnt terminal, we check to see if it is stored in the transposition table.
    # to do this we get the board key, which is done by convertin the board to bytes, making it smaller and unique
    board_key = board.tobytes()
    # now we make the key a tuple as this allows us to hash more data which is necessary
    board_key = (board_key, depth, maximising_player, origin_player_token)
    # now if a there is a key in the transpos table which matches our board key, we simply return that value
    # As this means we have previously explored up to the depth we are currently exploring from this point, from the perspective of this player!
    if board_key in transposition_table:
        return transposition_table[board_key]

    #Get the best move order
    best_move_order = move_order_maker(board, cols_valid_move, maximising_player, origin_player_token)

    if maximising_player:
        value = -math.inf
        best_col = best_move_order[0]
        for col in best_move_order:
            theoretical_board = board.copy()
            theoretical_cols = cols_valid_move.copy()
            new_insert_pos = handle_move(col, theoretical_cols)
            theoretical_board[new_insert_pos] = origin_player_token

            new_score, _ = min_max(theoretical_board, theoretical_cols, new_insert_pos,
                                   False, origin_player_token, depth - 1, alpha, beta)
            if new_score > value:
                value = new_score
                best_col = col
            alpha = max(alpha, value)
            # Alpha beta pruning at work, if this condition is met, it means this path will never be traversed
            if beta <= alpha:
                break
    else:
        value = math.inf
        best_col = best_move_order[0]
        for col in best_move_order:
            theoretical_board = board.copy()
            theoretical_cols = cols_valid_move.copy()
            new_insert_pos = handle_move(col, theoretical_cols)
            theoretical_board[new_insert_pos] = 3 - origin_player_token

            new_score, _ = min_max(theoretical_board, theoretical_cols, new_insert_pos,
                                   True, origin_player_token, depth - 1, alpha, beta)
            if new_score < value:
                value = new_score
                best_col = col
            beta = min(beta, value)
            # Alpha beta pruning at work, if this condition is met, it means this path will never be traversed
            if beta <= alpha:
                break
    # Store the best move and its value from the current board key in the table
    transposition_table[board_key] = (value, best_col)
    # Return the best value and best col
    return value, best_col


