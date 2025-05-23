from SetupAndConfig import columns,rows


# The directional_win_check is a great function!
# Notice that their is only one function, no messy: check_left, check_right etc...
#This is done through passing in the original position and the next position to check,
# in doing so we can keep recursively calling incrementing in the same manner these two vars are passed in
def directional_win_check(origin_position, next_position, board, tokens_in_row, check_token):
    #Big check here which is essentially checking if the win check is finished, if any of these conditions are met, end the check
    if tokens_in_row == 3 or next_position[0] < 0 or next_position[0] == rows or next_position[1] < 0 or next_position[
        1] == columns or check_token != board[next_position]:
        return tokens_in_row
    # If the previous condition wasn't triggered, then we know we the next_position contains our check_token and thus we can finish
    tokens_in_row += 1
    # From here we recursively call the function again, passing the next pos as the original pos and the next pos = (next pos + difference between original and next pos)
    return directional_win_check(
        next_position,
        (next_position[0] + (next_position[0] - origin_position[0]),
         next_position[1] + (next_position[1] - origin_position[1])),
        board,
        tokens_in_row,
        check_token
    )

# Directional score check is used to recursively traverse the board in a given direction, discovering all matching and consecutive tokens:
# This works using the same principles as the directional win check, however with the added necessity of knowing how to sequence ends.
# The reason for this is clear, imagine a three in a row, not all three in a rows are equal. Imagine if P1 had tokens in ((0,1)(0,2)(0,3)),
# you may initally think this is a good thing and P1 has won, but what if P2 has tokens in (0,0) & (0,4) now the three in a row is redundant.
# Thus tokens in a row are only valuable if the are followed by an empty and accessible space. That is what this score check facilitates
def directional_score_check(origin_position, next_position, board,check_value, consecutive_tokens=0):

    if (next_position[0] < 0 or next_position[0] == rows or next_position[1] < 0 or next_position[1] == columns or
            board[next_position] == 3 - check_value or consecutive_tokens == 3):
        return 0, consecutive_tokens

    if board[next_position] == check_value:
        consecutive_tokens+=1
    else:
        return 1, consecutive_tokens

    return directional_score_check(
        next_position,
        (next_position[0] + (next_position[0] - origin_position[0]),
         next_position[1] + (next_position[1] - origin_position[1])),
        board,
        check_value,
        consecutive_tokens
    )

# Score assigner is a heuristic eval function, its role is to score a given row of tokens.
# This is relatively complex to perfectly do and this function is undoubtedly imperfect, as is the nature of heuristic functions.
def score_assigner(consecutive_tokens, empty_spaces_d1, empty_spaces_d2):
    score = 0
    # If there are 3 or over consecutive tokens, there is a win, and thus we return a massive positive value
    if consecutive_tokens >= 3:
        return 100000
    # This condition is for handling 3 connecting tokens. Notice how the empty space determines the value as previously discussed!
    elif consecutive_tokens == 2 and (empty_spaces_d1 + empty_spaces_d2) > 0:
        if empty_spaces_d1 > 0 and empty_spaces_d2 > 0:
            score += 20 * 20
        else:
            score += 20
    # This condition is used to handle if there is only a single connecting token, thus it is scored much lower.
    elif consecutive_tokens == 1 and (empty_spaces_d1 + empty_spaces_d2) > 1:
        score += 5 * 5

    #Finally, just to slightly encourage connection playing if no better moves are available, we increase the score by the amount of consecutive tokens
    score += consecutive_tokens
    return score

# Score position is used to score a given position in the board by traversing all adjoining paths and gathering data.
def score_position(row, col, board, check_token):
    temp_score = 0
    #Small bonus for tokens in middle col
    if col==3:
        temp_score+=30
    #Small negative for tokens in end cols
    if col==0 or col==6:
        temp_score-=2

    # As demonstrated in the win_check_position function, we do not consider single directions at a time, we combine each direction with its opposing direction.
    # The reason for this is clear if you imagine P1 has token in slots ((0,0)(0,1)(0,3)) and then dropped a 4th token in slot (0,2). To correctly interpret that
    # pos we would have to score it as score = left score + right score
    right_empty_spaces_in_path, right_consec = directional_score_check(
        (row, col), (row, col + 1), board, check_token)

    left_empty_spaces_in_path, left_consec = directional_score_check(
        (row, col), (row, col - 1), board, check_token)

    # Score assigner is where really get into the heuristic evaluation component of this project
    temp_score += score_assigner(
        right_consec + left_consec,
        right_empty_spaces_in_path,
        left_empty_spaces_in_path
    )

    up_empty_spaces_in_path, up_consec = directional_score_check(
        (row, col), (row - 1, col), board, check_token)

    down_empty_spaces_in_path, down_consec = directional_score_check(
        (row, col), (row + 1, col), board, check_token)

    temp_score += score_assigner(
        up_consec + down_consec,
        up_empty_spaces_in_path,
        down_empty_spaces_in_path
    )

    right_up_empty_spaces_in_path, right_up_consec = directional_score_check(
        (row, col), (row - 1, col - 1), board, check_token)

    left_down_empty_spaces_in_path, left_down_consec = directional_score_check(
        (row, col), (row + 1, col + 1), board, check_token)

    temp_score += score_assigner(
        right_up_consec + left_down_consec,

        right_up_empty_spaces_in_path,
        left_down_empty_spaces_in_path
    )

    left_up_empty_spaces_in_path, left_up_consec = directional_score_check(
        (row, col), (row - 1, col + 1), board, check_token)

    right_down_spaces_in_path, right_down_consec = directional_score_check(
        (row, col), (row + 1, col - 1), board, check_token)

    temp_score += score_assigner(
        left_up_consec + right_down_consec,

        left_up_empty_spaces_in_path,
        right_down_spaces_in_path
    )
    return temp_score


# Score board simplified is the function which goes through every index of the board with a player token and calls the score position function
# The score of a board for player X can be seen as (Player X Score - Player Y Score)
def score_board_simplified(board, check_token):
    score = 0
    for row in range(5,  -1, -1):
        for col in range(0, columns):
            if board[row, col] == check_token:
                score += score_position(row, col, board, check_token)
            if board[row, col] == 3 - check_token:
                score -= score_position(row, col, board, 3 - check_token)
    return score


#is_there_a_win checks the entire board for a win, this is used
def is_there_a_win(board):
    for row in range(5,  -1, -1): # This is a reverse loop, start at 5 go down to 0
        for col in range(0, columns): # This loop starts at 0 goes to 6, as 7 columns
            if board[row, col] !=0: # If the token in the position is 0 there cant be a win from here no need to check
                # Win check position explained below
                win= win_check_position(row, col, board, board[row,col])
                if win:
                    return board[row,col]
    return False


# Win_check_position is the function used to find if there is a win corresponding to given cell in the board.
# It does this by checking how many matching tokens are in the possible win directions, meaning (below+above) & (left+right) & (both diagonals)
# If any of these checks sum to three that means there is 4 in a row, as there are 3 plus the original token.
def win_check_position(row, col, board, check_token):
    # if the token in the position is 0, we know there can be no win from it. This means the index is empty.
    if check_token==0:
        return False

    # we do not check for a win one direction at a time, instead we combine to opposing directions.
    # Consider if a player had their tokens in positions: ((0,0),(0,1),(0,2),(0,3))
    # If we checked is there a win to left of pos (0,1) it would be false. If we checked to the right it would be false. You have to check the entire axis
    right_consec = directional_win_check(
        (row, col), (row, col + 1), board, 0,  check_token)

    left_consec = directional_win_check(
        (row, col), (row, col - 1), board, 0, check_token)

    if right_consec+left_consec>=3:return True

    up_consec = directional_win_check(
        (row, col), (row - 1, col), board, 0,  check_token)

    down_consec = directional_win_check(
        (row, col), (row + 1, col), board, 0, check_token)

    if up_consec + down_consec >= 3: return True

    right_up_consec = directional_win_check(
        (row, col), (row - 1, col - 1), board, 0, check_token)

    left_down_consec = directional_win_check(
        (row, col), (row + 1, col + 1), board, 0,  check_token)

    if right_up_consec + left_down_consec >= 3: return True

    left_up_consec = directional_win_check(
        (row, col), (row - 1, col + 1), board, 0, check_token)

    right_down_consec = directional_win_check(
        (row, col), (row + 1, col - 1), board, 0, check_token)

    if left_up_consec + right_down_consec >= 3: return True
    #If no wins found it must be false
    return False








