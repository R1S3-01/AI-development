def handle_move(col_selection, cols_valid_move):
    next_free_space = cols_valid_move.get(col_selection)
    cols_valid_move[col_selection] = next_free_space - 1
    if next_free_space == 0:
        cols_valid_move.pop(col_selection)
    return next_free_space, col_selection
