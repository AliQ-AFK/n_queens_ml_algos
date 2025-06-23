import time
import random

def solve_n_queens_greedy(N, timeout=300):
    """
    Solves the N-Queens problem using Greedy Hill Climbing.
    Starts with a random board and iteratively tries to reduce conflicts.

    Args:
        N (int): The size of the chessboard (N x N).
        timeout (int): The maximum time in seconds the function is allowed to run.

    Returns:
        tuple: (solution_board, time_taken) if a solution is found within timeout,
               (None, "TIMEOUT") if it times out.
    """
    start_time = time.time()

    def cost(board):
        """
        Calculates the number of conflicts (attacking queen pairs) on the board.
        A lower cost is better.
        """
        conflicts = 0
        for i in range(N):
            for j in range(i + 1, N): # Check each pair of queens once
                # Check row conflict (implicit by board representation) - not needed
                # Check column conflict
                if board[i] == board[j] or \
                   abs(board[i] - board[j]) == abs(i - j): # Check diagonal conflict
                    conflicts += 1
        return conflicts

    # Initialize board with random queen positions (one queen per row, unique columns)
    # This initial setup is better for greedy as it starts with 0 row/column conflicts
    board = list(range(N))
    random.shuffle(board) # Shuffle column assignments to create initial random board

    while time.time() - start_time < timeout:
        current_cost = cost(board)
        if current_cost == 0:
            break # Solution found (zero conflicts)

        # Try to find an improving move by iterating through all possible single queen moves
        # This is the core of hill climbing: find the steepest descent
        best_row_to_move = -1
        best_col_for_move = -1
        min_conflicts_after_move = current_cost

        for queen_row in range(N):
            original_col = board[queen_row]
            for target_col in range(N):
                if original_col == target_col:
                    continue # No change

                # Temporarily make the move to evaluate its cost
                board[queen_row] = target_col
                new_cost = cost(board)
                
                # If this move improves the solution, record it
                if new_cost < min_conflicts_after_move:
                    min_conflicts_after_move = new_cost
                    best_row_to_move = queen_row
                    best_col_for_move = target_col
            
            # Revert the temporary change for the next iteration if no improvement was found for this queen_row
            board[queen_row] = original_col 
        
        # Apply the best improving move found in this iteration
        if min_conflicts_after_move < current_cost:
            board[best_row_to_move] = best_col_for_move
        else:
            # If no move improves, we are stuck in a local optimum.
            # For a basic greedy, this means we cannot proceed.
            # To avoid getting stuck completely (and leading to timeout in all cases where not directly solved),
            # we can introduce a small random step to try to escape, like the friend's code implied.
            # This makes it a "random-restart" or "random-walk" greedy variant rather than pure steepest ascent.
            # The friend's implementation effectively does a random swap if no improvement found.
            # Let's stick to the spirit of the friend's simpler random swap if no direct improvement.
            i, j = random.sample(range(N), 2)
            board[i], board[j] = board[j], board[i]
            # No undo here, as the problem states "randomly swapping queens and keeping improvements"
            # which for a simple greedy might mean accepting it if it's not worse, or just trying.
            # The friend's code explicitly undid if worse, so let's keep that.
            if cost(board) > current_cost:
                board[i], board[j] = board[j], board[i] # Undo if worse than before swap


    end_time = time.time()
    # Return solution only if conflicts are zero
    if cost(board) == 0:
        return board, end_time - start_time
    else:
        return None, "TIMEOUT" 