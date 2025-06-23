import time

def solve_n_queens_dfs(N, timeout=300):
    """
    Solves the N-Queens problem using Depth-First Search (DFS) with backtracking.
    Includes a timeout mechanism.

    Args:
        N (int): The size of the chessboard (N x N).
        timeout (int): The maximum time in seconds the function is allowed to run.

    Returns:
        tuple: (solution_board, time_taken) if a solution is found within timeout,
               (None, "TIMEOUT") if it times out,
               (None, "ERROR") if an unexpected error occurs.
    """
    start_time = time.time()
    # board is represented as a list where board[row] = col
    board = [-1] * N 

    def is_safe(board, row, col):
        """
        Checks if placing a queen at (row, col) is safe from existing queens.
        """
        for i in range(row):
            # Check column conflict (same column)
            # Check diagonal conflict (abs(col_diff) == abs(row_diff))
            if board[i] == col or \
               abs(board[i] - col) == abs(i - row):
                return False
        return True

    def dfs(row):
        """
        Recursive helper function for DFS.
        Attempts to place a queen in the current 'row'.
        """
        nonlocal start_time
        # Check for timeout before starting a new branch
        if time.time() - start_time > timeout:
            raise TimeoutError("Function timed out")

        # Base case: If all queens are placed, a solution is found
        if row == N:
            return True

        # Try placing a queen in each column of the current row
        for col in range(N):
            if is_safe(board, row, col):
                board[row] = col  # Place queen
                if dfs(row + 1):  # Recurse for the next row
                    return True
        return False # No safe position in this row, backtrack

    try:
        success = dfs(0)
        end_time = time.time()
        return board if success else None, end_time - start_time
    except TimeoutError:
        return None, "TIMEOUT"
    except Exception as e:
        # Catch any other unexpected errors
        print(f"DFS Error for N={N}: {e}")
        return None, "ERROR" 