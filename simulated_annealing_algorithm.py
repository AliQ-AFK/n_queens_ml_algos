import time
import random
import math

def solve_n_queens_simulated_annealing(N, initial_temp=1.0, cooling_rate=0.995, max_iterations_per_temp=100, timeout=300):
    """
    Solves the N-Queens problem using Simulated Annealing.
    Allows acceptance of worse moves with decreasing probability.

    Args:
        N (int): The size of the chessboard (N x N).
        initial_temp (float): Starting temperature for annealing.
        cooling_rate (float): Rate at which temperature decreases (alpha).
        max_iterations_per_temp (int): Number of moves attempted at each temperature.
        timeout (int): The maximum time in seconds the function is allowed to run.

    Returns:
        tuple: (solution_board, time_taken) if a solution is found within timeout,
               (None, "TIMEOUT") if it times out.
    """
    start_time = time.time()

    def cost(board):
        """
        Calculates the number of conflicts (attacking queen pairs) on the board.
        """
        conflicts = 0
        for i in range(N):
            for j in range(i + 1, N):
                if board[i] == board[j] or \
                   abs(board[i] - board[j]) == abs(i - j):
                    conflicts += 1
        return conflicts

    # Initial random board configuration
    current_board = list(range(N))
    random.shuffle(current_board)
    current_cost = cost(current_board)
    
    temp = initial_temp

    while temp > 0.0001 and (time.time() - start_time < timeout):
        if current_cost == 0:
            break # Solution found

        for _ in range(max_iterations_per_temp):
            if time.time() - start_time > timeout: # Check timeout within inner loop too
                break

            # Generate a neighbor by swapping two random queens
            i, j = random.sample(range(N), 2)
            new_board = list(current_board) # Create a copy to modify
            new_board[i], new_board[j] = new_board[j], new_board[i]
            
            new_cost = cost(new_board)
            delta = new_cost - current_cost

            # Metropolis criterion: Accept if better, or with a probability if worse
            if delta < 0 or (temp > 0 and random.random() < math.exp(-delta / temp)):
                current_board = new_board
                current_cost = new_cost
                if current_cost == 0: # Check for solution immediately after move
                    break
        
        temp *= cooling_rate # Decrease temperature

    end_time = time.time()
    if current_cost == 0:
        return current_board, end_time - start_time
    else:
        return None, "TIMEOUT" 