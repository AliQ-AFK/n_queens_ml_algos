import time
import random
import math
import tracemalloc
import matplotlib.pyplot as plt
import numpy as np

# --- 1. N-Queens Algorithm Implementations ---

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


def solve_n_queens_genetic(N, population_size=100, generations=500, mutation_rate=0.3, elitism_percentage=0.1, timeout=300):
    """
    Solves the N-Queens problem using a Genetic Algorithm.

    Args:
        N (int): The size of the chessboard (N x N).
        population_size (int): The number of individuals in the population.
        generations (int): The maximum number of generations to evolve.
        mutation_rate (float): The probability of an individual mutating.
        elitism_percentage (float): Percentage of top individuals to carry directly to next generation.
        timeout (int): The maximum time in seconds the function is allowed to run.

    Returns:
        tuple: (solution_board, time_taken) if a solution is found within timeout,
               (None, "TIMEOUT") if it times out.
    """
    start_time = time.time()

    def fitness(board):
        """
        Calculates the fitness of a board. Higher is better (more non-attacking pairs).
        Max non_attacks for N queens is N*(N-1)/2
        """
        non_attacks = 0
        for i in range(N):
            for j in range(i + 1, N):
                if board[i] != board[j] and \
                   abs(board[i] - board[j]) != abs(i - j):
                    non_attacks += 1
        return non_attacks

    def mutate(board):
        """
        Mutates a board by randomly changing the column of one queen.
        """
        new_board = board[:]
        idx = random.randint(0, N - 1)
        new_board[idx] = random.randint(0, N - 1) # Assign a new random column
        return new_board

    def crossover(p1, p2):
        """
        Performs single-point crossover between two parent boards.
        """
        point = random.randint(1, N - 1) # Crossover point, avoid 0 or N
        return p1[:point] + p2[point:]

    # Population initialization: Each individual is a list of column indices
    # where index is row and value is column.
    population = [[random.randint(0, N - 1) for _ in range(N)] for _ in range(population_size)]
    max_possible_non_attacks = (N * (N - 1)) // 2 # Target fitness value

    for gen in range(generations):
        if time.time() - start_time > timeout:
            return None, "TIMEOUT"

        # Evaluate fitness for each individual in the population
        # (individual, fitness_score) pairs
        fitness_scores = [(ind, fitness(ind)) for ind in population]
        # Sort population by fitness (higher fitness is better, so reverse=True)
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Check if a solution is found
        best_individual, best_fitness = fitness_scores[0]
        if best_fitness == max_possible_non_attacks:
            return best_individual, time.time() - start_time

        # Create next generation using elitism and selection for reproduction
        next_gen = [ind for ind, _ in fitness_scores[:int(elitism_percentage * population_size)]]

        # Fill the rest of the next generation through crossover and mutation
        # Selection: Roulettewheel selection based on fitness
        total_fitness = sum(f for _, f in fitness_scores)
        if total_fitness == 0: # Avoid division by zero if all fitness are zero
            weights = [1] * population_size # Assign equal weights
        else:
            weights = [f / total_fitness for _, f in fitness_scores]
        
        # Ensure weights sum to 1 for random.choices if needed
        # (random.choices handles non-normalized weights but explicit normalization can be good)

        while len(next_gen) < population_size:
            # Select parents based on their fitness
            p1, p2 = random.choices([ind for ind, _ in fitness_scores], weights=weights, k=2)
            
            # Crossover
            child = crossover(p1, p2)
            
            # Mutation
            if random.random() < mutation_rate:
                child = mutate(child)
            
            next_gen.append(child)
        
        population = next_gen

    # If no solution found after all generations or timeout, return best found or TIMEOUT
    # Re-sort one last time to ensure we return the absolute best if no perfect solution found
    fitness_scores = [(ind, fitness(ind)) for ind in population]
    fitness_scores.sort(key=lambda x: x[1], reverse=True)
    best_individual, best_fitness = fitness_scores[0]

    if best_fitness == max_possible_non_attacks:
        return best_individual, time.time() - start_time
    else:
        return None, "TIMEOUT"

# --- 2. Testing and Data Collection ---

def run_tests_and_collect_data(algorithms, N_values, num_runs=5, timeout=300):
    """
    Runs each algorithm for specified N values and collects average time and memory data.
    """
    results = {} # Store {N: {Alg: {'times': [], 'memories': []}}}

    for N in N_values:
        results[N] = {}
        print(f"--- Testing N = {N} ---")
        for alg_name, alg_func in algorithms.items():
            run_times = []
            peak_memories = []
            
            for i in range(num_runs):
                print(f"  Running {alg_name} for N={N} (run {i+1}/{num_runs})...")
                tracemalloc.start()
                _, t = alg_func(N, timeout=timeout)
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                if t == "TIMEOUT":
                    run_times.append(timeout) # Record timeout as the max allowed time
                    peak_memories.append(np.nan) # Memory is not applicable for timeout
                    print(f"    {alg_name} timed out.")
                    # IMPORTANT: If an algorithm times out on the first run, it's very likely to
                    # timeout on subsequent runs for the same N. Break early to save time.
                    if i == 0: # Only break if it's the first run
                        break 
                elif t == "ERROR":
                    run_times.append(np.nan) # Indicate error
                    peak_memories.append(np.nan)
                    print(f"    {alg_name} encountered an error.")
                    if i == 0:
                        break
                else:
                    run_times.append(t)
                    peak_memories.append(peak / 1024) # Convert bytes to KB
                    print(f"    {alg_name} finished in {t:.2f}s, Memory: {peak / 1024:.2f}KB")
            
            # Calculate averages, handling timeouts/errors
            avg_time = np.mean([r for r in run_times if isinstance(r, float)]) if run_times and any(isinstance(r, float) for r in run_times) else "TIMEOUT"
            avg_memory = np.nanmean(peak_memories) if peak_memories and not all(np.isnan(m) for m in peak_memories) else "N/A"

            # If all runs for a specific N and algorithm timed out, explicitly set avg_time to "TIMEOUT"
            if all(t == timeout for t in run_times):
                avg_time = "TIMEOUT"
            
            results[N][alg_name] = {
                'avg_time': avg_time,
                'avg_memory': avg_memory,
                'raw_times': run_times # Keep raw times for debugging/detailed analysis
            }
    return results

# --- 3. Data Presentation and Graph Generation ---

def plot_runtime_bar_chart(results, N_values, algorithms_order, timeout_limit=300):
    """
    Generates a bar chart comparing average runtimes for each algorithm across N values.
    Handles 'TIMEOUT' values by plotting them at the timeout limit.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    width = 0.15 # Width of each bar
    
    # Colors for each algorithm
    colors = {
        'DFS': 'skyblue',
        'Greedy': 'lightcoral',
        'Anneal': 'lightgreen',
        'Genetic': 'gold'
    }

    # Prepare data for plotting
    x_positions = np.arange(len(algorithms_order)) # Base positions for each algorithm group
    
    for i, N in enumerate(N_values):
        # Calculate offset for current N group
        offset = i * (width + 0.02) - (len(N_values) - 1) * (width + 0.02) / 2
        
        current_x = x_positions + offset
        
        runtimes = []
        labels = []
        for alg_name in algorithms_order:
            avg_time = results[N][alg_name]['avg_time']
            if avg_time == "TIMEOUT":
                runtimes.append(timeout_limit)
                labels.append("TIMEOUT")
            elif isinstance(avg_time, float):
                runtimes.append(avg_time)
                labels.append(f"{avg_time:.2f}s")
            else: # Handle "ERROR" or other non-numeric statuses
                runtimes.append(0) # Plot at 0 or small value if error
                labels.append(str(avg_time))
            
        bars = ax.bar(current_x, runtimes, width, label=f'N={N}', color=colors[algorithms_order[0]]) # Use a consistent color for legend
        
        # Add N labels to bars for clarity when bars are grouped
        for j, bar in enumerate(bars):
            # Only add runtime label if not timeout and value is significant
            if labels[j] != "TIMEOUT" and float(labels[j].replace('s', '')) > 0.00:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{float(labels[j].replace('s', '')):.2f}s",
                        ha='center', va='bottom', fontsize=7, color='black')
            elif labels[j] == "TIMEOUT":
                 ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, labels[j],
                        ha='center', va='bottom', color='red', fontsize=8, weight='bold')

    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Average Runtime (seconds)')
    ax.set_title('Figure 1: Average Runtime Comparison of N-Queens Algorithms')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(algorithms_order)
    
    # Create custom legend for N values using the general color scheme
    custom_handles = []
    for N_val, color_key in zip(N_values, algorithms_order): # Re-purpose colors for N values
        custom_handles.append(plt.Rectangle((0,0),1,1, fc=colors[color_key], label=f'N={N_val}'))
    
    # Corrected legend approach for N values
    # We need to draw a dummy bar for each N value to make the legend work correctly.
    # Instead, let's create a legend based on the N values directly without relying on bars.
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=color, label=f'N={N_value}') 
                      for N_value, color in zip(N_values, [colors['DFS'], colors['Greedy'], colors['Anneal'], colors['Genetic'], 'purple'])] # Add one more color for N=200 if needed
    
    # A cleaner approach for the legend of N values:
    legend_labels = [f'N={N}' for N in N_values]
    # For a grouped bar chart, the default legend often lumps all bars together.
    # A cleaner way is to create an explicit legend for N values.
    # Let's adjust the bar colors based on N, not algorithm, for clarity in this chart type.
    
    # Resetting colors to make bars grouped by algorithm, but distinct by N
    # This requires more complex color mapping or different plotting logic.
    # For simplicity, let's make all bars for a given algorithm the same color,
    # and then use a separate legend for the N values as a guide.
    
    # Let's try to assign colors based on Algorithm and then distinguish N values
    # by adding a second legend or using distinct bar styles.
    
    # For a grouped bar chart, the simplest legend is for the N values directly.
    # The current `colors` dictionary is for algorithms.
    
    # Let's assign a set of colors for each N value.
    N_colors = plt.cm.get_cmap('viridis', len(N_values)) # Get a colormap
    
    # Re-draw the bars with N-specific colors for the legend to be meaningful
    ax.cla() # Clear current axes to re-plot
    
    for i, N in enumerate(N_values):
        current_x = x_positions + offset
        runtimes = []
        labels = []
        for alg_name in algorithms_order:
            avg_time = results[N][alg_name]['avg_time']
            if avg_time == "TIMEOUT":
                runtimes.append(timeout_limit)
                labels.append("TIMEOUT")
            elif isinstance(avg_time, float):
                runtimes.append(avg_time)
                labels.append(f"{avg_time:.2f}s")
            else: # Handle "ERROR" or other non-numeric statuses
                runtimes.append(0) # Plot at 0 or small value if error
                labels.append(str(avg_time))
        
        # Use N_colors for plotting and legend
        bars = ax.bar(x_positions + i * width - (len(N_values) - 1) * width / 2, # Adjust x positions for grouping
                      runtimes, width, label=f'N={N}', color=N_colors(i)) 
        
        for j, bar in enumerate(bars):
            if labels[j] != "TIMEOUT" and float(labels[j].replace('s', '')) > 0.00:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{float(labels[j].replace('s', '')):.2f}",
                        ha='center', va='bottom', fontsize=7, color='black')
            elif labels[j] == "TIMEOUT":
                 ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, labels[j],
                        ha='center', va='bottom', color='red', fontsize=8, weight='bold')

    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Average Runtime (seconds)')
    ax.set_title('Figure 1: Average Runtime Comparison of N-Queens Algorithms')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(algorithms_order)
    
    # Corrected legend for N values based on the new N_colors
    ax.legend(title="Board Size (N)")

    # Add a horizontal line for the timeout limit
    ax.axhline(y=timeout_limit, color='r', linestyle='--', label=f'Timeout Limit ({timeout_limit}s)')
    ax.set_ylim(bottom=0, top=timeout_limit + 50) # Extend y-axis slightly above timeout
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_results_table_graph(results, N_values, algorithms_order):
    """
    Generates a graphical table of the performance results.
    """
    fig, ax = plt.subplots(figsize=(10, 6)) # Adjust figure size as needed
    ax.axis('off') # Hide axes
    ax.set_title('Table 1: Average Time and Memory Usage for N-Queens Algorithms', loc='center', fontsize=14, pad=20)

    # Prepare data for the table
    header = ['N', 'Algorithm', 'Avg Time (s)', 'Avg Memory (KB)']
    table_data = []

    for N in N_values:
        for alg_name in algorithms_order:
            data = results[N].get(alg_name, {'avg_time': 'N/A', 'avg_memory': 'N/A'})
            
            # Format time
            avg_t_str = f"{data['avg_time']:.2f}" if isinstance(data['avg_time'], float) else str(data['avg_time'])
            
            # Format memory
            avg_mem_str = f"{data['avg_memory']:.2f}" if isinstance(data['avg_memory'], float) else str(data['avg_memory'])
            
            table_data.append([str(N), alg_name, avg_t_str, avg_mem_str])

    # Create the table
    table = ax.table(cellText=table_data,
                     colLabels=header,
                     loc='center',
                     cellLoc='center', # Center text in cells
                     colColours=['#f2f2f2']*len(header), # Header background color
                     colWidths=[0.1, 0.2, 0.25, 0.25] # Adjust column widths
                    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2) # Scale the table to make it larger/more readable

    # Style the table cells
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor('black')
        cell.set_linewidth(0.5)
        if i == 0: # Header row
            cell.set_facecolor('#d9d9d9') # Header row background
            cell.set_text_props(weight='bold', color='black')
        else:
            cell.set_facecolor('white') # Data row background
            # Highlight 'TIMEOUT' in red
            if 'TIMEOUT' in cell.get_text().get_text():
                cell.set_text_props(color='red', weight='bold')
    
    plt.tight_layout()
    plt.show()

# --- Main Execution Block ---
if __name__ == "__main__":
    # Define algorithms and their respective functions
    algorithms = {
        "DFS": solve_n_queens_dfs,
        "Greedy": solve_n_queens_greedy,
        "Anneal": solve_n_queens_simulated_annealing,
        "Genetic": solve_n_queens_genetic
    }
    
    # Order of algorithms for consistent table/chart display
    algorithms_order = ["DFS", "Greedy", "Anneal", "Genetic"]

    # Define N values to test
    N_values_to_test = [10, 30, 50, 100, 200]
    
    # Number of runs per algorithm for each N value
    num_experimental_runs = 5
    
    # Timeout limit in seconds
    experiment_timeout = 300

    print("Starting N-Queens performance analysis...")
    collected_data = run_tests_and_collect_data(
        algorithms, 
        N_values_to_test, 
        num_runs=num_experimental_runs, 
        timeout=experiment_timeout
    )

    print("\n--- Analysis Complete ---")

    # Now, instead of just printing, we'll plot the table
    # print_results_table(collected_data, N_values_to_test, algorithms_order) # This line is no longer needed

    # Plot the runtime bar chart
    plot_runtime_bar_chart(collected_data, N_values_to_test, algorithms_order, experiment_timeout)

    # Plot the results table as a graph
    plot_results_table_graph(collected_data, N_values_to_test, algorithms_order)

    print("\nGraphs (Bar Chart and Table) generated. You can now add these to your report.")


