import time
import tracemalloc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Import all the algorithm implementations
from dfs_algorithm import solve_n_queens_dfs
from greedy_algorithm import solve_n_queens_greedy
from simulated_annealing_algorithm import solve_n_queens_simulated_annealing
from genetic_algorithm import solve_n_queens_genetic

# --- Testing and Data Collection ---

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

# --- Data Presentation and Graph Generation ---

def plot_runtime_bar_chart(results, N_values, algorithms_order, timeout_limit=300):
    """
    Generates a bar chart comparing average runtimes for each algorithm across N values.
    Handles 'TIMEOUT' values by plotting them at the timeout limit.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    width = 0.15 # Width of each bar
    
    # Prepare data for plotting
    x_positions = np.arange(len(algorithms_order)) # Base positions for each algorithm group
    
    # Get a colormap for N values
    N_colors = plt.cm.get_cmap('viridis', len(N_values)) # Get a colormap
    
    for i, N in enumerate(N_values):
        # Calculate offset for current N group
        offset = i * width - (len(N_values) - 1) * width / 2
        
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
        bars = ax.bar(current_x, runtimes, width, label=f'N={N}', color=N_colors(i))
        
        for j, bar in enumerate(bars):
            if labels[j] != "TIMEOUT" and labels[j] != "ERROR" and float(labels[j].replace('s', '')) > 0.00:
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
    
    # Legend for N values
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

# --- Main Function ---
def main():
    """
    Main function that runs the complete N-Queens performance analysis.
    """
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

    # Plot the runtime bar chart
    plot_runtime_bar_chart(collected_data, N_values_to_test, algorithms_order, experiment_timeout)

    # Plot the results table as a graph
    plot_results_table_graph(collected_data, N_values_to_test, algorithms_order)

    print("\nGraphs (Bar Chart and Table) generated. You can now add these to your report.")

# --- Main Execution Block ---
if __name__ == "__main__":
    main() 