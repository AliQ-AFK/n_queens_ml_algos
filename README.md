# N-Queens Algorithm Performance Analysis

This project analyzes the performance of four different algorithms for solving the N-Queens problem.

## Project Structure

The project has been organized into separate files for better modularity:

### Algorithm Files
- `dfs_algorithm.py` - Depth-First Search (DFS) with backtracking implementation
- `greedy_algorithm.py` - Greedy Hill Climbing algorithm implementation  
- `simulated_annealing_algorithm.py` - Simulated Annealing algorithm implementation
- `genetic_algorithm.py` - Genetic Algorithm implementation

### Main File
- `main.py` - Main execution file that imports all algorithms and runs the complete analysis

### Dependencies
- `requirements.txt` - Lists all required Python packages

## Installation

1. **Create a virtual environment** (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install the required dependencies**:
```bash
pip install -r requirements.txt
```

**Note**: This project requires `matplotlib` and `numpy`. If you encounter any issues with matplotlib installation, you can install it directly:
```bash
pip install matplotlib numpy
```

## Usage

To run the complete N-Queens performance analysis:

```bash
python3 main.py
```

This will:
1. Test all 4 algorithms on board sizes N = [10, 30, 50, 100, 200]
2. Run each algorithm 5 times per board size for statistical accuracy
3. Collect runtime and memory usage data
4. Generate two visualizations:
   - Figure 1: Bar chart comparing average runtimes
   - Table 1: Detailed performance table with time and memory usage

## Algorithms Tested

1. **DFS (Depth-First Search)** - Uses backtracking to systematically explore all possible solutions
2. **Greedy** - Hill climbing approach that tries to minimize conflicts at each step
3. **Anneal (Simulated Annealing)** - Probabilistic algorithm that accepts worse moves with decreasing probability
4. **Genetic** - Evolutionary algorithm that evolves a population of candidate solutions

## Output

The program generates:
- Console output showing progress and results for each test run
- Figure 1: Runtime comparison bar chart 
- Table 1: Performance summary table
- Timeout handling for algorithms that take longer than 5 minutes (300 seconds)

## Individual Algorithm Usage

You can also import and use individual algorithms:

```python
from dfs_algorithm import solve_n_queens_dfs
from greedy_algorithm import solve_n_queens_greedy
from simulated_annealing_algorithm import solve_n_queens_simulated_annealing
from genetic_algorithm import solve_n_queens_genetic

# Example: Solve 8-Queens using DFS
solution, time_taken = solve_n_queens_dfs(8)
if solution:
    print(f"Solution found in {time_taken:.2f}s: {solution}")
else:
    print(f"No solution found or timed out: {time_taken}")
```

Each algorithm function returns a tuple of (solution, time_taken) where:
- `solution` is a list representing queen positions (None if no solution found)
- `time_taken` is the execution time in seconds (or "TIMEOUT"/"ERROR" strings) 