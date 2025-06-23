import time
import random

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