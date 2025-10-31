import random

def fitness(x):
    # Example fitness function: maximize x^2, x in range [-10,10]
    return x**2

def generate_population(size, lower_bound, upper_bound):
    return [random.uniform(lower_bound, upper_bound) for _ in range(size)]

def selection(population, fitnesses, num_parents):
    # Select individuals with highest fitness
    sorted_pop = [x for _, x in sorted(zip(fitnesses, population), reverse=True)]
    return sorted_pop[:num_parents]

def crossover(parent1, parent2):
    # Single point crossover (for float, simple average)
    return (parent1 + parent2) / 2

def mutation(x, mutation_rate=0.1, mutation_range=1.0):
    if random.random() < mutation_rate:
        x += random.uniform(-mutation_range, mutation_range)
    return x

def genetic_algorithm(pop_size, generations, lower_bound, upper_bound):
    population = generate_population(pop_size, lower_bound, upper_bound)

    for gen in range(generations):
        fitnesses = [fitness(ind) for ind in population]
        parents = selection(population, fitnesses, pop_size // 2)
        next_generation = []
        
        # Generate offspring
        while len(next_generation) < pop_size:
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            child = mutation(child)
            # Keep child within bounds
            child = max(min(child, upper_bound), lower_bound)
            next_generation.append(child)
        
        population = next_generation

    # Return best individual
    fitnesses = [fitness(ind) for ind in population]
    best_index = fitnesses.index(max(fitnesses))
    return population[best_index], fitnesses[best_index]

best_x, best_fit = genetic_algorithm(pop_size=20, generations=100, lower_bound=-10, upper_bound=10)
print(f"Genetic Algorithm best solution: x={best_x:.3f}, fitness={best_fit:.3f}")