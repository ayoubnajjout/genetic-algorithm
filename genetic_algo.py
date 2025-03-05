import numpy as np

class GeneticAlgorithm:
    def __init__(self, num_itr, Pc, Pm, N, genome_length=4):
        """Initialize the genetic algorithm with parameters."""
        self.num_itr = num_itr
        self.Pc = Pc
        self.Pm = Pm
        self.N = N
        self.genome_length = genome_length
        self.population = self.initialize_population()

    def initialize_population(self):
        """Create an initial random population using NumPy."""
        return np.random.randint(0, 2, size=(self.N, self.genome_length), dtype=np.uint8)

    def evaluate(self):
        """Evaluate fitness: f(x) = x (decimal value of binary string) using vectorization."""
        # Convert binary to decimal using powers of 2
        powers = np.arange(self.genome_length - 1, -1, -1)
        return np.sum(self.population * (2 ** powers), axis=1)

    def select(self, fitnesses):
        """Tournament selection using NumPy."""
        tournament_size = 2
        idx = np.random.choice(self.N, tournament_size, replace=False)
        winner_idx = idx[np.argmax(fitnesses[idx])]
        return self.population[winner_idx].copy()

    def crossover(self, parent1, parent2):
        """Single-point crossover with probability Pc."""
        if np.random.random() < self.Pc:
            point = np.random.randint(1, self.genome_length)
            offspring1 = np.concatenate((parent1[:point], parent2[point:]))
            offspring2 = np.concatenate((parent2[:point], parent1[point:]))
            return offspring1, offspring2
        return parent1.copy(), parent2.copy()

    def mutate(self, individual):
        """Mutate with probability Pm using NumPy."""
        mask = np.random.random(self.genome_length) < self.Pm
        return np.logical_xor(individual, mask).astype(np.uint8)

    def replace(self, fitnesses, offspring):
        """Replacement with elitism using NumPy."""
        elite_size = self.N // 2
        # Sort indices by fitness
        elite_idx = np.argsort(fitnesses)[-elite_size:]
        elite = self.population[elite_idx]
        # Convert offspring list to array and take required number
        offspring_array = np.array(offspring)[:self.N - elite_size]
        return np.vstack((elite, offspring_array))

    def run(self):
        """Run the genetic algorithm."""
        print("Initial Population:", [''.join(map(str, ind)) for ind in self.population])

        for generation in range(self.num_itr):
            fitnesses = self.evaluate()
            print(f"\nGeneration {generation + 1}:")
            print("Fitnesses:", fitnesses.tolist())

            offspring = []
            for _ in range(self.N // 2):  # Generate N offspring (pairs)
                parent1 = self.select(fitnesses)
                parent2 = self.select(fitnesses)
                print("Parents:", ''.join(map(str, parent1)), ''.join(map(str, parent2)))

                child1, child2 = self.crossover(parent1, parent2)
                print("After Crossover:", ''.join(map(str, child1)), ''.join(map(str, child2)))

                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                print("After Mutation:", ''.join(map(str, child1)), ''.join(map(str, child2)))

                offspring.extend([child1, child2])

            self.population = self.replace(fitnesses, offspring)
            print("New Population:", [''.join(map(str, ind)) for ind in self.population])

            if np.max(fitnesses) == 15:
                print("Optimal solution (1111 = 15) found!")
                break

        final_fitnesses = self.evaluate()
        best_idx = np.argmax(final_fitnesses)
        print("\nBest Solution:", ''.join(map(str, self.population[best_idx])), 
              "Fitness:", final_fitnesses[best_idx])

if __name__ == "__main__":
    # Parameters
    num_itr = 10
    Pc = 0.8
    Pm = 0.1
    N = 15

    # Set random seeds for reproducibility
    np.random.seed(42)

    # Create and run the GA
    ga = GeneticAlgorithm(num_itr=num_itr, Pc=Pc, Pm=Pm, N=N)
    ga.run()