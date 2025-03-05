import random

class GeneticAlgorithm:
    def __init__(self, num_itr, Pc, Pm, N, genome_length=4):
        """Initialize the genetic algorithm with parameters."""
        self.num_itr = num_itr  # Number of generations
        self.Pc = Pc            # Crossover probability
        self.Pm = Pm            # Mutation probability
        self.N = N              # Population size
        self.genome_length = genome_length  # Length of binary string
        self.population = self.initialize_population()

    def initialize_population(self):
        """Create an initial random population."""
        return [[random.randint(0, 1) for _ in range(self.genome_length)] 
                for _ in range(self.N)]

    def evaluate(self, individual):
        """Evaluate fitness: f(x) = x (decimal value of binary string)."""
        decimal = int(''.join(map(str, individual)), 2)
        return decimal

    def select(self, fitnesses):
        """Tournament selection: Pick the best of 2 random individuals."""
        tournament_size = 2
        tournament = random.sample(list(zip(self.population, fitnesses)), tournament_size)
        winner = max(tournament, key=lambda x: x[1])[0]  # Best fitness
        return winner

    def crossover(self, parent1, parent2):
        """Single-point crossover with probability Pc."""
        if random.random() < self.Pc:
            point = random.randint(1, self.genome_length - 1)
            offspring1 = parent1[:point] + parent2[point:]
            offspring2 = parent2[:point] + parent1[point:]
            return offspring1, offspring2
        return parent1[:], parent2[:]  # No crossover, return copies

    def mutate(self, individual):
        """Mutate each bit with probability Pm."""
        individual = individual[:]  # Copy to avoid modifying original
        for i in range(self.genome_length):
            if random.random() < self.Pm:
                individual[i] = 1 - individual[i]  # Flip bit
        return individual

    def replace(self, fitnesses, offspring):
        """Replacement with elitism: Keep best N/2, add offspring."""
        sorted_pop = [x for _, x in sorted(zip(fitnesses, self.population), reverse=True)]
        elite_size = self.N // 2  # Keep top half
        new_population = sorted_pop[:elite_size]  # Elitism
        new_population.extend(offspring[:self.N - elite_size])  # Fill with offspring
        return new_population

    def run(self):
        """Run the genetic algorithm."""
        print("Initial Population:", [''.join(map(str, ind)) for ind in self.population])

        for generation in range(self.num_itr):
            # Evaluate population
            fitnesses = [self.evaluate(ind) for ind in self.population]
            print(f"\nGeneration {generation + 1}:")
            print("Fitnesses:", fitnesses)

            # Generate offspring
            offspring = []
            while len(offspring) < self.N:
                # Selection
                parent1 = self.select(fitnesses)
                parent2 = self.select(fitnesses)
                print("Parents:", ''.join(map(str, parent1)), ''.join(map(str, parent2)))

                # Crossover
                child1, child2 = self.crossover(parent1, parent2)
                print("After Crossover:", ''.join(map(str, child1)), ''.join(map(str, child2)))

                # Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                print("After Mutation:", ''.join(map(str, child1)), ''.join(map(str, child2)))

                offspring.extend([child1, child2])

            # Replacement
            self.population = self.replace(fitnesses, offspring)
            print("New Population:", [''.join(map(str, ind)) for ind in self.population])

            # Check for optimal solution
            best_fitness = max(fitnesses)
            if best_fitness == 15:
                print("Optimal solution (1111 = 15) found!")
                break

        # Final result
        final_fitnesses = [self.evaluate(ind) for ind in self.population]
        best_idx = final_fitnesses.index(max(final_fitnesses))
        print("\nBest Solution:", ''.join(map(str, self.population[best_idx])), 
              "Fitness:", final_fitnesses[best_idx])

if __name__ == "__main__":
    # Parameters
    num_itr = 10   # Number of iterations
    Pc = 0.8       # Crossover probability (80%)
    Pm = 0.1       # Mutation probability (10% per bit)
    N = 15         # Population size

    # Set random seed for reproducibility
    random.seed(42)

    # Create and run the GA
    ga = GeneticAlgorithm(num_itr=num_itr, Pc=Pc, Pm=Pm, N=N)
    ga.run()