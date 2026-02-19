import numpy as np
import random
import matplotlib.pyplot as plt
from dataclasses import dataclass
from slopes import get_slope_angle

# =========================================================================
# Plot the vehicle's state history. 
# =========================================================================
def plot(vehicle_state):

    horizontal = vehicle_state.distance

    slope = vehicle_state.slope
    speed = vehicle_state.speed
    brake_pressure = vehicle_state.brake_pressure
    brake_temperature = vehicle_state.brake_temperature
    gear = vehicle_state.gear

    fig, axes = plt.subplots(5, 1, sharex=True)

    axes[0].plot(horizontal, slope)
    axes[0].set_title("Slope vs Distance")

    axes[1].plot(horizontal, brake_pressure)
    axes[1].set_title("Brake Pressure vs Distance")

    axes[2].plot(horizontal, gear)
    axes[2].set_title("Gear vs Distance")

    axes[3].plot(horizontal, speed)
    axes[3].set_title("Speed vs Distance")

    axes[4].plot(horizontal, brake_temperature)
    axes[4].set_title("Brake Temperature vs Distance")

    plt.tight_layout()
    plt.show()
    return

# =========================================================================
# Run the network on validation or test dataset and get the error.
# =========================================================================
def validate_network(best_chromosome, validation_dataset_index, 
                     constant_params):
    
    max_possible_speed = constant_params.vehicle_params.v_max
    target_distance = constant_params.universal_params.target_distance
    maximum_possible_fitness = max_possible_speed * target_distance
    
    average_validation_fitness = 0
    for validation_slope_index in range(1,6):
        validation_fitness = evaluate_individual(best_chromosome, 
                                                 validation_slope_index, 
                                                 validation_dataset_index, 
                                                 constant_params,
                                                 True, False)
        average_validation_fitness += validation_fitness
    
    average_validation_fitness /= 5
    error = abs(average_validation_fitness - maximum_possible_fitness)

    return average_validation_fitness, error

# =========================================================================
# Compute the fitness for the vehicle's performance for training dataset.
# =========================================================================
def compute_fitness_training(vehicle_state, constant_params):

    distance_traveled = vehicle_state.distance[-1]
    avg_speed = np.mean(vehicle_state.speed)
    fitness = avg_speed * distance_traveled

    # compute penalties.
    penalty = 0

    # speed violation penalties (quadratic).
    for speed in vehicle_state.speed:
        if speed > constant_params.vehicle_params.v_max:
            penalty += 5 * (speed - constant_params.vehicle_params.v_max) ** 2

        if speed < constant_params.vehicle_params.v_min:
            penalty += (constant_params.vehicle_params.v_min - speed) ** 2

    # brake temperature violation penalties (linear).
    for brake_temp in vehicle_state.brake_temperature:
        if brake_temp > constant_params.vehicle_params.T_max:
            penalty += abs(brake_temp - constant_params.vehicle_params.T_max)

    # scale down penalty and penalize.
    fitness /= (1 + 0.001 * penalty)

    return fitness

# =========================================================================
# Compute the fitness for the vehicle's performance for validation or test
# dataset.
# =========================================================================
def compute_fitness_validation(vehicle_state, constant_params):

    last_speed = vehicle_state.speed[-1]
    last_brake_temp = vehicle_state.brake_temperature[-1]

    max_allowed_speed = constant_params.vehicle_params.v_max
    min_allowed_speed = constant_params.vehicle_params.v_min
    max_allowed_brake_temp = constant_params.vehicle_params.T_max

    # fitness = avg speed x distance traveled
    # ignore last entries of speed and distance traveled if any constraint 
    # is violated as these entries are invalid (in the context of 
    # evaluating fitness)
    if last_speed > max_allowed_speed or\
            last_speed < min_allowed_speed or\
            last_brake_temp > max_allowed_brake_temp:
                distance_traveled = vehicle_state.distance[-2]
                avg_speed = np.mean(vehicle_state.speed[:-1])
    else:
        distance_traveled = vehicle_state.distance[-1]
        avg_speed = np.mean(vehicle_state.speed)

    fitness = avg_speed * distance_traveled

    return fitness

# =========================================================================
# Compute sigmoid of a vector.
# =========================================================================
def sigmoid(vector, network_params):
    c = network_params.c
    return 1 / (1 + np.exp(-c * vector))

# =========================================================================
# Compute activation of the next layer given the current one.
# =========================================================================
def compute_activation(input_vector, weight_matrix, network_params):
    output_vector = np.dot(weight_matrix, input_vector)
    output_vector = sigmoid(output_vector, network_params)
    return output_vector

# =========================================================================
# Evaluate the neural network with one hidden layer using sigmoid 
# activation function.
# =========================================================================
def evaluate_neural_network(input_vector, 
                            chromosome, 
                            network_params):

    w_ih, w_ho = decode_chromosome(chromosome, 
                                   network_params.n_i, 
                                   network_params.n_h, 
                                   network_params.n_o, 
                                   network_params.w_max)

    # add 1 on top as a multiplier for the bias see figure A.6 in the 
    # textbook.
    input_vector = np.vstack(([1], input_vector))

    hidden_neuron_vector = compute_activation(input_vector, 
                                              w_ih, network_params)

    # add 1 on top as a multiplier for the bias see figure A.6 in the 
    # textbook.
    hidden_neuron_vector = np.vstack(([1], hidden_neuron_vector))

    output_vector = compute_activation(hidden_neuron_vector, 
                                       w_ho, network_params)
    return output_vector

# =========================================================================
# Decode a chromosome to get weight matrices of the neural network. 
# =========================================================================
def decode_chromosome(chromosome, n_i, n_h, n_o, w_max):
    w_ih = []
    w_ho = []

    gene_counter = 0
    
    #decode w_ih
    for i in range(n_h):
        row = []
        for j in range(n_i + 1):
            gene = chromosome[gene_counter]
            w = w_max * (2 * gene - 1)
            row.append(w)
            gene_counter += 1
        w_ih.append(row)

    # decode w_ho
    for i in range(n_o):
        row = []
        for j in range(n_h + 1):
            gene = chromosome[gene_counter]
            w = w_max * (2 * gene - 1)
            row.append(w)
            gene_counter += 1
        w_ho.append(row)

    return w_ih, w_ho

# =========================================================================
# Initialize population of chromosomes.
# =========================================================================
def initialize_population(training_params, network_params):

    population_size = training_params.population_size
    n_i = network_params.n_i
    n_h = network_params.n_h
    n_o = network_params.n_o
    gene_count = (n_i + 1) * n_h + (n_h + 1) * n_o

    population = [[random.random() for _ in range(gene_count)] \
                for i in range(population_size)]
    
    return population

# =========================================================================
# Compute brake temperature per assignment statement. 
# =========================================================================
def compute_brake_temperature(previous_T_b, P_p, constant_params):

    tau = constant_params.vehicle_params.tau
    C_h = constant_params.vehicle_params.C_h
    T_amb = constant_params.universal_params.T_amb
    dt = constant_params.universal_params.dt

    delta_T_b = previous_T_b - T_amb

    if P_p < 0.01:
        delta_T_b_new = delta_T_b - (delta_T_b / tau) * dt
    else:
        delta_T_b_new = delta_T_b + C_h * P_p * dt

    T_b_new = T_amb + delta_T_b_new

    # brake temp. never falls below ambient.
    if T_b_new < T_amb:
        T_b_new = T_amb

    return T_b_new

# =========================================================================
# Compute foot brake force per assignment statement.
# =========================================================================
def compute_brake_force(P_p, T_b, constant_params):

    M = constant_params.vehicle_params.M
    T_max = constant_params.vehicle_params.T_max
    g = constant_params.universal_params.g

    t1 = M * g / 20
    if T_b < (T_max - 100):
        return t1 * P_p
    else:
        t2 = (T_b - (T_max - 100)) / 100

        return t1 * P_p * np.exp(-t2)

# =========================================================================
# Compute force of gravity.
# =========================================================================
def compute_force_of_gravity(slope, constant_params):
    M = constant_params.vehicle_params.M
    g = constant_params.universal_params.g
    return M * g * np.sin(np.deg2rad(slope))

# =========================================================================
# Compute engine braking force per assignent statement.
# =========================================================================
def compute_engine_brake_force(gear, vehicle_params):

    C_b = vehicle_params.C_b

    match(gear):
        case 1:
            return 7.0 * C_b
        case 2:
            return 5.0 * C_b
        case 3:
            return 4.0 * C_b
        case 4:
            return 3.0 * C_b
        case 5:
            return 2.5 * C_b
        case 6:
            return 2.0 * C_b
        case 7:
            return 1.6 * C_b
        case 8:
            return 1.4 * C_b
        case 9:
            return 1.2 * C_b
        case 10:
            return C_b

# =========================================================================
# Compute next gear given the current gear, time since the last gear 
# change and the network output delta_gear.
# =========================================================================
def compute_next_gear(current_gear, time_since_gear_change, 
                      delta_gear):

    if time_since_gear_change >= 2:
        if delta_gear > 0.7:
            if current_gear < 10:
                next_gear = current_gear + 1
                time_since_gear_change = 0
            else:
                next_gear = current_gear

        elif delta_gear < 0.3:
            if current_gear > 1:
                next_gear = current_gear - 1
                time_since_gear_change = 0
            else:
                next_gear = current_gear

        else:
            next_gear = current_gear

    else:
        next_gear = current_gear

    return next_gear, time_since_gear_change

# =========================================================================
# Compute vehicle acceleration per assignment statement.
# =========================================================================
def compute_vehicle_acceleration(P_p, T_b, slope, gear, constant_params):

    F_g = compute_force_of_gravity(slope, constant_params)
    F_b = compute_brake_force(P_p, T_b, constant_params)
    F_eb = compute_engine_brake_force(gear, constant_params.vehicle_params)
    M = constant_params.vehicle_params.M

    acceleration = (F_g - F_b - F_eb) / M

    return acceleration

# =========================================================================
# Initialize the first entry in the time series of the vehicle state.
# =========================================================================
def init_vehicle_state(slope_index, dataset_index, constant_params):

    # starting time.
    t_0 = 0

    start_condition = constant_params.start_condition

    # starting brake temperature.
    T_b0 = start_condition.T_b0

    # starting slope.
    slope_0 = get_slope_angle(start_condition.x_0, slope_index, 
                              dataset_index)

    # staring gear.
    gear_0 = start_condition.gear_0

    # starting speed.
    v_0 = start_condition.v_0

    # starting distance.
    x_0 = start_condition.x_0

    # starting acceleration.
    P_p0 = 0
    a_0 = compute_vehicle_acceleration(P_p0, T_b0, slope_0, gear_0,
                                       constant_params)

    # initial state of the vehicle.
    vehicle_state = \
            Vehicle_State(time = [t_0],
                          brake_pressure = [P_p0],
                          brake_temperature = [T_b0],
                          slope = [slope_0],
                          gear = [gear_0],
                          distance = [x_0],
                          speed = [v_0],
                          acceleration = [a_0],
                          time_since_gear_change = 1000)

    return vehicle_state

# =========================================================================
# Update the input vector to the neural network.
# =========================================================================
def update_input_vector(vehicle_state, constant_params):

    speed = vehicle_state.speed[-1]
    slope = vehicle_state.slope[-1]
    T_b = vehicle_state.brake_temperature[-1]

    v_max = constant_params.vehicle_params.v_max
    T_max = constant_params.vehicle_params.T_max
    slope_max = constant_params.universal_params.slope_max

    input_vector = np.zeros((constant_params.network_params.n_i, 1))
    input_vector[0][0] = speed / v_max
    input_vector[1][0] = slope / slope_max
    input_vector[2][0] = T_b / T_max

    return input_vector

# =========================================================================
# Compute current speed of the vehicle after a time step.
# =========================================================================
def compute_current_speed(vehicle_state, dt):
    previous_speed = vehicle_state.speed[-1]
    previous_acceleration = vehicle_state.acceleration[-1]
    current_speed = previous_speed + previous_acceleration * dt

    if current_speed < 0:
        current_speed = 0
        if previous_acceleration < 0:
            vehicle_state.acceleration[-1] = 0

    return current_speed

# =========================================================================
# Compute current distance of the vehicle after a time step.
# =========================================================================
def compute_current_distance(vehicle_state, dt):
    previous_distance = vehicle_state.distance[-1]
    previous_speed = vehicle_state.speed[-1]
    previous_acceleration = vehicle_state.acceleration[-1]
    current_distance = previous_distance + \
            previous_speed * dt + \
            0.5 * previous_acceleration * (dt ** 2)
    return current_distance

# =========================================================================
# Update the vehicle state after each time step.
# =========================================================================
def update_vehicle_state(vehicle_state, slope_index, dataset_index, 
                         output_vector, constant_params):

    # neural network outputs.
    P_p = output_vector[0][0]
    delta_gear = output_vector[1][0]

    dt = constant_params.universal_params.dt

    # update time.
    previous_time = vehicle_state.time[-1]
    current_time = previous_time + dt

    # update vehice speed and distance.
    current_speed = compute_current_speed(vehicle_state, dt)
    current_distance = compute_current_distance(vehicle_state, dt)

    # update slope.
    current_slope = get_slope_angle(current_distance, slope_index, 
                                    dataset_index)

    # update brake temperature.
    previous_T_b = vehicle_state.brake_temperature[-1]
    current_T_b = compute_brake_temperature(previous_T_b, P_p,
                                            constant_params)

    # update gear.
    previous_gear = vehicle_state.gear[-1]
    time_since_gear_change = vehicle_state.time_since_gear_change + dt
    current_gear, time_since_gear_change = \
            compute_next_gear(previous_gear, time_since_gear_change, 
                              delta_gear)

    # update acceleration of the vehicle.
    current_acceleration = compute_vehicle_acceleration(P_p, current_T_b, 
                                                        current_slope, 
                                                        current_gear, 
                                                        constant_params)

    # append latest data to the vehicle state time series.
    vehicle_state.time.append(current_time)
    vehicle_state.gear.append(current_gear)
    vehicle_state.brake_pressure.append(P_p)
    vehicle_state.brake_temperature.append(current_T_b)
    vehicle_state.distance.append(current_distance)
    vehicle_state.speed.append(current_speed)
    vehicle_state.acceleration.append(current_acceleration)
    vehicle_state.slope.append(current_slope)
    vehicle_state.time_since_gear_change = time_since_gear_change

    return vehicle_state

# =========================================================================
# Evaluate indvidual
# =========================================================================
def evaluate_individual(chromosome, slope_index, dataset_index, 
                        constant_params, constrained, is_test):

    # Vehicle start condition initialization.
    vehicle_state = init_vehicle_state(slope_index, dataset_index, 
                                       constant_params)

    # Input vector for the neural network, initial.
    input_vector = update_input_vector(vehicle_state, constant_params) 

    # Max. distance the vehicle shall run.
    target_distance = constant_params.universal_params.target_distance

    # Vehicle run simulation loop.
    while True:

        # compute network output for this time slice.
        output = evaluate_neural_network(input_vector,
                                         chromosome,
                                         constant_params.network_params)

        # update vehicle state.
        vehicle_state = update_vehicle_state(vehicle_state, slope_index, 
                                             dataset_index, output,
                                             constant_params)

        current_speed = vehicle_state.speed[-1]
        current_brake_temperature = vehicle_state.brake_temperature[-1]

        max_speed = constant_params.vehicle_params.v_max
        min_speed = constant_params.vehicle_params.v_min
        max_temp = constant_params.vehicle_params.T_max
        ambient_temp = constant_params.universal_params.T_amb

        current_distance = vehicle_state.distance[-1]
        if current_distance >= target_distance:
            break

        # check constraint violation.
        if constrained:
            if current_speed > max_speed:
                break

            if current_speed < min_speed:
                break

            if current_brake_temperature > max_temp:
                break

        # vehicle stopped, exit loop.
        if current_speed == 0:
            break

        # update input vector for next iteration.
        input_vector = update_input_vector(vehicle_state, constant_params) 

    if not constrained:
        fitness = compute_fitness_training(vehicle_state, 
                                           constant_params)
    else:
        fitness = compute_fitness_validation(vehicle_state, 
                                             constant_params)

    if is_test:
        return fitness, vehicle_state
    else:
        return fitness

# =========================================================================
# Cross two chromosomes using averaging crossover.
# =========================================================================
def cross(chromosome1, chromosome2):

    number_of_genes = len(chromosome1)

    new_chromosome_1 = []
    new_chromosome_2 = []

    for gene_index in range(number_of_genes):
        a = random.random()
        gene_1 = a * chromosome1[gene_index] + \
                (1 - a) * chromosome2[gene_index]
        gene_2 = a * chromosome2[gene_index] + \
                (1 - a) * chromosome1[gene_index]

        new_chromosome_1.append(gene_1)
        new_chromosome_2.append(gene_2)

    return [new_chromosome_1, new_chromosome_2]

# =========================================================================
# Select individuals using tournament selection.
# =========================================================================
def tournament_select(fitness_list, training_params):

    tournament_size = training_params.tournament_size
    tournament_probability = training_params.tournament_probability
    population_size = len(fitness_list)

    random_fitness_list = []
    for i in range(tournament_size):
        random_index = random.randint(0, population_size - 1)
        fitness_value = fitness_list[random_index]
        random_fitness_list.append((fitness_value, random_index))

    random_fitness_list.sort(key=lambda x:x[0], reverse=True)

    while(len(random_fitness_list) > 1):
        r = random.random()
        if r < tournament_probability:
            return random_fitness_list[0][1]
        else:
            random_fitness_list.pop(0)

    return random_fitness_list[0][1]

# =========================================================================
# Mutate individuals using creep mutation.
# =========================================================================
def mutate(chromosome, training_params, network_params):

    mutation_probability = training_params.mutation_probability
    C_r = training_params.C_r
    w_max = network_params.w_max

    number_of_genes = len(chromosome)
    mutated_chromosome = chromosome.copy()

    for gene_index in range(number_of_genes):
        r = random.random()
        if r < mutation_probability:
            q = random.random()
            gene = mutated_chromosome[gene_index]
            mutated_gene = gene - C_r / 2 + C_r * q
            if mutated_gene > w_max:
                mutated_gene = w_max
            if mutated_gene < -w_max:
                mutated_gene = -w_max
            mutated_chromosome[gene_index] = mutated_gene

    return mutated_chromosome

# =========================================================================
# Genetic algorithm
# =========================================================================
def run_function_optimization(constant_params, training_params):

    population = initialize_population(training_params, 
                                       constant_params.network_params)
    population_size = training_params.population_size

    best_ever_chromosome = []
    least_validation_error = 10000000

    generation_list = []
    test_fitness_list = []
    validation_fitness_list = []

    for generation_index in range(training_params.number_of_generations):

        if least_validation_error < 5000:
            break

        generation_list.append(generation_index + 1)
        
        maximum_fitness = 0
        best_chromosome = []
        fitness_list = []
        best_validation_fitness = 0

        training_dataset_index = 1
        validation_dataset_index = 2
        test_dataset_index = 3
        slope_indices = [_ for _ in range(1,11)]

        print(f"Fitness, Gen {generation_index + 1}: ")
        for chromosome in population:

            average_fitness = 0
            random.shuffle(slope_indices)
            for slope_index in slope_indices:

                # sum the fitness over the training dataset
                fitness = evaluate_individual(chromosome, 
                                              slope_index, 
                                              training_dataset_index, 
                                              constant_params,
                                              False, False)

                average_fitness += fitness

            # Avg. fitness over the training dataset
            average_fitness /= 10
            if average_fitness > maximum_fitness:
                maximum_fitness = average_fitness
                best_chromosome = chromosome.copy()

                # run on validation set
                validation_fitness, validation_error = validate_network(best_chromosome, 
                                                                        validation_dataset_index, 
                                                                        constant_params)

                if validation_fitness > best_validation_fitness:
                    best_validation_fitness = validation_fitness

                # track validation error and the corresponding chromosome.
                if validation_error < least_validation_error:
                    least_validation_error = validation_error
                    best_ever_chromosome = chromosome.copy()

                # run on test set
                test_fitness, test_error = validate_network(best_chromosome, 
                                                            test_dataset_index, 
                                                            constant_params)

                print(f"        {maximum_fitness:.2f}, Validation Error={validation_error:.2f}, Test Error={test_error:.2f}")

            fitness_list.append(average_fitness)

        test_fitness_list.append(maximum_fitness)
        validation_fitness_list.append(best_validation_fitness)

        temp_population = []
        for i in range(0, population_size, 2):
            index_1 = tournament_select(fitness_list, training_params)
            index_2 = tournament_select(fitness_list, training_params)

            chromosome1 = population[index_1].copy()
            chromosome2 = population[index_2].copy()
            r = random.random()

            if r < training_params.crossover_probability:
                [new_chromosome_1, new_chromosome_2] = cross(chromosome1,
                                                             chromosome2)
                temp_population.append(new_chromosome_1)
                temp_population.append(new_chromosome_2) 
            else:
                temp_population.append(chromosome1)
                temp_population.append(chromosome2)

        for i in range(population_size):
            original_chromosome = temp_population[i]
            mutated_chromosome = mutate(original_chromosome, 
                                        training_params,
                                        constant_params.network_params)
            temp_population[i] = mutated_chromosome

        if best_chromosome != []:
            temp_population[0] = best_chromosome
        population = temp_population.copy()

    # Plot the generation vs. fitness plot.
    plt.plot(generation_list, test_fitness_list, linestyle="-", label="Training Fitness")
    plt.plot(generation_list, validation_fitness_list, linestyle="--", label="Validation Fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.legend()
    plt.savefig("fitness.pdf", bbox_inches="tight")
    plt.xticks(range(min(generation_list), max(generation_list)+1))
    plt.show()

    # done.
    return best_ever_chromosome

# =========================================================================
# Run the vehicle dynamics on a dataset. 
# =========================================================================
def run_test(chromosome, slope_index, dataset_index):

    fitness, vehicle_state = evaluate_individual(chromosome, 
                                                 slope_index, 
                                                 dataset_index, 
                                                 constant_params,
                                                 True, True)

    plot(vehicle_state)
    return

# =========================================================================
# Main program.
# =========================================================================
@dataclass(frozen=True)
class Network_Parameters:
    n_i:    int
    n_h:    int
    n_o:    int
    c:      float
    w_max:  float

@dataclass(frozen=True)
class Vehicle_Parameters:
    M:          float
    T_max:      float
    tau:        float
    C_h:        float
    C_b:        float
    v_max:      float
    v_min:      float

@dataclass(frozen=True)
class Universal_Parameters:
    g:                  float
    slope_max:          float
    T_amb:              float
    dt:                 float
    target_distance:    float

@dataclass(frozen=True)
class Start_Condition:
    x_0:    float
    v_0:    float
    T_b0:   float
    gear_0: int

@dataclass
class Training_Parameters:
    population_size:        int
    number_of_generations:  int
    tournament_size:        int
    tournament_probability: float
    crossover_probability:  float
    mutation_probability:   float
    C_r:                    float

@dataclass
class Vehicle_State:
    time:                       list
    gear:                       list
    slope:                      list
    brake_pressure:             list
    brake_temperature:          list
    distance:                   list
    speed:                      list
    acceleration:               list
    time_since_gear_change:     float

@dataclass
class Constant_Parameters:
    network_params:             Network_Parameters
    vehicle_params:             Vehicle_Parameters
    universal_params:           Universal_Parameters
    start_condition:            Start_Condition

# Initialization
network_params = Network_Parameters(n_i = 3, 
                                    n_h = 10, 
                                    n_o = 2, 
                                    c = 1, 
                                    w_max = 8)

vehicle_params = Vehicle_Parameters(M = 20000, 
                                    T_max = 750, 
                                    tau = 30, 
                                    C_h = 40, 
                                    C_b = 3000, 
                                    v_max = 25, 
                                    v_min = 1)

training_params = Training_Parameters(population_size = 200, 
                                      number_of_generations = 100, 
                                      tournament_size = 7, 
                                      tournament_probability = 0.7, 
                                      crossover_probability = 0.7, 
                                      mutation_probability = 0.3,
                                      C_r = 1)

universal_params = Universal_Parameters(g = 9.80665, 
                                        slope_max = 10, 
                                        T_amb = 283,
                                        dt = 0.05,
                                        target_distance = 1000)

start_condition = Start_Condition(x_0 = 0, 
                                  v_0 = 20, 
                                  T_b0 = 500, 
                                  gear_0 = 7)

constant_params = Constant_Parameters(network_params = network_params,
                                      universal_params = universal_params,
                                      start_condition = start_condition,
                                      vehicle_params = vehicle_params)


output_filename = "best_chromosome.py"
try:
    with open(output_filename, "r") as f:
        print("", end="")
except FileNotFoundError:
    best_chromosome = run_function_optimization(constant_params, training_params)
    with open(output_filename, "w") as f:
        f.write(f"best_chromosome = {best_chromosome}\n")
