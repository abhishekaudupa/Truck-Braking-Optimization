from best_chromosome import best_chromosome
from run_ffnn_optimization import run_test
import random

slope_index = random.randint(1, 5)
dataset_index = 3   #Test dataset.

match(dataset_index):
    case 1:
        dataset = "Training dataset"
    case 2:
        dataset = "Validation dataset"
    case 3:
        dataset = "Test dataset"
    case _:
        dataset = "Unknown dataset"

print(f"Running the network on the {dataset}, slope index {slope_index}")
run_test(best_chromosome, slope_index, dataset_index)
