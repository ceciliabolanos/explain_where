import json
import random
from pathlib import Path
import os
from itertools import product
import math

class RandomDataGenerator:
    def __init__(self, path, model_name, filename, config, num_samples):
        self.path = Path(path)
        self.possible_functions = ['euclidean'] 
        self.num_samples = num_samples
        self.filename = filename
        self.model_name = model_name
        self.name = list(config.keys())[0]

        self.possible_mask_types = config[self.name]["possible_mask_types"]
        self.possible_mask_percentages = config[self.name]["mask_percentage"]
        self.possible_windows = config[self.name]["possible_windows"]
        self.num_for_each = math.ceil(self.num_samples / config[self.name]["combinations"])

    def process_random_file(self):
        # Step 1: Get list of all JSON files
        filename = os.path.basename(self.filename)
        output_file = (
            Path(self.path) / filename / self.model_name /
            f"scores_{self.name}.json"
        )
        if os.path.exists(output_file):
            return

        # Step 2: Generate all possible combinations
        all_combinations = list(product(
            self.possible_mask_types,
            self.possible_mask_percentages,
            self.possible_windows,
            self.possible_functions
        ))

        # Step 3: Sample each combination `num_for_each` times
        samples = []
        for combination in all_combinations:
            mask_type, mask_percentage, window_size, function = combination
            samples.append((mask_type, mask_percentage, window_size, function))

        # Step 4: Randomize the order of samples
        random.shuffle(samples)

        # Step 5: Process each sample
        scores = []
        neighborhood = []
        snrs = []

        for mask_type, mask_percentage, window_size, function in samples:
            chosen_file = (
                Path(self.path) / filename / self.model_name /
                f"scores_p{mask_percentage}_w{window_size}_f{function}_m{mask_type}.json"
            )

            # Step 6: Load the chosen file
            with open(chosen_file, "r") as f:
                data = json.load(f)

            # Step 7: Select a random SNR
            possible_snrs = data.get("snrs", [])

            for i in range(self.num_for_each):
                random_index = random.randint(1, len(possible_snrs) - 1)  # Get a random index

                snrs.append(possible_snrs[random_index])
                scores.append(data["scores"][random_index])
                neighborhood.append(data["neighborhood"][random_index])
                score_real = data["score_real"]

        data_to_save = {
            "scores": scores,
            "neighborhood": neighborhood,
            "score_real": score_real,
            "snrs": snrs
        }

        # Step 8: Save the new file
        output_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

        with open(output_file, "w") as f:
            json.dump(data_to_save, f, indent=4)

        print(f"New file saved at: {output_file}")

