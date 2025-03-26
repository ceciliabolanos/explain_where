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
        self.stds_noise = config[self.name]["possible_std_noise"]

    def process_random_file(self):
        filename = os.path.basename(self.filename)
        output_file = (
            Path(self.path) / filename / self.model_name /
            f"scores_{self.name}_samples{self.num_samples}.json"
        )

        if os.path.exists(output_file):
            return

        all_combinations = list(product(
            self.possible_mask_types,
            self.possible_mask_percentages,
            self.possible_windows,
            self.possible_functions,
            self.stds_noise
        ))

        samples = []
        for combination in all_combinations:
            mask_type, mask_percentage, window_size, function, std_noise = combination
            samples.append((mask_type, mask_percentage, window_size, function, std_noise))

        random.shuffle(samples)

        scores = []
        neighborhood = []
        snrs = []
        # cambiar para que no sea predeterminado
        chosen_file = (
                Path(self.path) / filename / self.model_name /
                f"scores_p0.4_w3_feuclidean_mnoise.json"
            )
        with open(chosen_file, "r") as f:
            data = json.load(f)
        snrs.append(data["snrs"][0])
        scores.append(data["scores"][0])
        neighborhood.append(data["neighborhood"][0])

        for mask_type, mask_percentage, window_size, function, std_noise in samples:
            if mask_type == "noise":
                chosen_file = (
                    Path(self.path) / filename / self.model_name /
                    f"scores_p{mask_percentage}_w{window_size}_f{function}_m{mask_type}_a{std_noise}.json"
                )
            else:
                chosen_file = (
                    Path(self.path) / filename / self.model_name /
                    f"scores_p{mask_percentage}_w{window_size}_f{function}_m{mask_type}.json"
                )

            with open(chosen_file, "r") as f:
                data = json.load(f)

            possible_snrs = data.get("snrs", [])

            for i in range(self.num_for_each):
                random_index = random.randint(1, len(possible_snrs) - 1)  

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

        output_file.parent.mkdir(parents=True, exist_ok=True)  
        with open(output_file, "w") as f:
            json.dump(data_to_save, f, indent=4)

        print(f"New file saved at: {output_file}")

