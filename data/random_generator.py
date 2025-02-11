import json
import random
from pathlib import Path

class RandomDataGenerator:
    def __init__(self, path, model_name, windows, functions, mask_types, mask_percentages, num_samples):
        self.path = Path(path)
        self.possible_windows = windows[1, 2, 3, 4, 5]
        self.possible_functions = functions ["euclidean"]
        self.possible_mask_types = mask_types ["zeros", "stat", "noise"]
        self.possible_mask_percentages = mask_percentages[0.1, 0.2, 0.3, 0.4]
        self.num_samples = num_samples
        self.filename
        self.model_name = model_name

    def process_random_file(self):
        # Step 1: Get list of all JSON files
        scores = []
        neighborhood = []
        snrs = []
        
        for i in range(self.num_samples):
            window_size = random.choice(self.possible_windows)
            mask_type = random.choice(self.possible_mask_types)
            mask_percentage = random.choice(self.possible_mask_percentages)

            chosen_file = (
                Path(self.path) / self.filename / self.model_name /
                f"scores_p{mask_percentage}_w{window_size}_f{self.possible_functions}_m{mask_type}.json"
            )

            # Step 3: Load the chosen file
            with open(chosen_file, "r") as f:
                data = json.load(f)

            # Step 4: Select a random SNR
            possible_snrs = data.get("snrs", [])
            random_index = random.randrange(len(possible_snrs))  # Get a random index
            
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

        # Step 6: Save the new file
        output_file = (
                Path(self.path) / self.filename / self.model_name /
                f"scores_p{mask_percentage}_w{window_size}_f{self.function}_m{mask_type}.json"
            )
        output_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

        with open(output_file, "w") as f:
            json.dump(data_to_save, f, indent=4)

        print(f"New file saved at: {output_file}")

