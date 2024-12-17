import json 

class NaiveAudioAnalyzer:
    """
    A class for analyzing audio importance through masking techniques.
    
    This class implements a naive approach to determine feature importance
    in audio by systematically masking portions of the audio and measuring
    the effect on model predictions.
    """
    
    def __init__(self, path, filename):
        self.path = path
        self.filename = filename

        
    def get_feature_importance(self, label_to_explain):
        with open(self.path, 'r') as file:
            data = json.load(file)

        return self.distance_to_gt(data['scores'], data['score_real'], label_to_explain)
        

    def distance_to_gt(self, masked_results, real_results, real_predicted_class):
        gt_prob = real_results[0][real_predicted_class]
        gt_probs = [prob[0][real_predicted_class] for prob in masked_results]

        # Calculate the distances between the GT probability and the probabilities for other classes
        distances = []
        for prob in gt_probs:
            distances.append(gt_prob - prob)

        return distances    
