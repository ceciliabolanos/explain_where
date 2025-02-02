from explainers.random_forest import RFExplainer
from explainers.linear_regression import LRExplainer
from explainers.naive import NaiveExplainer

from data.mask_windows import WindowMaskingDataGenerator
from data.base_generator import MaskingConfig

from models.ast.model import ASTModel
# from models.yamnet.model import YAMNetModel

import json
from pathlib import Path

PATH = '/home/ec2-user/results1/explanations_audioset'

def generate_explanation(filename: str, 
                  model_name: str, 
                  id_to_explain: int,
                  config: MaskingConfig):
    
    if model_name == 'ast':
       model = ASTModel(filename, id_to_explain) 

    input, real_score_id = model.process_input()     
    predict_fn = model.get_predict_fn()   
    
    ############## Generate data ##############
    
    data_generator = WindowMaskingDataGenerator(
            model_name=model_name,
            audio=input,
            sample_rate=16000, 
            mask_config=config,
            predict_fn=predict_fn, 
        )
    
    # Generate masked data
    data_generator.mode = 'naive_masked'
    data_generator.generate(filename)
    
    data_generator.mode = 'all_masked'
    data_generator.generate(filename)

    ########## Generate the importances for each method ##########

    # Naive analysis
    print('Running Naive analysis...')
    naive_analyzer = NaiveExplainer(
        path= Path(PATH) / filename / model_name / f"scores_w{config.window_size}_m{config.mask_type}.json",
        filename=filename
    )
    importances_naive = naive_analyzer.get_feature_importance(label_to_explain=id_to_explain)

    # Random Forest analysis    
    print('Running Random Forest analysis...')
    rf_analyzer = RFExplainer(
        path= Path(PATH) / filename / model_name / f"scores_p{config.mask_percentage}_w{config.window_size}_f{config.function}_m{config.mask_type}.json",
        filename=filename,
    )
    importances_rf_tree = rf_analyzer.get_feature_importances(label_to_explain=id_to_explain, method='tree')
    importances_rf_shap = rf_analyzer.get_feature_importances(label_to_explain=id_to_explain, method='shap')

    # LR analysis
    print('Running Linear Regression analysis...')
    lime_analyzer = LRExplainer(
        Path(PATH) / filename / model_name / f"scores_p{config.mask_percentage}_w{config.window_size}_f{config.function}_m{config.mask_type}.json",
        verbose=False,
        absolute_feature_sort=False
    )
    importances_lime = lime_analyzer.explain_instance(
        label_to_explain=id_to_explain
    ).get_feature_importances(label=id_to_explain)
    
    ############## Prepare output ##############
    output_data = {
        "metadata": {
            "filename": filename,
            "id_explained": float(id_to_explain),
            "widow_size": config.window_size,
            "num_samples": config.num_samples,
            "mask_percentages": config.mask_percentage,
            "segment_length": config.segment_length,
        },
        "importance_scores": {
            "naive": {
                "method": "NaiveAudioAnalyzer",
                "values": importances_naive.tolist() if hasattr(importances_naive, 'tolist') else importances_naive
            },
            "random_forest": {
                "tree_importance": {
                    "method": "masked rf with tree importance",
                    "values": importances_rf_tree.tolist() if hasattr(importances_rf_tree, 'tolist') else importances_rf_tree
                },
                "shap": {
                    "method": "masked rf with shap importance",
                    "values": importances_rf_shap.tolist() if hasattr(importances_rf_shap, 'tolist') else importances_rf_shap
                }
            },
            "linear_regression": {
                "masked": {
                    "method": "Linear Regression with masking",
                    "values": importances_lime.tolist() if hasattr(importances_lime, 'tolist') else importances_lime
                }
            }
        }
    }

    # Save results
    output_path = Path(PATH) / filename / model_name / f"ft_{id_to_explain}_p{config.mask_percentage}_w{config.window_size}_f{config.function}_m{config.mask_type}.json"
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Importance scores saved to: {output_path}")

    return output_data