from explainers.random_forest import RFExplainer
from explainers.linear_regression import LRExplainer
from explainers.naive import NaiveExplainer
from explainers.kernel_shap import KernelShapExplainer
from explainers.greedy import GreedyExplainer

from data.mask_windows import WindowMaskingDataGenerator
from data.base_generator import MaskingConfig

from models.ast.model import ASTModel

from utils import get_segments
import os
import json
from pathlib import Path

def generate_explanation(filename: str, 
                  model_name: str, 
                  id_to_explain: int,
                  config: MaskingConfig, 
                  path: str):
    
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
            filename=filename,
            id_to_explain=id_to_explain, 
            path=path
        )

    data_generator.mode = 'naive_masked'
    if not os.path.exists(Path(path) / filename / model_name / f"scores_w1_m{config.mask_type}.json"):
        data_generator.generate(filename)

    # data_generator.mode = 'greedy_masked'
    # if not os.path.exists(Path(PATH) / filename / model_name / f"scores_w1_m{config.mask_type}_greedy.json"):
    #     data_generator.generate(filename)

    data_generator.mode = 'all_masked'
    if not os.path.exists(Path(path) / filename / model_name / f"scores_p{config.mask_percentage}_w{config.window_size}_f{config.function}_m{config.mask_type}.json"):
        data_generator.generate(filename)

    ########## Generate the importances for each method ##########

    # # Naive analysis
    # print('Running Naive analysis...')
    # naive_analyzer = NaiveExplainer(
    #     path= Path(path) / filename / model_name / f"scores_w1_m{config.mask_type}.json",
    #     filename=filename
    # )
    # importances_naive = naive_analyzer.get_feature_importance(label_to_explain=id_to_explain)

    # print('Running Greedy analysis...')
    # greedy_analyzer = GreedyExplainer(
    #     path= Path(path) / filename / model_name / f"scores_w1_m{config.mask_type}_greedy.json",
    #     filename=filename
    # )
    # importances_greedy = greedy_analyzer.get_feature_importance()

    # print('Running Kernel Shap...')
    # kernelshap_analyzer = KernelShapExplainer(
    #     path= Path(path) / filename / model_name / f"scores_p{config.mask_percentage}_w{config.window_size}_f{config.function}_m{config.mask_type}.json",
    # )
    # importances_kernelshap = kernelshap_analyzer.explain_instance(
    #     label_to_explain=id_to_explain
    # ).get_feature_importances()

    # # Random Forest analysis    
    # print('Running Random Forest analysis...')
    # rf_analyzer = RFExplainer(
    #     path= Path(path) / filename / model_name / f"scores_p{config.mask_percentage}_w{config.window_size}_f{config.function}_m{config.mask_type}.json",
    #     filename=filename,
    # )
    # importances_rf_tree = rf_analyzer.get_feature_importances(label_to_explain=id_to_explain, method='tree')
    # importances_rf_shap = rf_analyzer.get_feature_importances(label_to_explain=id_to_explain, method='shap')

    # # LR analysis
    # print('Running Linear Regression analysis...')
    # lime_analyzer = LRExplainer(
    #     Path(path) / filename / model_name / f"scores_p{config.mask_percentage}_w{config.window_size}_f{config.function}_m{config.mask_type}.json",
    #     verbose=False,
    #     absolute_feature_sort=False
    # )
    # importances_lime = lime_analyzer.explain_instance(
    #     label_to_explain=id_to_explain
    # ).get_feature_importances()
    
    # true_markers = get_segments(filename, id_to_explain)

    # ############## Prepare output ##############
    # output_data = {
    #     "metadata": {
    #         "filename": filename,
    #         "id_explained": float(id_to_explain),
    #         "widow_size": config.window_size,
    #         "num_samples": config.num_samples,
    #         "mask_percentages": config.mask_percentage,
    #         "segment_length": config.segment_length,
    #         "true_markers": true_markers,
    #         "true_score": real_score_id
    #     },
    #     "importance_scores": {
    #         "naive": {
    #             "method": "NaiveAudioAnalyzer",
    #             "values": importances_naive.tolist() if hasattr(importances_naive, 'tolist') else importances_naive
    #         },
    #         'greedy': {
    #             'method': 'GreedyAudioAnalyzer',
    #             'values': importances_greedy.tolist() if hasattr(importances_greedy, 'tolist') else importances_greedy
    #         },
    #         "random_forest_tree_importance": {
    #                 "method": "masked rf with tree importance",
    #                 "values": importances_rf_tree.tolist() if hasattr(importances_rf_tree, 'tolist') else importances_rf_tree
    #         },
    #         "random_forest_shap_importance": {
    #             "method": "masked rf with shap importance",
    #             "values": importances_rf_shap.tolist() if hasattr(importances_rf_shap, 'tolist') else importances_rf_shap
    #         },
    #         "linear_regression": {
    #             "method": "Linear Regression with kernel as pi",
    #             "values": importances_lime.tolist() if hasattr(importances_lime, 'tolist') else importances_lime
    #         },
    #         "kernel_shap": {
    #             "method": "Linear Regression with shap as pi",
    #             "values": importances_kernelshap.tolist() if hasattr(importances_kernelshap, 'tolist') else importances_kernelshap  
    #         }
    #     }
    # }

    # # Save results
    # output_path = Path(path) / filename / model_name / f"ft_{id_to_explain}_p{config.mask_percentage}_w{config.window_size}_f{config.function}_m{config.mask_type}.json"
    # with open(output_path, 'w') as f:
    #     json.dump(output_data, f, indent=2)

    # print(f"Importance scores saved to: {output_path}")

    return 