from explainers.random_forest import RFExplainer
from explainers.linear_regression import LRExplainer
from explainers.linear_regression_noconstrain import LRnoconExplainer
from explainers.utils import compute_log_odds

from explainers.kernel_shap import KernelShapExplainer
from explainers.kernel_shap_1constrain import KernelShapExplainer1constraint

from data.mask_windows import WindowMaskingDataGenerator
from data.base_generator import MaskingConfig

from models.ast.model import ASTModel
from models.cough.model import CoughModel
from models.drums.model import DrumsModel
from models.kws.model import KWSModel

import numpy as np
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
       complete_filename = filename

    if model_name == 'cough':
       model = CoughModel(filename, id_to_explain) 
       complete_filename = filename
       filename = os.path.basename(filename)

    if model_name == 'drums':
       model = DrumsModel(filename, id_to_explain) 
       complete_filename = filename
       filename = os.path.basename(filename)
    
    if model_name == 'kws':
       model = KWSModel(filename, id_to_explain) 
       complete_filename = filename
       filename = os.path.basename(filename)
    
    input, outputs = model.process_input()
    real_score_id = outputs[id_to_explain]      
    predict_fn = model.get_predict_fn()   

    pred_zeros = predict_fn([np.zeros(len(input))])
    if model_name == 'drums':
        empty_score = compute_log_odds(pred_zeros, id_to_explain)[0]
    else:
        empty_score = pred_zeros[0][id_to_explain]

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

    data_generator.mode = 'all_masked'
    if not os.path.exists(Path(path) / filename / model_name / f"scores_p{config.mask_percentage}_w{config.window_size}_f{config.function}_m{config.mask_type}.json"):
        data_generator.generate(filename)

    ######### Generate the importances for each method ##########
    output_path = Path(path) / filename / model_name / f"ft_{id_to_explain}_p{config.mask_percentage}_w{config.window_size}_f{config.function}_m{config.mask_type}.json"

    if os.path.exists(output_path):
        return 

    kernelshap_analyzer = KernelShapExplainer(
        path= Path(path) / filename / model_name / f"scores_p{config.mask_percentage}_w{config.window_size}_f{config.function}_m{config.mask_type}.json",
    )
    importances_kernelshap = kernelshap_analyzer.explain_instance(
        label_to_explain=id_to_explain, empty_score=empty_score, model=model_name).get_feature_importances()

    kernelshap_analyzer_1constraint = KernelShapExplainer1constraint(
        path= Path(path) / filename / model_name / f"scores_p{config.mask_percentage}_w{config.window_size}_f{config.function}_m{config.mask_type}.json",
    )
    importances_kernelshap_analyzer_1constraint = kernelshap_analyzer_1constraint.explain_instance(
        label_to_explain=id_to_explain, empty_score=empty_score, model=model_name).get_feature_importances()

    rf_analyzer = RFExplainer(
        path= Path(path) / filename / model_name / f"scores_p{config.mask_percentage}_w{config.window_size}_f{config.function}_m{config.mask_type}.json",
        filename=filename,
    )
    importances_rf_tree = rf_analyzer.get_feature_importances(label_to_explain=id_to_explain, method='tree', model=model_name)

    # LR analysis
    lime_nocon_analyzer = LRnoconExplainer(
        Path(path) / filename / model_name / f"scores_p{config.mask_percentage}_w{config.window_size}_f{config.function}_m{config.mask_type}.json",
        verbose=False,
        absolute_feature_sort=False
    )
    importances_nocon = lime_nocon_analyzer.explain_instance(
        label_to_explain=id_to_explain, model=model_name).get_feature_importances()
    
    lime_analyzer = LRExplainer(
        Path(path) / filename / model_name / f"scores_p{config.mask_percentage}_w{config.window_size}_f{config.function}_m{config.mask_type}.json",
        verbose=False,
        absolute_feature_sort=False
    )
    importances_lime = lime_analyzer.explain_instance(
        label_to_explain=id_to_explain, model=model_name).get_feature_importances()
    
    true_markers = get_segments(complete_filename, id_to_explain, model_name)
    ############## Prepare output ##############
    output_data = {
        "metadata": {
            "filename": filename,
            "id_explained": float(id_to_explain),
            "widow_size": config.window_size,
            "num_samples": config.num_samples,
            "mask_percentages": config.mask_percentage,
            "segment_length": config.segment_length,
            "true_markers": true_markers,
            "true_score": real_score_id
        },
        "importance_scores": {
            "importances_kernelshap_analyzer_1constraint": {
                "method": "NaiveAudioAnalyzer",
                "values": importances_kernelshap_analyzer_1constraint.tolist() if hasattr(importances_kernelshap_analyzer_1constraint, 'tolist') else importances_kernelshap_analyzer_1constraint
            },
            "random_forest_tree_importance": {
                    "method": "masked rf with tree importance",
                    "values": importances_rf_tree.tolist() if hasattr(importances_rf_tree, 'tolist') else importances_rf_tree
            },
            "linear_regression": {
                "method": "Linear Regression with kernel as pi",
                "values": importances_lime.tolist() if hasattr(importances_lime, 'tolist') else importances_lime
            },
            "linear_regression_nocon": {
                "method": "Linear Regression with kernel as pi",
                "values": importances_nocon.tolist() if hasattr(importances_nocon, 'tolist') else importances_nocon
            },
            "kernel_shap": {
                "method": "Linear Regression with shap as pi",
                "values": importances_kernelshap.tolist() if hasattr(importances_kernelshap, 'tolist') else importances_kernelshap  
            }
        }
    }
   
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Importance scores saved to: {output_path}")

    return 


def generate_explanation_from_file(filename: str, 
                  model_name: str, 
                  id_to_explain: int,
                  name: str,
                  path: str):
    
    if model_name == 'ast':
       model = ASTModel(filename, id_to_explain) 
       complete_filename = filename

    if model_name == 'cough':
       model = CoughModel(filename, id_to_explain) 
       complete_filename = filename
       filename = os.path.basename(filename)

    if model_name == 'drums':
       model = DrumsModel(filename, id_to_explain) 
       complete_filename = filename
       filename = os.path.basename(filename)
    
    if model_name == 'kws':
       model = KWSModel(filename, id_to_explain) 
       complete_filename = filename
       filename = os.path.basename(filename)
    
    output_path = Path(path) / filename / model_name / f"ft1_{id_to_explain}_{name}.json"
    
    if os.path.exists(output_path):
        return 
    
    input, outputs = model.process_input()
    real_score_id = outputs[id_to_explain]        
    predict_fn = model.get_predict_fn()   

    pred_zeros = predict_fn([np.zeros(len(input))])

    if model_name == 'drums':
        empty_score = compute_log_odds(pred_zeros, id_to_explain)[0]
    elif model_name == 'kws':
        empty_score = pred_zeros[id_to_explain]
    else:
        empty_score = pred_zeros[0][id_to_explain]
        
    kernelshap_analyzer = KernelShapExplainer(
        path= Path(path) / filename / model_name / f"scores_{name}.json",
    )
    importances_kernelshap = kernelshap_analyzer.explain_instance(
        label_to_explain=id_to_explain, empty_score=empty_score, model=model_name).get_feature_importances()

    kernelshap_analyzer_1constraint = KernelShapExplainer1constraint(
        path= Path(path) / filename / model_name / f"scores_{name}.json",
    )
    importances_kernelshap_analyzer_1constraint = kernelshap_analyzer_1constraint.explain_instance(
        label_to_explain=id_to_explain, empty_score=empty_score, model=model_name).get_feature_importances()

    rf_analyzer = RFExplainer(
        path= Path(path) / filename / model_name / f"scores_{name}.json",
        filename=filename,
    )
    importances_rf_tree = rf_analyzer.get_feature_importances(label_to_explain=id_to_explain, method='tree', model=model_name)

    # LR analysis
    lime_nocon_analyzer = LRnoconExplainer(
        Path(path) / filename / model_name / f"scores_{name}.json",
        verbose=False,
        absolute_feature_sort=False
    )
    importances_nocon = lime_nocon_analyzer.explain_instance(
        label_to_explain=id_to_explain, model=model_name).get_feature_importances()
    
    lime_analyzer = LRExplainer(
        Path(path) / filename / model_name / f"scores_{name}.json",
        verbose=False,
        absolute_feature_sort=False
    )
    importances_lime = lime_analyzer.explain_instance(
        label_to_explain=id_to_explain, model=model_name).get_feature_importances()
    
    true_markers = get_segments(complete_filename, id_to_explain, model_name)
    ############## Prepare output ##############
    output_data = {
        "metadata": {
            "filename": filename,
            "id_explained": float(id_to_explain),
            "segment_length": 100,
            "true_markers": true_markers,
            "true_score": real_score_id
        },
        "importance_scores": {
            "importances_kernelshap_analyzer_1constraint": {
                "method": "NaiveAudioAnalyzer",
                "values": importances_kernelshap_analyzer_1constraint.tolist() if hasattr(importances_kernelshap_analyzer_1constraint, 'tolist') else importances_kernelshap_analyzer_1constraint
            },
            "random_forest_tree_importance": {
                    "method": "masked rf with tree importance",
                    "values": importances_rf_tree.tolist() if hasattr(importances_rf_tree, 'tolist') else importances_rf_tree
            },
            "linear_regression": {
                "method": "Linear Regression with kernel as pi",
                "values": importances_lime.tolist() if hasattr(importances_lime, 'tolist') else importances_lime
            },
            "linear_regression_nocon": {
                "method": "Linear Regression with kernel as pi",
                "values": importances_nocon.tolist() if hasattr(importances_nocon, 'tolist') else importances_nocon
            },
            "kernel_shap": {
                "method": "Linear Regression with shap as pi",
                "values": importances_kernelshap.tolist() if hasattr(importances_kernelshap, 'tolist') else importances_kernelshap  
            }
        }
    }

    # Save results
   
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Importance scores saved to: {output_path}")
   
    return 

