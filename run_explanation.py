from explainers.shap import SHAPExplainer
from explainers.lr import LRExplainer
from explainers.rf import RFExplainer

from data_generator.mask_windows import WindowMaskingDataGenerator
from data_generator.base_generator import MaskingConfig

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
                  std_dataset: float,
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
            std_dataset=std_dataset, 
            path=path
        )

    data_generator.mode = 'all_masked'
    if not os.path.exists(Path(path) / filename / model_name / f"scores_p{config.mask_percentage}_w{config.window_size}_f{config.function}_m{config.mask_type}.json"):
        data_generator.generate(filename)

    ######### Generate the importances for each method ##########
    output_path = Path(path) / filename / model_name / f"ft_{id_to_explain}_p{config.mask_percentage}_w{config.window_size}_f{config.function}_m{config.mask_type}.json"

    if os.path.exists(output_path):
        return 

    kernelshap_analyzer = SHAPExplainer(
        path= Path(path) / filename / model_name / f"scores_p{config.mask_percentage}_w{config.window_size}_f{config.function}_m{config.mask_type}.json",
    )
    importances_kernelshap, local_pred = kernelshap_analyzer.get_feature_importances(label_to_explain=id_to_explain)

    rf_analyzer = RFExplainer(
        path= Path(path) / filename / model_name / f"scores_p{config.mask_percentage}_w{config.window_size}_f{config.function}_m{config.mask_type}.json")
    
    importances_rf_tree, local_pred = rf_analyzer.get_feature_importances(label_to_explain=id_to_explain, method='tree')

    # LR analysis
    lr_analyzer = LRExplainer(
        Path(path) / filename / model_name / f"scores_p{config.mask_percentage}_w{config.window_size}_f{config.function}_m{config.mask_type}.json")
    
    importances_lr, local_pred  = lr_analyzer.get_feature_importances(label_to_explain=id_to_explain)
    
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
            "SHAP": {
                "method": "NaiveAudioAnalyzer",
                "values": importances_kernelshap.tolist() if hasattr(importances_kernelshap, 'tolist') else importances_kernelshap
            },
            "RF": {
                    "method": "masked rf with tree importance",
                    "values": importances_rf_tree.tolist() if hasattr(importances_rf_tree, 'tolist') else importances_rf_tree
            },
            "LR": {
                "method": "Linear Regression with kernel as pi",
                "values": importances_lr.tolist() if hasattr(importances_lr, 'tolist') else importances_lr
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
    
    output_path = Path(path) / filename / model_name / f"ft_{id_to_explain}_{name}.json"
    
    if os.path.exists(output_path):
        return 
    
    input, outputs = model.process_input()
    real_score_id = outputs[id_to_explain]        
    predict_fn = model.get_predict_fn()   

    pred_zeros = predict_fn([np.zeros(len(input))])
    empty_score = pred_zeros[0][id_to_explain]
        

    kernelshap = SHAPExplainer(
        path= Path(path) / filename / model_name / f"scores_{name}.json",
    )
    importances_kernelshap, local_pred = kernelshap.get_feature_importances(label_to_explain=id_to_explain)

    rf_analyzer = RFExplainer(
        path= Path(path) / filename / model_name / f"scores_{name}.json")

    importances_rf_tree, local_pred = rf_analyzer.get_feature_importances(label_to_explain=id_to_explain, method='tree')

    lime_analyzer = LRExplainer(
        Path(path) / filename / model_name / f"scores_{name}.json")
    
    importances_lr, local_pred = lime_analyzer.get_feature_importances(label_to_explain=id_to_explain)
    
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
            "SHAP": {
                "method": "Shap",
                "values": importances_kernelshap.tolist() if hasattr(importances_kernelshap, 'tolist') else importances_kernelshap
            },
            "RF": {
                    "method": "masked rf with tree importance",
                    "values": importances_rf_tree.tolist() if hasattr(importances_rf_tree, 'tolist') else importances_rf_tree
            },
            "LR": {
                "method": "Linear Regression with kernel as pi",
                "values": importances_lr.tolist() if hasattr(importances_lr, 'tolist') else importances_lr
            }
           
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Importance scores saved to: {output_path}")
   
    return 

