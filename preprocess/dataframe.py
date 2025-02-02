import pandas as pd
import json
from transformers import ASTForAudioClassification

def load_and_preprocess_segments(filepath):
    """
    Load and preprocess AudioSet segments from TSV file.
    """
    df_segmented = pd.read_csv(filepath, sep='\t')
    df_segmented['base_segment_id'] = df_segmented['segment_id'].str.rsplit('_', n=1).str[0]
    return df_segmented

def merge_continuous_segments(grouped_segments, threshold=0.1):
    """
    Merge segments that are temporally close to each other.
    """
    merged_segments = {}
    
    for key, segments in grouped_segments.items():
        if len(segments) <= 1:
            merged_segments[key] = segments
            continue
            
        segments = sorted(segments, key=lambda x: x[0])
        merged = []
        current_start, current_end = segments[0]
        
        for next_start, next_end in segments[1:]:
            if next_start - current_end <= threshold:
                current_end = next_end
            else:
                merged.append((current_start, current_end))
                current_start, current_end = next_start, next_end
        
        merged.append((current_start, current_end))
        merged_segments[key] = merged
    
    return merged_segments

def convert_merged_segments_to_df(merged_segments):
    rows = []
    for (base_segment_id, label), segments in merged_segments.items():
        row = {
            'base_segment_id': base_segment_id,
            'label': label
        }
  
        for i, (start, end) in enumerate(segments, 1):
            row[f'segment_{i}'] = [start, end]
        
        total_duration = sum(end - start for start, end in segments)
        row['duration'] = total_duration        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    segment_cols = sorted([col for col in df.columns if col.startswith('segment_')])
    other_cols = ['base_segment_id', 'label', 'duration']
    df = df[other_cols + segment_cols]
    
    return df


def get_label_name(label_id, ontology_data):
    for item in ontology_data:
        if item['id'] == label_id:
            return item['name']
    return None


def find_parent_chain(child_id, data):
    id_to_item = {item["id"]: item for item in data}

    parent_chain = []
    current_id = child_id
    
    while True:
        found_parent = False
        
        # Check all items to see if current_id is in their child_ids
        for item in id_to_item.values():
            if current_id in item["child_ids"]:
                parent_chain.append(get_label_name(item["id"], data))  # Found the parent, add it to the chain
                current_id = item["id"]  # Move up to the parent
                found_parent = True
                break        
        if not found_parent:
            break
    
    parent_chain.reverse()
    parent_chain_dict = {
        child_id: {
            "parent_0": parent_chain[0] if len(parent_chain) > 0 else get_label_name(child_id, data),
            "parent_1": parent_chain[1] if len(parent_chain) > 1 else get_label_name(child_id, data),
            "parent_2": parent_chain[2] if len(parent_chain) > 2 else get_label_name(child_id, data)
        }
    }
    return parent_chain_dict

def get_father_id_ast(father_label, model):
    # Check each parent in the chain
    for key, items in father_label.items():
        for parent_key in reversed(list(items.keys())):
            parent_id = items.get(parent_key, '')
            if parent_id != '':
                father_id = model.config.label2id.get(parent_id, -1)
                if father_id != -1:
                    return father_id  # Return the first valid parent ID
    return -1 

def get_father_id_yamnet(father_label):
    with open('/home/ec2-user/Datasets/Audioset/labels/labels_yamnet.json', 'r') as file:
        yamnet_label = json.load(file)
    model = yamnet_label['label']
    for key, items in father_label.items():
        for parent_key in reversed(list(items.keys())):
            parent_id = items.get(parent_key, '')
            if parent_id != '':
                for i, word in enumerate(model):
                    if word == parent_id:
                        return i
    return -1 

def process_audioset_data(tsv_filepath, ontology_filepath, threshold=0.1):
    # Load and preprocess segments
    df_segmented = load_and_preprocess_segments(tsv_filepath)
    
    # Group segments
    grouped = df_segmented.groupby(['base_segment_id', 'label']).apply(
        lambda x: list(zip(x['start_time_seconds'], x['end_time_seconds']))
    ).to_dict()
    
    # Merge continuous segments
    merged_segments = merge_continuous_segments(grouped, threshold)
    
    # Convert back to DataFrame
    df_segmented = convert_merged_segments_to_df(merged_segments)
    
    # Load ontology data
    with open(ontology_filepath, 'r') as file:
        ontology_data = json.load(file)
    
    # Load AST model
    model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
 
    df_segmented['parent_labels'] =df_segmented['label'].apply(
        lambda x: find_parent_chain(x, ontology_data)
    ) 

    df_segmented['father_id_ast'] = df_segmented['parent_labels'].apply(
        lambda x: get_father_id_ast(x, model=model)
    ).astype('Int64')

    df_segmented['father_id_yamnet'] = df_segmented['parent_labels'].apply(
        lambda x: get_father_id_yamnet(x)
    )

    df_segmented = df_segmented[df_segmented['father_id_ast'] != -1]
    df_segmented = df_segmented[df_segmented['duration'] < 5.0]
    df_segmented = df_segmented.dropna(axis=1, how='all')

    return df_segmented

# Example usage
if __name__ == "__main__":
    tsv_file =  '/home/ec2-user/Datasets/Audioset/labels/audioset_eval_strong.tsv'
    ontology_file = '/home/ec2-user/Datasets/Audioset/labels/ontology.json'
    
    processed_df = process_audioset_data(
        tsv_filepath=tsv_file,
        ontology_filepath=ontology_file,
        threshold=0.1  # Merge segments with gaps <= 0.1 seconds
    )
    processed_df.to_csv('/home/ec2-user/Datasets/Audioset/labels/audioset_eval.csv')



