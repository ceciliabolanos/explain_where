import json

def open_json(file_path):
    """
    Open and read a JSON file
    """
    with open(file_path, 'r') as f:
        return json.load(f)


