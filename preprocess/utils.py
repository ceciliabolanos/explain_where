import networkx as nx
from collections import defaultdict
from typing import Dict, List, Set, Tuple

def get_name_id_mappings(data: List[dict]) -> Dict[str, Tuple[int, str, str]]:
    """
    Create a mapping for each node including both IDs and names:
    - For levels 0-2: maps to their own ID and name
    - For levels 3+: maps to their level 2 ancestor's ID and name
    
    Returns: dict with node IDs as keys and tuples of (level, mapped_id, mapped_name) as values
    """
    # Create directed graph
    G = nx.DiGraph()
    
    # Create node mapping for quick access
    node_map = {item['id']: item for item in data}
    
    # Build graph
    for item in data:
        G.add_node(item['id'])
        if 'child_ids' in item:
            for child_id in item['child_ids']:
                G.add_edge(item['id'], child_id)
    
    # Find root nodes
    roots = [node for node, in_degree in G.in_degree() if in_degree == 0]
    
    # Initialize level tracking
    node_levels = {}
    seen_nodes = set()
    
    def assign_levels(node: str, level: int, seen: Set[str]) -> None:
        """Recursively assign levels to nodes."""
        if node in seen:
            return
        
        seen.add(node)
        current_level = node_levels.get(node, level)
        node_levels[node] = min(level, current_level) if node in node_levels else level
        
        for child in G.successors(node):
            assign_levels(child, level + 1, seen)
    
    # Assign levels starting from each root
    for root in roots:
        assign_levels(root, 0, set())
    
    # Find level 2 ancestors for each node
    def find_level_2_ancestor(node: str) -> str:
        """Find the level 2 ancestor for a given node."""
        if node_levels[node] <= 2:
            return node
            
        # Walk up the graph until we find a level 2 node
        current = node
        while current in G:
            predecessors = list(G.predecessors(current))
            if not predecessors:
                break
                
            current = predecessors[0]
            if node_levels[current] == 2:
                return current
                
        return node
    
    # Create the final mapping
    mappings = {}
    for node in G.nodes():
        level = node_levels[node]
        if level <= 2:
            mappings[node] = (level, node, node_map[node]['name'])
        else:
            level_2_ancestor = find_level_2_ancestor(node)
            mappings[node] = (level, level_2_ancestor, node_map[level_2_ancestor]['name'])
    
    return mappings