import json
import random
from typing import List, Dict, Any

def create_mock_tree(max_depth: int, assistant_degree: int, fraction_1: float = 0.5, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Create a mock conversation tree with arbitrary depth and assistant degree.
    
    Args:
        max_depth: Maximum depth of the tree (leaf nodes will be at this depth)
        assistant_degree: Number of assistant children each user node has
        fraction_1: Fraction of assistant leaf nodes that get score 1.0 (rest get 0.0)
        seed: Random seed for reproducibility
    
    Returns:
        List of tree nodes in the format expected by the analysis code
    """
    random.seed(seed)
    tree = []
    
    def create_node(node_id: str, depth: int, role: str, parent: str = None) -> Dict[str, Any]:
        """Helper function to create a node with the standard structure"""
        return {
            "id": node_id,
            "depth": depth,
            "role": role,
            "parent": parent,
            "children": [],
            "is_leaf": False,
            "backpropagated_score": None
        }
    
    def generate_nodes_recursive(parent_id: str, current_depth: int, parent_role: str):
        """Recursively generate nodes up to max_depth"""
        if current_depth > max_depth:
            return
            
        # Determine role for current level (alternating pattern)
        if parent_role == "assistant":
            current_role = "user"
            num_children = 1  # Each assistant has 1 user child
        else:  # parent_role == "user"
            current_role = "assistant"
            num_children = assistant_degree  # Each user has assistant_degree assistant children
        
        # Generate children for this level
        children_ids = []
        for i in range(num_children):
            if parent_id == "S":
                child_id = f"{parent_id}.U{i}"
            elif current_role == "assistant":
                child_id = f"{parent_id}.A{i}"
            else:  # current_role == "user"
                child_id = f"{parent_id}.U{i}"
            
            children_ids.append(child_id)
            
            # Create the child node
            child_node = create_node(child_id, current_depth, current_role, parent_id)
            tree.append(child_node)
            
            # Recursively generate grandchildren
            generate_nodes_recursive(child_id, current_depth + 1, current_role)
        
        # Update parent's children list
        parent_node = next(node for node in tree if node["id"] == parent_id)
        parent_node["children"] = children_ids
    
    # Create root node (always assistant at depth 0)
    root = create_node("S", 0, "assistant")
    tree.append(root)
    
    # Generate the rest of the tree
    generate_nodes_recursive("S", 1, "assistant")
    
    # Identify leaf nodes and assign scores
    leaf_nodes = [node for node in tree if node["depth"] == max_depth]
    assistant_leaves = [node for node in leaf_nodes if node["role"] == "assistant"]
    
    # Mark leaf nodes
    for node in tree:
        if node["depth"] == max_depth:
            node["is_leaf"] = True
            node["children"] = []
    
    num_leaves = len(assistant_leaves)
    num_ones = int(num_leaves * fraction_1)
    num_zeros = num_leaves - num_ones
    scores = [1.0] * num_ones + [0.0] * num_zeros
    random.shuffle(scores)
    
    for i, leaf in enumerate(assistant_leaves):
        leaf["score"] = scores[i]
    
    return tree

def save_mock_tree(tree: List[Dict[str, Any]], filename: str = "data/mock_tree_data.json"):
    """Save the mock tree to a JSON file"""
    with open(filename, "w") as f:
        json.dump(tree, f, indent=2)

def print_tree_summary(tree: List[Dict[str, Any]]):
    """Print a summary of the tree structure"""
    print(f"Created mock tree with {len(tree)} nodes")
    print("Tree structure:")
    for node in tree:
        role_indicator = "🤖" if node["role"] == "assistant" else "👤"
        leaf_indicator = "🍃" if node["is_leaf"] else "🌿"
        score = node.get("score", None)
        score_str = f"{score:.3f}" if score is not None else "None"
        print(f"  {role_indicator}{leaf_indicator} {node['id']}: depth={node['depth']}, score={score_str}")

# Example usage
if __name__ == "__main__":
    # Create tree with depth 4 and assistant degree 3 (matching your original request)
    mock_tree = create_mock_tree(max_depth=8, assistant_degree=8, fraction_1=0.2)
    save_mock_tree(mock_tree)
    print_tree_summary(mock_tree)
    