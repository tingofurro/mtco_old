from flask import Flask, render_template, jsonify, request
from datetime import datetime
from tasks import get_task
import os, json

app = Flask(__name__)

def load_conversations_from_experiment(exp_folder):
    """Load conversations from tree_logs.jsonl in an experiment folder"""
    log_file = os.path.join(exp_folder, "tree_logs.jsonl")
    conversations = []
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            for line in f:
                if line.strip():
                    # Replace NaN values with null in the JSON line
                    line = line.replace('NaN', 'null')
                    conversations.append(json.loads(line))
    return conversations

def format_timestamp(timestamp_str):
    if not timestamp_str:
        return "No timestamp"
    try:
        dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%H:%M:%S")
    except ValueError:
        return "Invalid timestamp"

def find_node_by_id(trace, node_id):
    for node in trace:
        if node["id"] == node_id:
            return node
    return None

def get_siblings(trace, node):
    if not node["parent"]:
        return []
    parent = find_node_by_id(trace, node["parent"])
    return [child_id for child_id in parent["children"] if child_id != node["id"]]

def build_tree_structure(conversation):
    """Build a complete tree structure for client-side rendering"""
    trace = conversation['trace']

    # Build node lookup and enrich nodes with sibling information
    nodes = {}
    for node in trace:
        node_copy = node.copy()
        node_copy['timestamp_formatted'] = format_timestamp(node.get('timestamp', ''))

        if node['role'] == 'assistant':
            siblings = sorted(set(get_siblings(trace, node) + [node['id']]))
            node_copy['siblings'] = siblings
            node_copy['variant_index'] = siblings.index(node['id'])
            node_copy['total_variants'] = len(siblings)
            
            # Add log probability information if available
            if 'logprobs' in node:
                node_copy['logprobs'] = node['logprobs']
            
            # Add strategy information if available
            if 'strategy_name' in node:
                node_copy['strategy_name'] = node['strategy_name']
            
            # Add advantage information if available
            if 'advantage' in node:
                node_copy['advantage'] = node['advantage']
            
            if 'response_tokens' in node:
                num_tokens = len(node['response_tokens'])
                node_copy['num_tokens'] = num_tokens
                
                # Calculate normalized logprob (logprobs / num_tokens)

                if 'logprobs' not in node or node['logprobs'] is None:
                    node_copy["logprobs"] = 0.0

                if 'logprobs' in node and node['logprobs'] is not None and num_tokens > 0:
                    node_copy['normalized_logprob'] = node['logprobs'] / num_tokens
                else:
                    node_copy['normalized_logprob'] = 0.0

        nodes[node['id']] = node_copy

    # Build the conversation flow starting from root
    conversation_flow = []
    current_node = trace[0]  # System message

    while current_node:
        conversation_flow.append(current_node['id'])

        if not current_node['children']:
            break

        # For the default path, always take the first child
        next_node_id = current_node['children'][0]
        current_node = find_node_by_id(trace, next_node_id)

    return {
        'nodes': nodes,
        'conversation_flow': conversation_flow,
        'conv_id': conversation['conv_id']
    }

def load_alias(exp_folder):
    """Load experiment alias from alias.txt file"""
    alias_file = os.path.join(exp_folder, "alias.txt")
    if os.path.exists(alias_file):
        with open(alias_file, 'r') as f:
            alias = f.read().strip()
            return alias if alias else None
    return None

def get_display_name(exp_name, exp_folder):
    """Get display name for experiment (alias if available, otherwise folder name)"""
    alias = load_alias(exp_folder)
    return alias if alias else exp_name

@app.route('/')
def index():
    EXPERIMENTS_PATH = os.path.expanduser("~/mtco/experiments")

    if not os.path.exists(EXPERIMENTS_PATH):
        return render_template('error.html', error=f"Experiments folder not found: {EXPERIMENTS_PATH}")

    experiments = [d for d in os.listdir(EXPERIMENTS_PATH) if os.path.isdir(os.path.join(EXPERIMENTS_PATH, d))]
    
    # Sort by modification time (newest first)
    experiments_with_time = []
    for exp_name in experiments:
        exp_folder = os.path.join(EXPERIMENTS_PATH, exp_name)
        mod_time = os.path.getmtime(exp_folder)
        experiments_with_time.append((exp_name, mod_time))
    
    # Sort by modification time in reverse order (newest first)
    experiments_with_time.sort(key=lambda x: x[1], reverse=True)
    experiments = [exp[0] for exp in experiments_with_time]

    if not experiments:
        return render_template('error.html', error="No experiments found")

    # Create experiment data with both folder names and display names
    experiment_data = []
    for exp_name in experiments:
        exp_folder = os.path.join(EXPERIMENTS_PATH, exp_name)
        display_name = get_display_name(exp_name, exp_folder)
        experiment_data.append({
            'folder_name': exp_name,
            'display_name': display_name
        })

    return render_template('index.html',
                         experiments=experiment_data,
                         experiments_path=EXPERIMENTS_PATH)

@app.route('/get_conversation_data')
def get_conversation_data():
    experiment = request.args.get('experiment')
    iteration = int(request.args.get('iteration', 1))

    EXPERIMENTS_PATH = os.path.expanduser("~/mtco/experiments")
    exp_folder = os.path.join(EXPERIMENTS_PATH, experiment)

    conversations = load_conversations_from_experiment(exp_folder)
    if not conversations:
        return jsonify({'error': f'No conversations found in experiment: {experiment}'})

    # Get available iterations
    iterations = sorted(set(conv.get("iteration", 1) for conv in conversations))

    # Filter by iteration
    iteration_conversations = [conv for conv in conversations if conv.get("iteration", 1) == iteration]

    if not iteration_conversations:
        return jsonify({'error': f'No conversations found for iteration {iteration}'})

    # Get task
    first_conv = iteration_conversations[0]
    task_name = first_conv.get("task", "unknown")

    selected_conversation = iteration_conversations[0]

    # Build tree structure
    tree_data = build_tree_structure(selected_conversation)

    # Extract additional_info stats
    additional_info = {"avg_leaf_node_scores": selected_conversation.get("avg_leaf_node_scores", 0.0), "avg_corr_A_LP": selected_conversation.get("avg_corr_A_LP", 0.0), "avg_corr_A_RL": selected_conversation.get("avg_corr_A_RL", 0.0), "avg_response_length": selected_conversation.get("avg_response_length", 0.0), "avg_depth_rls": selected_conversation.get("avg_depth_rls", {}), "timings": selected_conversation.get("timings", {}), "num_backprop_updates": selected_conversation.get("num_backprop_updates", 0)}

    return jsonify({
        'tree_data': tree_data,
        'iterations': iterations,
        'current_iteration': iteration,
        'stats': additional_info
    })

if __name__ == '__main__':
    app.run(debug=True, port=5002, host="0.0.0.0")
