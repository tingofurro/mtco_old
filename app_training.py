import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json, os, glob, datetime, numpy as np
from datetime import datetime
from utils_experiments import sync_experiments
import sys

if len(sys.argv) > 1:
    base_dir = sys.argv[1]
else:
    base_dir = "experiments"

# streamlit run app_training.py --server.address 0.0.0.0 --server.port 8501

# Page configuration
st.set_page_config(
    page_title="MTCO Training Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_experiments(base_dir="experiments"):
    """Get all experiment folders that contain run_stats.jsonl"""
    experiments = []
    exp_pattern = f"{base_dir}/*/run_stats.jsonl"
    for path in glob.glob(exp_pattern):
        exp_folder = os.path.dirname(path)
        exp_name = os.path.basename(exp_folder)
        # Get modification time of the folder
        mod_time = os.path.getmtime(exp_folder)
        experiments.append((exp_name, exp_folder, mod_time))
    
    # Sort by modification time in reverse order (newest first)
    experiments.sort(key=lambda x: x[2], reverse=True)
    
    # Return tuples of (name, folder) without the modification time
    return [(exp[0], exp[1]) for exp in experiments]

def load_run_stats(exp_folder):
    """Load run statistics from jsonl file"""
    stats_file = os.path.join(exp_folder, "run_stats.jsonl")
    data = []
    
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

    if os.path.exists(os.path.join(exp_folder, "run_tasks.jsonl")):
        with open(os.path.join(exp_folder, "run_tasks.jsonl"), 'r') as f:
            tasks = []
            for line in f:
                if line.strip():
                    tasks.append(json.loads(line))

        for task, d in zip(tasks, data):
            if task["iteration"] == d["iteration"]:
                d["task"] = task["task"]

    if data:
        df = pd.DataFrame(data)
        # Convert timestamp if it exists
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                # Calculate relative time in hours from first timestamp
                first_timestamp = df['timestamp'].min()
                df['relative_time_hours'] = (df['timestamp'] - first_timestamp).dt.total_seconds() / 3600
            except:
                df['relative_time_hours'] = range(len(df))
        else:
            df['relative_time_hours'] = range(len(df))
        
        return df
    return pd.DataFrame()

def load_run_params(exp_folder):
    """Load run parameters from json file"""
    params_file = os.path.join(exp_folder, "run_params.json")
    if os.path.exists(params_file):
        with open(params_file, 'r') as f:
            return json.load(f)
    return {}

def load_alias(exp_folder):
    """Load experiment alias from alias.txt file"""
    alias_file = os.path.join(exp_folder, "alias.txt")
    if os.path.exists(alias_file):
        with open(alias_file, 'r') as f:
            alias = f.read().strip()
            return alias if alias else None
    return None

def save_alias(exp_folder, alias):
    """Save experiment alias to alias.txt file"""
    alias_file = os.path.join(exp_folder, "alias.txt")
    with open(alias_file, 'w') as f:
        f.write(alias.strip())

def get_display_name(exp_name, exp_folder):
    """Get display name for experiment (alias if available, otherwise folder name)"""
    alias = load_alias(exp_folder)
    return alias if alias else exp_name

def smooth_data(data, window_size):
    """Apply moving average smoothing"""
    if len(data) < window_size:
        return data
    return data.rolling(window=window_size, center=True, min_periods=1).mean()

def create_line_plot(df, y_col, title, x_axis='iteration', smooth_window=1, show_best_markers=False, show_mean_line=False, show_task_markers=False):
    """Create a line plot with optional smoothing and best model markers"""
    if y_col not in df.columns:
        return go.Figure().add_annotation(text=f"Column '{y_col}' not found", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    x_col = x_axis
    fig = go.Figure()
    
    # Original data (if smoothing is applied)
    if smooth_window > 1:
        fig.add_trace(go.Scatter(
            x=df[x_col], y=df[y_col],
            mode='markers+lines',
            name='Raw',
            opacity=0.3,
            line=dict(width=1),
            marker=dict(size=3)
        ))
    
    # Main line (smoothed or original)
    y_data = smooth_data(df[y_col], smooth_window) if smooth_window > 1 else df[y_col]
    fig.add_trace(go.Scatter(
        x=df[x_col], y=y_data,
        mode='lines+markers',
        name='Smoothed' if smooth_window > 1 else 'Data',
        line=dict(width=2),
        marker=dict(size=4)
    ))
    
    # Add mean line if requested
    if show_mean_line and not df[y_col].empty:
        mean_value = df[y_col].mean()
        fig.add_hline(
            y=mean_value,
            line=dict(color="yellow", width=2, dash="solid"),
            annotation_text=f"Mean: {mean_value:.4f}",
            annotation_position="right",
            annotation=dict(
                bgcolor="yellow",
                bordercolor="orange",
                borderwidth=1,
                font=dict(color="black", size=12)
            )
        )
    
    # Add best model markers if requested
    if show_best_markers and 'is_best_model' in df.columns:
        best_points = df[df['is_best_model'] == True]
        if not best_points.empty:
            fig.add_trace(go.Scatter(
                x=best_points[x_col],
                y=best_points[y_col] if smooth_window == 1 else smooth_data(df[y_col], smooth_window)[best_points.index],
                mode='markers',
                name='Best Model',
                marker=dict(symbol='star', size=12, color='gold', line=dict(color='orange', width=1))
            ))

    # Add rollback markers
    if 'is_rollback' in df.columns:
        rollback_points = df[df['is_rollback'] == True]
        if not rollback_points.empty:
            fig.add_trace(go.Scatter(
                x=rollback_points[x_col],
                y=rollback_points[y_col] if smooth_window == 1 else smooth_data(df[y_col], smooth_window)[rollback_points.index],
                mode='markers',
                name='Rollback',
                marker=dict(symbol='x', size=14, color='purple', line=dict(color='darkviolet', width=2))
            ))

    if show_task_markers and 'task' in df.columns:
        
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode='markers',
            name='Task',
            marker=dict(
                symbol=[
                    'circle' if task == 'math' else
                    'square' if task == 'code' else
                    'diamond' if task == 'database' else
                    'cross' if task == 'actions' else
                    'triangle-up'
                    for task in df['task']
                ],
                color=[
                    'indianred' if task == 'math' else
                    'steelblue' if task == 'code' else
                    'seagreen' if task == 'database' else
                    'darkorange' if task == 'actions' else
                    'mediumpurple'
                    for task in df['task']
                ],
                size=6,
            ),
            showlegend=False
        ))
        for task in ["code", "math", "database", "actions"]:
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None], 
                mode='markers',
                name=f'Task: {task}',
                marker=dict(
                    symbol=(
                        'circle' if task == 'math' else
                        'square' if task == 'code' else
                        'diamond' if task == 'database' else
                        'cross' if task == 'actions' else
                        'triangle-up'
                    ),
                    color=(
                        'indianred' if task == 'math' else
                        'steelblue' if task == 'code' else
                        'seagreen' if task == 'database' else
                        'darkorange' if task == 'actions' else
                        'mediumpurple'
                    ),
                    size=6
                ),
                showlegend=True
            ))
    
    # Add vertical grid lines at epoch boundaries
    if 'epoch' in df.columns and len(df) > 1:
        # Find points where epoch changes
        epoch_changes = df[df['epoch'] != df['epoch'].shift(1)]
        for idx, row in epoch_changes.iterrows():
            if idx > 0:  # Skip the first point (it's always a "change" from NaN)
                fig.add_vline(
                    x=row[x_col],
                    line=dict(color="rgba(128, 128, 128, 0.5)", width=1, dash="dash"),
                    annotation_text=f"Epoch {int(row['epoch'])}",
                    annotation_position="top",
                    annotation=dict(textangle=90, font=dict(size=10))
                )
    
    fig.update_layout(
        title=title,
        xaxis_title="Hours from Start" if x_axis == 'relative_time_hours' else "Iteration",
        yaxis_title=y_col.replace('_', ' ').title(),
        showlegend=True,
        height=400
    )
    
    return fig

def create_depth_response_plot(df, x_axis='iteration', smooth_window=1):
    """Create overlapped plot for all avg_response_length_depth_X columns"""
    depth_cols = [col for col in df.columns if col.startswith('avg_response_length_depth_')]
    
    if not depth_cols:
        return go.Figure().add_annotation(text="No depth response length data found", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    x_col = x_axis
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    for i, col in enumerate(sorted(depth_cols)):
        # Extract depth number from column name
        depth = col.split('_')[-1]
        y_data = smooth_data(df[col], smooth_window) if smooth_window > 1 else df[col]
        
        fig.add_trace(go.Scatter(
            x=df[x_col], y=y_data,
            mode='lines+markers',
            name=f'Depth {depth}',
            line=dict(width=2, color=colors[i % len(colors)]),
            marker=dict(size=4)
        ))
    
    # Add vertical grid lines at epoch boundaries
    if 'epoch' in df.columns and len(df) > 1:
        # Find points where epoch changes
        epoch_changes = df[df['epoch'] != df['epoch'].shift(1)]
        for idx, row in epoch_changes.iterrows():
            if idx > 0:  # Skip the first point (it's always a "change" from NaN)
                fig.add_vline(
                    x=row[x_col],
                    line=dict(color="rgba(128, 128, 128, 0.5)", width=1, dash="dash"),
                    annotation_text=f"Epoch {int(row['epoch'])}",
                    annotation_position="top",
                    annotation=dict(textangle=90, font=dict(size=10))
                )
    
    fig.update_layout(
        title="Average Response Length by Depth",
        xaxis_title="Hours from Start" if x_axis == 'relative_time_hours' else "Iteration",
        yaxis_title="Average Response Length",
        showlegend=True,
        height=400
    )
    
    return fig

def create_variance_plot(df, data_col, title, x_axis='iteration', smooth_window=1):
    """Create a variance plot showing min/max, percentiles, and median"""
    if data_col not in df.columns:
        return go.Figure().add_annotation(text=f"Column '{data_col}' not found", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    x_col = x_axis
    fig = go.Figure()
    
    # Process data: extract percentiles from arrays
    plot_data = {
        'x': [],
        'min': [],
        'q25': [],
        'median': [],
        'q75': [],
        'max': []
    }
    
    for idx, row in df.iterrows():
        data_array = row[data_col]
        
        # Skip if None
        if data_array is None:
            continue
        
        # Skip if it's a scalar and NaN (avoid the array issue)
        if not hasattr(data_array, '__len__') and pd.isna(data_array):
            continue
            
        try:
            # Convert to list if needed
            if isinstance(data_array, list):
                data_list = data_array
            elif isinstance(data_array, (tuple, np.ndarray)):
                data_list = list(data_array)
            else:
                # Try to convert to list
                data_list = list(data_array)
            
            # Skip if empty
            if len(data_list) == 0:
                continue
            
            # Filter out NaN values but keep numeric values
            data_list = [x for x in data_list if not pd.isna(x) and isinstance(x, (int, float))]
            if len(data_list) == 0:
                continue
                
            plot_data['x'].append(row[x_col])
            plot_data['min'].append(np.min(data_list))
            plot_data['q25'].append(np.percentile(data_list, 25))
            plot_data['median'].append(np.median(data_list))
            plot_data['q75'].append(np.percentile(data_list, 75))
            plot_data['max'].append(np.max(data_list))
            
        except Exception as e:
            if idx < 3:
                print(f"DEBUG: Error processing row {idx}: {e}")
            continue
    
    if not plot_data['x']:
        return go.Figure().add_annotation(text=f"No valid data for '{data_col}'", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    # Apply smoothing if requested
    if smooth_window > 1:
        for key in ['min', 'q25', 'median', 'q75', 'max']:
            plot_data[key] = smooth_data(pd.Series(plot_data[key]), smooth_window).tolist()
    
    # Add traces
    # Add background fill areas first (so they appear behind the lines)
    # Fill between min and max (dark blue background)
    fig.add_trace(go.Scatter(
        x=plot_data['x'], y=plot_data['min'],
        mode='lines',
        name='Min-Max Range',
        line=dict(width=0),
        showlegend=False,
        fill=None
    ))
    
    fig.add_trace(go.Scatter(
        x=plot_data['x'], y=plot_data['max'],
        mode='lines',
        name='Min-Max Range',
        line=dict(width=0),
        fillcolor='rgba(173, 216, 230, 0.3)',  # Light blue with 0.3 opacity
        fill='tonexty',
        showlegend=True
    ))
    
    # Fill between 25th and 75th percentiles (blue background)
    fig.add_trace(go.Scatter(
        x=plot_data['x'], y=plot_data['q25'],
        mode='lines',
        name='25th-75th Percentile Range',
        line=dict(width=0),
        showlegend=False,
        fill=None
    ))
    
    fig.add_trace(go.Scatter(
        x=plot_data['x'], y=plot_data['q75'],
        mode='lines',
        name='25th-75th Percentile Range',
        line=dict(width=0),
        fillcolor='rgba(30, 144, 255, 0.3)',  # Darker blue with 0.3 opacity
        fill='tonexty',
        showlegend=True
    ))
    
    # Min/Max (dotted lines) - now in blue
    fig.add_trace(go.Scatter(
        x=plot_data['x'], y=plot_data['min'],
        mode='lines',
        name='Min',
        line=dict(dash='dot', width=1, color='lightblue'),
        opacity=0.7
    ))
    
    fig.add_trace(go.Scatter(
        x=plot_data['x'], y=plot_data['max'],
        mode='lines',
        name='Max',
        line=dict(dash='dot', width=1, color='lightblue'),
        opacity=0.7
    ))
    
    # 25th/75th percentiles (dashed lines) - now in blue
    fig.add_trace(go.Scatter(
        x=plot_data['x'], y=plot_data['q25'],
        mode='lines',
        name='25th %ile',
        line=dict(dash='dash', width=2, color='dodgerblue'),
        opacity=0.8
    ))
    
    fig.add_trace(go.Scatter(
        x=plot_data['x'], y=plot_data['q75'],
        mode='lines',
        name='75th %ile',
        line=dict(dash='dash', width=2, color='dodgerblue'),
        opacity=0.8
    ))
    
    # Median (solid line) - now in blue
    fig.add_trace(go.Scatter(
        x=plot_data['x'], y=plot_data['median'],
        mode='lines+markers',
        name='Median',
        line=dict(width=3, color='darkblue'),
        marker=dict(size=4)
    ))
    
    # Add vertical grid lines at epoch boundaries
    if 'epoch' in df.columns and len(df) > 1:
        # Find points where epoch changes
        epoch_changes = df[df['epoch'] != df['epoch'].shift(1)]
        for idx, row in epoch_changes.iterrows():
            if idx > 0:  # Skip the first point (it's always a "change" from NaN)
                fig.add_vline(
                    x=row[x_col],
                    line=dict(color="rgba(128, 128, 128, 0.5)", width=1, dash="dash"),
                    annotation_text=f"Epoch {int(row['epoch'])}",
                    annotation_position="top",
                    annotation=dict(textangle=90, font=dict(size=10))
                )
    
    fig.update_layout(
        title=title,
        xaxis_title="Hours from Start" if x_axis == 'relative_time_hours' else "Iteration",
        yaxis_title=data_col.replace('_', ' ').title(),
        showlegend=True,
        height=400
    )
    
    return fig

def calculate_mean_from_arrays(df, col_name):
    """Calculate mean values from array columns in dataframe"""
    means = []
    for idx, row in df.iterrows():
        data_array = row[col_name]
        if data_array is None or (hasattr(data_array, '__len__') and len(data_array) == 0):
            means.append(np.nan)
        else:
            try:
                if isinstance(data_array, list):
                    data_list = data_array
                elif isinstance(data_array, (tuple, np.ndarray)):
                    data_list = list(data_array)
                else:
                    data_list = list(data_array)
                
                # Filter out NaN values
                data_list = [x for x in data_list if not pd.isna(x) and isinstance(x, (int, float))]
                if len(data_list) > 0:
                    means.append(np.mean(data_list))
                else:
                    means.append(np.nan)
            except:
                means.append(np.nan)
    return means

def create_dual_line_plot(df, col1, col2, title, x_axis='iteration', smooth_window=1, color1='blue', color2='red'):
    """Create a plot with two lines from array data (using means)"""
    fig = go.Figure()
    
    x_col = x_axis
    
    # Calculate means for both columns
    y1_data = calculate_mean_from_arrays(df, col1)
    y2_data = calculate_mean_from_arrays(df, col2)
    
    # Create a temporary dataframe for smoothing
    temp_df = pd.DataFrame({
        x_col: df[x_col],
        'y1': y1_data,
        'y2': y2_data
    })
    
    # Apply smoothing if requested
    if smooth_window > 1:
        temp_df['y1'] = smooth_data(temp_df['y1'], smooth_window)
        temp_df['y2'] = smooth_data(temp_df['y2'], smooth_window)
    
    # Add first line
    fig.add_trace(go.Scatter(
        x=temp_df[x_col], y=temp_df['y1'],
        mode='lines+markers',
        name=col1.replace('_', ' ').title(),
        line=dict(width=2, color=color1),
        marker=dict(size=4)
    ))
    
    # Add second line
    fig.add_trace(go.Scatter(
        x=temp_df[x_col], y=temp_df['y2'],
        mode='lines+markers',
        name=col2.replace('_', ' ').title(),
        line=dict(width=2, color=color2),
        marker=dict(size=4)
    ))
    
    # Add vertical grid lines at epoch boundaries
    if 'epoch' in df.columns and len(df) > 1:
        epoch_changes = df[df['epoch'] != df['epoch'].shift(1)]
        for idx, row in epoch_changes.iterrows():
            if idx > 0:
                fig.add_vline(
                    x=row[x_col],
                    line=dict(color="rgba(128, 128, 128, 0.5)", width=1, dash="dash"),
                    annotation_text=f"Epoch {int(row['epoch'])}",
                    annotation_position="top",
                    annotation=dict(textangle=90, font=dict(size=10))
                )
    
    fig.update_layout(
        title=title,
        xaxis_title="Hours from Start" if x_axis == 'relative_time_hours' else "Iteration",
        yaxis_title="LogProb Values",
        showlegend=True,
        height=400
    )
    
    return fig

def create_combined_training_validation_plot(df, x_axis='iteration', smooth_window=1, show_best_markers=False, show_task_markers=False):
    """Create a combined plot for training and validation scores with both raw and smoothed lines"""
    training_col = 'avg_leaf_node_scores'
    validation_col = 'validation_score'
    
    if training_col not in df.columns and validation_col not in df.columns:
        return go.Figure().add_annotation(text="Training and validation data not found", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    x_col = x_axis
    fig = go.Figure()
    
    # Training scores (blue)
    if training_col in df.columns:
        # Raw training data (if smoothing is applied)
        if smooth_window > 1:
            fig.add_trace(go.Scatter(
                x=df[x_col], y=df[training_col],
                mode='markers+lines',
                name='Avg. Training Leaf Score (Raw)',
                opacity=0.3,
                line=dict(width=1, color='lightblue'),
                marker=dict(size=3, color='lightblue')
            ))
        
        # Main training line (smoothed or original)
        y_training = smooth_data(df[training_col], smooth_window) if smooth_window > 1 else df[training_col]
        fig.add_trace(go.Scatter(
            x=df[x_col], y=y_training,
            mode='lines+markers',
            name='Avg. Training Leaf Score' + (' (Smoothed)' if smooth_window > 1 else ''),
            line=dict(width=2, color='blue'),
            marker=dict(size=4, color='blue')
        ))
    
    # Validation scores (red)
    if validation_col in df.columns:
        # Raw validation data (if smoothing is applied)
        if smooth_window > 1:
            fig.add_trace(go.Scatter(
                x=df[x_col], y=df[validation_col],
                mode='markers+lines',
                name='Validation Score (Raw)',
                opacity=0.3,
                line=dict(width=1, color='lightcoral'),
                marker=dict(size=3, color='lightcoral')
            ))
        
        # Main validation line (smoothed or original)
        y_validation = smooth_data(df[validation_col], smooth_window) if smooth_window > 1 else df[validation_col]
        fig.add_trace(go.Scatter(
            x=df[x_col], y=y_validation,
            mode='lines+markers',
            name='Validation Score' + (' (Smoothed)' if smooth_window > 1 else ''),
            line=dict(width=2, color='red'),
            marker=dict(size=4, color='red')
        ))
    
    # Add best model markers if requested
    if show_best_markers and 'is_best_model' in df.columns and validation_col in df.columns:
        best_points = df[df['is_best_model'] == True]
        if not best_points.empty:
            fig.add_trace(go.Scatter(
                x=best_points[x_col],
                y=best_points[validation_col] if smooth_window == 1 else smooth_data(df[validation_col], smooth_window)[best_points.index],
                mode='markers',
                name='Best Model',
                marker=dict(symbol='star', size=12, color='gold', line=dict(color='orange', width=1))
            ))

    # Add rollback markers
    if 'is_rollback' in df.columns and validation_col in df.columns:
        rollback_points = df[df['is_rollback'] == True]
        if not rollback_points.empty:
            fig.add_trace(go.Scatter(
                x=rollback_points[x_col],
                y=rollback_points[validation_col] if smooth_window == 1 else smooth_data(df[validation_col], smooth_window)[rollback_points.index],
                mode='markers',
                name='Rollback',
                marker=dict(symbol='x', size=14, color='purple', line=dict(color='darkviolet', width=2))
            ))

    # Add task markers if requested
    if show_task_markers and 'task' in df.columns:
        # Use validation scores for task markers (since that's what we care about for evaluation)
        if validation_col in df.columns:
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[validation_col],
                mode='markers',
                name='Task',
                marker=dict(
                    symbol=[
                        'circle' if task == 'math' else
                        'square' if task == 'code' else
                        'diamond' if task == 'database' else
                        'cross' if task == 'actions' else
                        'triangle-up'
                        for task in df['task']
                    ],
                    color=[
                        'indianred' if task == 'math' else
                        'steelblue' if task == 'code' else
                        'seagreen' if task == 'database' else
                        'darkorange' if task == 'actions' else
                        'mediumpurple'
                        for task in df['task']
                    ],
                    size=6,
                ),
                showlegend=False
            ))
            for task in ["code", "math", "database", "actions"]:
                fig.add_trace(go.Scatter(
                    x=[None],
                    y=[None], 
                    mode='markers',
                    name=f'Task: {task}',
                    marker=dict(
                        symbol=(
                            'circle' if task == 'math' else
                            'square' if task == 'code' else
                            'diamond' if task == 'database' else
                            'cross' if task == 'actions' else
                            'triangle-up'
                        ),
                        color=(
                            'indianred' if task == 'math' else
                            'steelblue' if task == 'code' else
                            'seagreen' if task == 'database' else
                            'darkorange' if task == 'actions' else
                            'mediumpurple'
                        ),
                        size=6
                    ),
                    showlegend=True
                ))
    
    # Add vertical grid lines at epoch boundaries
    if 'epoch' in df.columns and len(df) > 1:
        # Find points where epoch changes
        epoch_changes = df[df['epoch'] != df['epoch'].shift(1)]
        for idx, row in epoch_changes.iterrows():
            if idx > 0:  # Skip the first point (it's always a "change" from NaN)
                fig.add_vline(
                    x=row[x_col],
                    line=dict(color="rgba(128, 128, 128, 0.5)", width=1, dash="dash"),
                    annotation_text=f"Epoch {int(row['epoch'])}",
                    annotation_position="top",
                    annotation=dict(textangle=90, font=dict(size=10))
                )
    
    fig.update_layout(
        title="Training and Validation Scores",
        xaxis_title="Hours from Start" if x_axis == 'relative_time_hours' else "Iteration",
        yaxis_title="Score",
        showlegend=True,
        height=400
    )
    
    return fig

def create_per_task_validation_plot(df, x_axis='iteration', smooth_window=1):
    """Create a plot showing validation scores for each task separately"""
    # Find all validation score columns for individual tasks
    val_task_cols = [col for col in df.columns if col.startswith('validation_score_') and col != 'validation_score']
    
    if not val_task_cols:
        return go.Figure().add_annotation(text="No per-task validation data found", 
                                        xref="paper", yref="paper", x=0.5, y=0.5)
    
    x_col = x_axis
    fig = go.Figure()
    
    # Task colors matching the task markers
    task_colors = {
        'math': 'indianred',
        'code': 'steelblue', 
        'database': 'seagreen',
        'actions': 'darkorange'
    }
    
    # Default colors for any other tasks
    default_colors = ['mediumpurple', 'orange', 'pink', 'brown', 'gray']
    color_index = 0
    
    for col in sorted(val_task_cols):
        # Extract task name from column (e.g., 'validation_score_math' -> 'math')
        task_name = col.replace('validation_score_', '')
        
        # Get color for this task
        if task_name in task_colors:
            color = task_colors[task_name]
        else:
            color = default_colors[color_index % len(default_colors)]
            color_index += 1
        
        # Apply smoothing if requested
        y_data = smooth_data(df[col], smooth_window) if smooth_window > 1 else df[col]
        
        fig.add_trace(go.Scatter(
            x=df[x_col], y=y_data,
            mode='lines+markers',
            name=f'{task_name.title()}',
            line=dict(width=2, color=color),
            marker=dict(size=4, color=color)
        ))
    
    # Add vertical grid lines at epoch boundaries
    if 'epoch' in df.columns and len(df) > 1:
        epoch_changes = df[df['epoch'] != df['epoch'].shift(1)]
        for idx, row in epoch_changes.iterrows():
            if idx > 0:
                fig.add_vline(
                    x=row[x_col],
                    line=dict(color="rgba(128, 128, 128, 0.5)", width=1, dash="dash"),
                    annotation_text=f"Epoch {int(row['epoch'])}",
                    annotation_position="top",
                    annotation=dict(textangle=90, font=dict(size=10))
                )
    
    fig.update_layout(
        title="Validation Scores by Task",
        xaxis_title="Hours from Start" if x_axis == 'relative_time_hours' else "Iteration",
        yaxis_title="Validation Score",
        showlegend=True,
        height=400
    )
    
    return fig

# Sidebar
st.sidebar.title("🎛️ Controls")

# Sync experiments button
if st.sidebar.button("🔄 Sync Experiments", help="Sync experiments from remote machines"):
    with st.sidebar:
        with st.spinner("Syncing experiments from remote machines..."):
            sync_summary = sync_experiments()
        
        # Show sync results
        st.success("Sync completed!")
        
        # Display summary
        for machine, summary in sync_summary.items():
            st.write(f"**{machine}:**")
            if summary["synced"]:
                st.write(f"✅ Synced: {len(summary['synced'])} experiments")
            if summary["errors"]:
                st.write(f"❌ Errors: {len(summary['errors'])}")
                for error in summary["errors"]:
                    st.write(f"  • {error}")
        
        # Refresh the page to show new experiments
        st.rerun()

# Get experiments
experiments = get_experiments(base_dir)
if not experiments:
    st.error("No experiments found! Please make sure you have experiment folders with run_stats.jsonl files.")
    st.stop()

# Experiment selection
exp_names = [exp[0] for exp in experiments]
exp_display_names = [get_display_name(exp[0], exp[1]) for exp in experiments]
selected_exp_index = st.sidebar.selectbox("Select Experiment", range(len(experiments)), 
                                         format_func=lambda i: exp_display_names[i])
selected_exp_name = exp_names[selected_exp_index]
selected_exp_folder = experiments[selected_exp_index][1]

# Alias editing
st.sidebar.subheader("✏️ Experiment Alias")
current_alias = load_alias(selected_exp_folder)
alias_input = st.sidebar.text_input("Experiment Name", 
                                   value=current_alias or selected_exp_name,
                                   placeholder="Enter a custom name for this experiment")

# Save alias button
if st.sidebar.button("💾 Save Name"):
    if alias_input.strip() and alias_input.strip() != selected_exp_name:
        save_alias(selected_exp_folder, alias_input.strip())
        st.sidebar.success("✅ Name saved!")
        st.rerun()
    elif alias_input.strip() == selected_exp_name:
        # Remove alias file if user sets it back to the original name
        alias_file = os.path.join(selected_exp_folder, "alias.txt")
        if os.path.exists(alias_file):
            os.remove(alias_file)
            st.sidebar.success("✅ Reset to original name!")
            st.rerun()
    else:
        st.sidebar.error("❌ Please enter a valid name")

# Get display name for the current experiment
display_name = get_display_name(selected_exp_name, selected_exp_folder)

# X-axis selection
x_axis = st.sidebar.radio("X-axis", ["iteration", "relative_time_hours"], 
                         format_func=lambda x: "Iteration (Step)" if x == "iteration" else "Relative Time (Hours)")

# Smoothing parameter
smooth_window = st.sidebar.slider("Smoothing Window", 1, 50, 20, 
                                 help="Number of points to average for smoothing (1 = no smoothing)")

show_task_markers = st.sidebar.checkbox("Show Task Markers", value=False)

# Load data
df = load_run_stats(selected_exp_folder)
params = load_run_params(selected_exp_folder)

if df.empty:
    st.error(f"No data found for experiment {selected_exp_name}")
    st.stop()

# Display run parameters
st.sidebar.subheader("📋 Run Parameters")
if params:
    for key, value in params.items():
        st.sidebar.text(f"{key}: {value}")
else:
    st.sidebar.text("No parameters file found")

# Main panel
st.title("📊 MTCO Training Dashboard")

# Determine experiment status based on last update
def get_experiment_status(df):
    """Determine if experiment is running or stopped based on last update"""
    if df.empty or 'timestamp' not in df.columns:
        return "unknown", ""
    
    # Get the last timestamp
    last_timestamp = df['timestamp'].max()
    if pd.isna(last_timestamp):
        return "unknown", ""
    
    # Calculate time difference
    now = pd.Timestamp.now()
    time_diff = now - last_timestamp
    
    # If less than 30 minutes, consider it running
    if time_diff.total_seconds() < 30 * 60:  # 30 minutes
        return "running", ""
    else:
        # Calculate relative time for stopped experiments
        total_seconds = time_diff.total_seconds()
        if total_seconds < 3600:  # Less than 1 hour
            minutes = int(total_seconds / 60)
            return "stopped", f"({minutes}min ago)"
        elif total_seconds < 86400:  # Less than 1 day
            hours = int(total_seconds / 3600)
            return "stopped", f"({hours}hr{'s' if hours != 1 else ''} ago)"
        else:  # More than 1 day
            days = int(total_seconds / 86400)
            return "stopped", f"({days}day{'s' if days != 1 else ''} ago)"

# Get experiment status
status, time_ago = get_experiment_status(df)

# Create header with status
if status == "running":
    status_emoji = "🟢"
    status_text = "Running"
elif status == "stopped":
    status_emoji = "🔴"
    status_text = f"Stopped {time_ago}"
else:
    status_emoji = "⚪"
    status_text = "Unknown"

st.header(f"Experiment: {display_name} {status_emoji} {status_text}")

# Create plots
col1, col2 = st.columns(2)

with col1:
    # Plot 1: Combined training and validation scores
    fig1 = create_combined_training_validation_plot(df, x_axis, smooth_window, 
                                                   show_best_markers=True, show_task_markers=show_task_markers)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Plot 3: avg_response_length
    fig3 = create_line_plot(df, 'avg_response_length', 
                           'Average Response Length', 
                           x_axis, smooth_window, show_task_markers=show_task_markers)
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    # Plot 2: Per-task validation scores
    fig2 = create_per_task_validation_plot(df, x_axis, smooth_window)
    st.plotly_chart(fig2, use_container_width=True)
    
    # Plot 4: Response length by depth (moved from above)
    fig4 = create_depth_response_plot(df, x_axis, smooth_window)
    st.plotly_chart(fig4, use_container_width=True)

# Plot 5: Collision Rate (full width)
if 'avg_collision_rate' in df.columns:
    st.subheader("Generation Analysis")
    fig5 = create_line_plot(df, 'avg_collision_rate', 
                           'Average Collision Rate (%)', 
                           x_axis, smooth_window, show_task_markers=show_task_markers)
    st.plotly_chart(fig5, use_container_width=True)

# Plot 8: Epoch count (full width, no smoothing - it's a discrete counter)
st.subheader("Epoch Progress")
fig8 = create_line_plot(df, 'epoch', 'Epoch Count', x_axis, 1)
st.plotly_chart(fig8, use_container_width=True)

# New section: Logprobs and Boost Analysis
st.subheader("📈 Logprobs and Boost Analysis")

# Check if we have the new data columns
has_boost_data = any(col in df.columns for col in ['boost_LP_percs', 'unboost_LP_percs'])
has_corr_data = any(col in df.columns for col in ['corrs_A_LP', 'corrs_A_RL', 'corrs_A_NLP'])
has_logprobs_data = any(col in df.columns for col in ['logprobs', 'normalized_logprobs'])

if has_boost_data or has_corr_data or has_logprobs_data:
    col1, col2 = st.columns(2)

    with col1:
        if 'boost_LP_percs' in df.columns:
            fig_boost = create_variance_plot(df, 'boost_LP_percs', 
                                           'Logprob Boost Percentages (Positive Advantages)', 
                                           x_axis, smooth_window)
            st.plotly_chart(fig_boost, use_container_width=True)
        
        if 'boost_LP1s' in df.columns and 'boost_LP2s' in df.columns:
            fig_boost_lp = create_dual_line_plot(df, 'boost_LP1s', 'boost_LP2s',
                                               'Boost LogProbs: Before vs After',
                                               x_axis, smooth_window, 'blue', 'red')
            st.plotly_chart(fig_boost_lp, use_container_width=True)


        if 'corrs_A_LP' in df.columns:
            fig_corr_lp = create_variance_plot(df, 'corrs_A_LP', 
                                             'Correlation: Advantages vs LogProbs', 
                                             x_axis, smooth_window)
            st.plotly_chart(fig_corr_lp, use_container_width=True)
    
    with col2:
        if 'unboost_LP_percs' in df.columns:
            fig_unboost = create_variance_plot(df, 'unboost_LP_percs', 
                                             'Logprob Unboost Percentages (Negative Advantages)', 
                                             x_axis, smooth_window)
            st.plotly_chart(fig_unboost, use_container_width=True)

        if 'unboost_LP1s' in df.columns and 'unboost_LP2s' in df.columns:
            fig_unboost_lp = create_dual_line_plot(df, 'unboost_LP1s', 'unboost_LP2s',
                                                 'Unboost LogProbs: Before vs After',
                                                 x_axis, smooth_window, 'blue', 'red')
            st.plotly_chart(fig_unboost_lp, use_container_width=True)


        if 'corrs_A_RL' in df.columns:
            fig_corr_rl = create_variance_plot(df, 'corrs_A_RL', 
                                             'Correlation: Advantages vs Response Length', 
                                             x_axis, smooth_window)
            st.plotly_chart(fig_corr_rl, use_container_width=True)

    # Additional section for logprobs distributions
    if has_logprobs_data:
        st.subheader("📊 Logprobs Distributions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'temperature' in df.columns:
                fig_temperature = create_line_plot(df, 'temperature', 
                                                 'Temperature', 
                                                 x_axis, 1)
                st.plotly_chart(fig_temperature, use_container_width=True)
        
        with col2:
            if 'logprobs' in df.columns:
                fig_logprobs = create_variance_plot(df, 'logprobs', 
                                                  'LogProbs Distribution', 
                                                  x_axis, smooth_window)
                st.plotly_chart(fig_logprobs, use_container_width=True)
        
        with col3:
            if 'normalized_logprobs' in df.columns:
                fig_norm_logprobs = create_variance_plot(df, 'normalized_logprobs', 
                                                       'Normalized LogProbs Distribution', 
                                                       x_axis, smooth_window)
                st.plotly_chart(fig_norm_logprobs, use_container_width=True)
            
else:
    st.info("No logprobs/boost data found. This data is available in newer training runs.")

# Data table (optional)
with st.expander("📊 Raw Data"):
    st.dataframe(df)

# New section: Guardrails Analysis
st.subheader("🛡️ Guardrails Analysis")

def create_guardrails_plot(df, x_axis='iteration', smooth_window=1):
    """Create individual plots for each guardrail showing both count and percentage"""
    # Find all guardrail columns
    trigger_count_cols = [col for col in df.columns if col.startswith('guardrail_') and col.endswith('_triggered') and not col.startswith('perc_')]
    
    if not trigger_count_cols:
        return [], []
    
    x_col = x_axis
    
    # Extract guardrail names and create readable labels
    guardrail_info = []
    for col in trigger_count_cols:
        # Extract name from guardrail_{name}_triggered
        name = col.replace('guardrail_', '').replace('_triggered', '')
        perc_col = f'perc_guardrail_{name}_triggered'
        
        # Create readable label
        if name == 'repetition':
            label = 'Repetition (5-gram ≥5x)'
        elif name == 'max_length':
            label = 'Max Length (≥1000 tokens)'
        else:
            label = name.replace('_', ' ').title()
        
        guardrail_info.append({
            'name': name,
            'label': label,
            'count_col': col,
            'perc_col': perc_col if perc_col in df.columns else None
        })
    
    # Create individual plots for each guardrail
    figures = []
    for info in guardrail_info:
        # Create subplot with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add count trace as bar chart (primary y-axis, background)
        fig.add_trace(
            go.Bar(
                x=df[x_col], 
                y=df[info['count_col']],
                name='Count',
                marker=dict(color='lightblue', opacity=0.6),
                yaxis='y'
            ),
            secondary_y=False,
        )
        
        # Add percentage trace as line plot (secondary y-axis, foreground)
        if info['perc_col'] and info['perc_col'] in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df[x_col], 
                    y=df[info['perc_col']],
                    mode='lines+markers',
                    name='Percentage',
                    line=dict(width=3, color='red'),
                    marker=dict(size=6, color='red'),
                    yaxis='y2'
                ),
                secondary_y=True,
            )
        
        # Add epoch boundary lines
        if 'epoch' in df.columns and len(df) > 1:
            epoch_changes = df[df['epoch'] != df['epoch'].shift(1)]
            for idx, row in epoch_changes.iterrows():
                if idx > 0:
                    fig.add_vline(
                        x=row[x_col],
                        line=dict(color="rgba(128, 128, 128, 0.5)", width=1, dash="dash"),
                        annotation_text=f"Epoch {int(row['epoch'])}",
                        annotation_position="top",
                        annotation=dict(textangle=90, font=dict(size=10))
                    )
        
        # Set x-axis title
        fig.update_xaxes(title_text="Hours from Start" if x_axis == 'relative_time_hours' else "Iteration")
        
        # Set y-axes titles
        fig.update_yaxes(title_text="Number of Triggers", secondary_y=False)
        fig.update_yaxes(title_text="Percentage (%)", secondary_y=True)
        
        # Update layout
        fig.update_layout(
            title=f"{info['label']} - Triggers",
            showlegend=True,
            height=400
        )
        
        figures.append(fig)
    
    return figures, guardrail_info

# Check if we have guardrails data
has_guardrails_data = any(col.startswith('guardrail_') and col.endswith('_triggered') for col in df.columns)

if has_guardrails_data:
    # Create guardrails plots
    figures, guardrail_info = create_guardrails_plot(df, x_axis, smooth_window)
    
    if figures:
        # Display individual plots for each guardrail
        if len(figures) == 1:
            # Single guardrail - full width
            st.plotly_chart(figures[0], use_container_width=True)
        elif len(figures) == 2:
            # Two guardrails - side by side
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(figures[0], use_container_width=True)
            with col2:
                st.plotly_chart(figures[1], use_container_width=True)
        else:
            # More than two guardrails - display in rows of 2
            for i in range(0, len(figures), 2):
                cols = st.columns(2)
                with cols[0]:
                    st.plotly_chart(figures[i], use_container_width=True)
                if i + 1 < len(figures):
                    with cols[1]:
                        st.plotly_chart(figures[i + 1], use_container_width=True)
