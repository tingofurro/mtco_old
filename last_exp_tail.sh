#!/bin/bash

le() {
    local experiments_dir="$HOME/mtco_old/experiments"
    
    # Check if experiments directory exists
    if [[ ! -d "$experiments_dir" ]]; then
        echo "Error: Experiments directory not found: $experiments_dir"
        return 1
    fi
    
    # Find the most recently updated folder
    local latest_folder=$(find "$experiments_dir" -maxdepth 1 -type d ! -path "$experiments_dir" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
    
    if [[ -z "$latest_folder" ]]; then
        echo "Error: No experiment folders found in $experiments_dir"
        return 1
    fi
    
    local log_file="$latest_folder/run_logs.ans"
    
    # Check if the log file exists
    if [[ ! -f "$log_file" ]]; then
        echo "Error: Log file not found: $log_file"
        return 1
    fi
    
    echo "Tailing log file from latest experiment: $(basename "$latest_folder")"
    echo "File: $log_file"
    echo "---"
    
    # Tail the log file
    tail -f "$log_file"
}

# Call the function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    le "$@"
fi
