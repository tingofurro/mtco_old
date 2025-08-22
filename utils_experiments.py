from datetime import datetime
import os, sys, shutil
import subprocess


def make_exp_folder(prefix="exp"):
    server_name = os.environ.get("SERVER_NAME", "")
    if not server_name:
        print("\033[93mSERVER_NAME environment variable is not set. Add something like 'export SERVER_NAME=A' to your .bashrc\033[0m") # needed to differentiate experiments across servers.
        server_name = ""

    exp_taken, run_idx = True, 1
    while exp_taken:
        exp_id = f"{prefix}{datetime.now().strftime('%m%d')}_{server_name}_{str(run_idx)}"
        exp_folder = os.path.join(os.path.dirname(__file__), "experiments", exp_id)
        if not os.path.exists(exp_folder):
            exp_taken = False
        run_idx += 1

    os.makedirs(exp_folder, exist_ok=True)
    return exp_folder


def sync_experiments(machines=["gpu3", "gpu4", "gpu7"], skip_models=True): # "gpu1", "gpu2", 
    local_experiments_dir = os.path.join(os.environ["HOME"], "mtco/experiments/")
    os.makedirs(local_experiments_dir, exist_ok=True)
    
    sync_summary = {}
    
    for machine in machines:
        print(f"Syncing experiments from {machine}...")
        sync_summary[machine] = {"synced": [], "skipped": [], "errors": []}
        
        try:
            # Get list of experiments on remote machine
            # remote_experiments_dir = f"{machine}:~/mtco/experiments/"
            
            # Check if remote experiments directory exists
            check_cmd = f"ssh {machine} 'test -d ~/mtco/experiments && ls ~/mtco/experiments || echo NO_EXPERIMENTS'"
            result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                sync_summary[machine]["errors"].append(f"Failed to connect to {machine}")
                print(f"  Error: Failed to connect to {machine}")
                continue
                
            if result.stdout.strip() == "NO_EXPERIMENTS":
                print(f"  No experiments directory found on {machine}")
                continue
                
            remote_exp_folders = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
            
            if not remote_exp_folders:
                print(f"  No experiments found on {machine}")
                continue
                
            for exp_folder in remote_exp_folders:
                if not exp_folder:  # Skip empty strings
                    continue
                    
                local_exp_path = os.path.join(local_experiments_dir, exp_folder)
                
                # Check if local folder exists (naming conflict)
                # not needed anymore due to SERVER_NAME
                # if os.path.exists(local_exp_path):
                #     # Create modified name with machine suffix
                #     modified_name = f"{exp_folder}_{machine}"
                #     local_exp_path = os.path.join(local_experiments_dir, modified_name)
                #     print(f"  Conflict detected for {exp_folder}, using name: {modified_name}")
                
                # Use rsync to sync the folder (skips files with same size)
                remote_exp_path = f"{machine}:~/mtco/experiments/{exp_folder}/"
                
                # Create local directory if it doesn't exist
                os.makedirs(local_exp_path, exist_ok=True)

                rsync_cmd = ["rsync", "-avz", "--size-only", "--progress"]
                
                # Add exclude pattern for model files if skip_models is True
                if skip_models:
                    rsync_cmd.extend(["--exclude", "*.safetensors"])
                
                rsync_cmd.extend([remote_exp_path, local_exp_path])
                
                print(f"  Syncing {exp_folder}...")
                rsync_result = subprocess.run(rsync_cmd, capture_output=True, text=True, timeout=300)
                
                if rsync_result.returncode == 0:
                    sync_summary[machine]["synced"].append(os.path.basename(local_exp_path))
                    print(f"  ✓ Successfully synced {exp_folder}")
                else:
                    error_msg = f"Rsync failed for {exp_folder}: {rsync_result.stderr}"
                    sync_summary[machine]["errors"].append(error_msg)
                    print(f"  ✗ {error_msg}")
                    
        except subprocess.TimeoutExpired:
            error_msg = f"Timeout while connecting to {machine}"
            sync_summary[machine]["errors"].append(error_msg)
            print(f"  ✗ {error_msg}")
        except Exception as e:
            error_msg = f"Unexpected error with {machine}: {str(e)}"
            sync_summary[machine]["errors"].append(error_msg)
            print(f"  ✗ {error_msg}")
    
    # Print summary
    print("\n" + "="*50)
    print("SYNC SUMMARY")
    print("="*50)
    for machine, summary in sync_summary.items():
        print(f"\n{machine}:")
        if summary["synced"]:
            print(f"  Synced: {', '.join(summary['synced'])}")
        if summary["skipped"]:
            print(f"  Skipped: {', '.join(summary['skipped'])}")
        if summary["errors"]:
            print(f"  Errors: {len(summary['errors'])} error(s)")
            for error in summary["errors"]:
                print(f"    - {error}")
    
    return sync_summary


if __name__ == "__main__":
    # Example usage
    sync_experiments(skip_models=True) # run this to download models from other machines
