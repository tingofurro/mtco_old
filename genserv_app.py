import logging, multiprocessing, torch, sys, os, numpy # numpy needed to avoid MKL conflicts
from utils import print_colored, calculate_gpu_concurrency
from generation_service_vllm import GenerationServiceVLLM
from generation_service_hf import GenerationServiceHF
from flask import Flask, request, jsonify
import threading
import time
from collections import deque
from typing import Dict, List, Any
from datetime import datetime

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1' # needed to avoid MKL conflicts
os.environ['MKL_THREADING_LAYER'] = 'INTEL'

app = Flask(__name__)

# Utilization tracking system
class UtilizationTracker:
    def __init__(self):
        self.completed_requests = deque()  # Store (timestamp, num_tokens, job_id)
        self.total_requests = 0
        self.total_tokens = 0
        self.lock = threading.Lock()
        self.report_thread = None
        self.shutdown_flag = threading.Event()
        
    def start_reporting(self):
        """Start the background reporting thread"""
        if self.report_thread is None or not self.report_thread.is_alive():
            self.shutdown_flag.clear()
            self.report_thread = threading.Thread(target=self._report_loop, daemon=True)
            self.report_thread.start()
            print_colored("Started utilization reporting (every 30 seconds)", "green")
    
    def stop_reporting(self):
        """Stop the background reporting thread"""
        if self.report_thread and self.report_thread.is_alive():
            self.shutdown_flag.set()
            self.report_thread.join(timeout=1.0)
            print_colored("Stopped utilization reporting", "yellow")
    
    def track_completed_request(self, job_id: str, responses_or_tree):
        """Track a completed request with its token count"""
        current_time = time.time()
        total_tokens_in_job = 0
        
        # Handle tree results vs regular responses
        if isinstance(responses_or_tree, list) and len(responses_or_tree) > 0:
            # Check if this is a tree (list of nodes) or regular responses
            first_item = responses_or_tree[0]
            if isinstance(first_item, dict) and 'subtree_id' in first_item:
                # This is a tree - count tokens from all nodes
                for node in responses_or_tree:
                    if 'response_tokens' in node:
                        total_tokens_in_job += len(node['response_tokens'])
                    elif 'num_tokens' in node:
                        total_tokens_in_job += node['num_tokens']
                    else:
                        # Fallback: estimate tokens from response text
                        response_text = node.get('response_text', '')
                        total_tokens_in_job += max(1, len(response_text) // 4)
            else:
                # This is regular responses
                for response in responses_or_tree:
                    if isinstance(response, dict):
                        # Handle both response formats
                        if 'num_tokens' in response:
                            total_tokens_in_job += response['num_tokens']
                        elif 'response_tokens' in response:
                            total_tokens_in_job += len(response['response_tokens'])
                        else:
                            # Fallback: estimate tokens as roughly 1/4 of characters
                            response_text = response.get('response', response.get('content', ''))
                            total_tokens_in_job += max(1, len(response_text) // 4)
                    else:
                        # Handle string responses - estimate tokens
                        total_tokens_in_job += max(1, len(str(response)) // 4)
        
        with self.lock:
            self.completed_requests.append((current_time, total_tokens_in_job, job_id))
            self.total_requests += 1
            self.total_tokens += total_tokens_in_job
            
            # Keep only last hour of data to prevent memory growth
            cutoff_time = current_time - 3600  # 1 hour ago
            while self.completed_requests and self.completed_requests[0][0] < cutoff_time:
                self.completed_requests.popleft()
    
    def get_throughput_stats(self, time_window: float = 30.0) -> Dict[str, Any]:
        """Get throughput statistics for the specified time window (in seconds)"""
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        with self.lock:
            # Count requests and tokens in the time window
            recent_requests = [
                (timestamp, tokens, job_id) 
                for timestamp, tokens, job_id in self.completed_requests 
                if timestamp >= cutoff_time
            ]
            
            recent_request_count = len(recent_requests)
            recent_token_count = sum(tokens for _, tokens, _ in recent_requests)
            
            # Calculate throughput
            throughput_tps = recent_token_count / time_window if time_window > 0 else 0
            throughput_rps = recent_request_count / time_window if time_window > 0 else 0
            
            return {
                'time_window_seconds': time_window,
                'recent_requests': recent_request_count,
                'recent_tokens': recent_token_count,
                'throughput_tokens_per_second': throughput_tps,
                'throughput_requests_per_second': throughput_rps,
                'total_requests': self.total_requests,
                'total_tokens': self.total_tokens
            }
    
    def _report_loop(self):
        """Background thread that prints utilization reports every 30 seconds"""
        while not self.shutdown_flag.wait(30.0):  # Wait 30 seconds, check for shutdown
            stats = self.get_throughput_stats(30.0)
            timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
            
            # Get worker status from generation service
            worker_info = self._get_worker_status()
            
            # Print the main utilization report
            print_colored(
                f"{timestamp} Utilization Report: "
                f"{stats['throughput_tokens_per_second']:.1f} tok/s (last 30s), "
                f"{stats['recent_requests']} requests",
                "blue"
            )
            
            # Print per-worker status
            if worker_info:
                for worker_line in worker_info:
                    print_colored(f"  {worker_line}", "blue")
    
    def _get_worker_status(self):
        """Get detailed worker status information in condensed format"""
        # We need to access the global generation_service
        # This is a bit of coupling, but necessary for detailed worker info
        global generation_service
        
        if generation_service is None:
            return None
            
        try:
            worker_status = generation_service.get_worker_status()
            service_status = generation_service.get_status()
            queue_size = service_status.get('queue_size', 0)
            
            gpu_status_parts = []
            
            if BACKEND == "hf":
                # For HF backend with multiple workers per GPU
                # Group workers by GPU and aggregate their status
                gpu_stats = {}
                for worker_id in range(generation_service.total_workers):
                    gpu_id = worker_id // generation_service.workers_per_gpu
                    if gpu_id not in gpu_stats:
                        gpu_stats[gpu_id] = {'active_jobs': 0, 'total_workers': 0}
                    
                    # Count jobs for this worker
                    in_progress_jobs = 0
                    for job_info in generation_service.shared_jobs.values():
                        if job_info.get('status') == 'in_progress' and job_info.get('worker_id') == worker_id:
                            in_progress_jobs += 1
                    
                    gpu_stats[gpu_id]['active_jobs'] += in_progress_jobs
                    gpu_stats[gpu_id]['total_workers'] += 1
                
                # Create condensed GPU status
                for gpu_id in sorted(gpu_stats.keys()):
                    active = gpu_stats[gpu_id]['active_jobs']
                    total_workers = gpu_stats[gpu_id]['total_workers']
                    gpu_status_parts.append(f"GPU {gpu_id}: {active}/{total_workers}")
                    
            elif BACKEND == "vllm":
                # For vLLM backend with one worker per GPU
                for gpu_id in range(generation_service.num_gpus):
                    active_jobs = generation_service.shared_gpu_active_jobs[gpu_id] if gpu_id < len(generation_service.shared_gpu_active_jobs) else 0
                    max_concurrent = generation_service.max_concurrent_jobs_per_worker
                    gpu_status_parts.append(f"GPU {gpu_id}: {active_jobs}/{max_concurrent}")
            
            # Combine into single condensed line
            if gpu_status_parts:
                condensed_status = f"Active: {'; '.join(gpu_status_parts)}; Queued: {queue_size}"
                return [condensed_status]
            else:
                return [f"Queued: {queue_size}"]
            
        except Exception as e:
            return [f"Error getting worker status: {e}"]

# Global utilization tracker
utilization_tracker = UtilizationTracker()

# Configure logging to suppress check_on_job requests
@app.after_request
def log_request_info(response):
    """Custom logging that skips check_on_job requests"""
    # if request.endpoint not in ['check_on_job', 'health', "schedule_job"]:
    #     app.logger.info('%s %s %s %s %s', request.remote_addr, request.method, request.url, response.status_code, response.content_length or '-')
    return response

# Disable the default Flask request logging since we're handling it ourselves
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Global variables
generation_service = None
MAX_BATCH_SIZE = 4
BACKEND = "vllm"  # Will be set from command line args

@app.route('/load_model', methods=['POST'])
def load_model():
    global generation_service
    
    try:
        # Get model name from request
        data = request.get_json()
        if not data or 'model_name' not in data:
            return jsonify({
                'error': 'model_name parameter is required in JSON body'
            }), 400
        
        model_name = data['model_name']
        num_gpus = data.get('num_gpus', torch.cuda.device_count())
        
        # Unload existing model if any
        if generation_service is not None:
            print_colored("Unloading existing model...", "yellow")
            generation_service.shutdown()
            generation_service = None
        
        # Load new model with backend-specific parameters
        print_colored(f"Loading model: {model_name} with {num_gpus} GPUs using {BACKEND} backend...", "yellow")
        
        if BACKEND == "hf":
            workers_per_gpu = data.get('workers_per_gpu', 1)
            generation_service = GenerationServiceHF(model_name=model_name, num_gpus=num_gpus, workers_per_gpu=workers_per_gpu, max_batch_size=MAX_BATCH_SIZE)
            response_data = {"status": "success", "message": f"Model {model_name} loaded successfully with HF backend",  "backend": "hf", "num_gpus": num_gpus, "workers_per_gpu": workers_per_gpu, "worker_status": generation_service.get_worker_status()}
        elif BACKEND == "vllm":
            gpu_details = calculate_gpu_concurrency()
            print(gpu_details)
            max_concurrent_jobs_per_worker = data.get('max_concurrent_jobs_per_worker', gpu_details["gpu_concurrency"])
            generation_service = GenerationServiceVLLM(model_name=model_name, num_gpus=num_gpus, max_batch_size=MAX_BATCH_SIZE,max_concurrent_jobs_per_worker=max_concurrent_jobs_per_worker)
            response_data = {"status": "success", "message": f"Model {model_name} loaded successfully with vLLM backend", "backend": "vllm", "num_gpus": num_gpus, "max_concurrent_jobs_per_worker": max_concurrent_jobs_per_worker, "worker_status": generation_service.get_worker_status()}
        
        utilization_tracker.start_reporting()        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/what_backend', methods=['GET'])
def what_backend():
    """Return the current backend"""
    return jsonify({"backend": BACKEND})

@app.route('/unload_model', methods=['POST'])
def unload_model():
    """Unload the current model and clean up GPU memory"""
    global generation_service
    
    try:
        if generation_service is None:
            return jsonify({
                'status': 'warning',
                'message': 'No model currently loaded'
            })
        
        # Shutdown the service
        generation_service.shutdown()
        generation_service = None
        
        # Stop utilization tracking
        utilization_tracker.stop_reporting()
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        return jsonify({
            'status': 'success',
            'message': 'Model unloaded successfully'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/schedule_job', methods=['POST'])
def schedule_job():
    """Schedule a generation job"""
    global generation_service
    
    try:
        if generation_service is None:
            return jsonify({
                'error': 'No model loaded. Use /load_model first.'
            }), 400
        
        # Get job parameters from request
        data = request.get_json()
        if not data or 'conversation' not in data:
            return jsonify({
                'error': 'conversation parameter is required in JSON body'
            }), 400
        
        conversation = data['conversation']
        n_responses = data.get('n_responses', 4)
        temperature = data.get('temperature', 1.0)
        max_tokens = data.get('max_tokens', 1000)
        gen_kwargs = data.get('gen_kwargs', {})
        
        job_id = generation_service.schedule_job(conversation=conversation, n_responses=n_responses, temperature=temperature, max_tokens=max_tokens, gen_kwargs=gen_kwargs)
        
        return jsonify({'status': 'success', 'job_id': job_id, 'message': 'Job scheduled successfully'})
        
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/build_tree', methods=['POST'])
def build_tree():
    """Schedule a tree building job"""
    global generation_service
    
    try:
        if generation_service is None:
            return jsonify({
                'error': 'No model loaded. Use /load_model first.'
            }), 400
        
        # Check if the service supports tree building
        if not hasattr(generation_service, 'build_tree'):
            return jsonify({
                'error': 'Tree building not supported by the current backend'
            }), 400
        
        # Get job parameters from request
        data = request.get_json()
        if not data or 'conversation' not in data:
            return jsonify({
                'error': 'conversation parameter is required in JSON body'
            }), 400
        
        conversation = data['conversation']
        degree = data.get('degree', 2)
        depth = data.get('depth', 3)
        temperature = data.get('temperature', 1.0)
        max_tokens = data.get('max_tokens', 1000)
        logprobs = data.get('logprobs', 1)
        
        job_id = generation_service.build_tree(conversation=conversation, degree=degree, depth=depth, temperature=temperature, max_tokens=max_tokens, logprobs=logprobs)
        
        return jsonify({'status': 'success', 'job_id': job_id, 'message': 'Tree building job scheduled successfully'})
        
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/check_on_job', methods=['GET'])
def check_on_job():
    """Check the status of a job"""
    global generation_service
    
    try:
        if generation_service is None:
            return jsonify({'error': 'No model loaded. Use /load_model first.'}), 400
        job_id = request.args.get('job_id')
        if not job_id:
            return jsonify({'error': 'job_id parameter is required'}), 400

        result = generation_service.check_on_job(job_id)
        if result.get('status') == 'completed':
            if 'responses' in result:
                utilization_tracker.track_completed_request(job_id, result['responses'])
            elif 'tree' in result:
                utilization_tracker.track_completed_request(job_id, result['tree'])
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Get overall service status"""
    global generation_service
    
    try:
        if generation_service is None:
            return jsonify({
                'service_loaded': False,
                'backend': BACKEND,
                'message': 'No model currently loaded'
            })
        
        status = generation_service.get_status()
        worker_status = generation_service.get_worker_status()
        
        return jsonify({'service_loaded': True, 'backend': BACKEND, 'service_status': status, 'worker_status': worker_status})
        
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "backend": BACKEND, "service_loaded": generation_service is not None})

@app.route('/utilization', methods=['GET'])
def get_utilization():
    """Get utilization statistics"""
    try:
        # Get time window from query parameters (default 30 seconds)
        time_window = float(request.args.get('time_window', 30))
        if time_window <= 0:
            time_window = 30
        
        stats = utilization_tracker.get_throughput_stats(time_window)
        
        # Add worker status information
        worker_info = utilization_tracker._get_worker_status()
        stats['worker_status'] = worker_info if worker_info else []
        
        return jsonify({"status": "success", "utilization": stats})
        
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

def cleanup():
    global generation_service
    if generation_service is not None:
        generation_service.shutdown()
        generation_service = None
    
    # Stop utilization tracking
    utilization_tracker.stop_reporting()

# Register cleanup function
import atexit
atexit.register(cleanup)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Flask Generation Service with support for multiple backends')
    parser.add_argument("--port", type=int, default=5000, help="Port to run the service on")
    parser.add_argument("--max_batch_size", type=int, default=2, help="Maximum batch size for generation")
    parser.add_argument("--backend", choices=["hf", "vllm"], default="vllm", 
                       help="Backend to use for generation: 'hf' for HuggingFace or 'vllm' for vLLM")
    args = parser.parse_args()

    # Set global variables
    MAX_BATCH_SIZE = args.max_batch_size
    BACKEND = args.backend

    print_colored(f"Using {BACKEND} backend", "green")

    # Set multiprocessing start method (important for CUDA)
    multiprocessing.set_start_method('spawn', force=True)
    
    print(f"Starting Flask Generation Service with {BACKEND} backend...")
    print("Available endpoints:")
    print("  POST /load_model - Load a model")
    print("  POST /unload_model - Unload current model")
    print("  POST /schedule_job - Schedule a generation job")
    print("  POST /build_tree - Schedule a tree building job")
    print("  GET /check_on_job?job_id=<id> - Check job status")
    print("  GET /status - Get service status")
    print("  GET /health - Health check")
    print("  GET /utilization?time_window=<seconds> - Get utilization statistics")
    print(f"\nStarting server on http://localhost:{args.port}")
    print("📊 Utilization tracking will start when a model is loaded (reports every 30 seconds)")
    
    app.run(host='localhost', port=args.port, debug=True, use_reloader=False) 
