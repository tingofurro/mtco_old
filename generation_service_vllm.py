import multiprocessing, queue, uuid, json, time, os, torch, setproctitle, asyncio, shutil
from model_generator_vllm import AsyncVLLMGenerationModel
from typing import Dict, Any, List
import threading

async def async_worker_main(gpu_id: int, model_name: str, max_concurrent_jobs: int,
                           job_queue, shared_jobs, shared_gpu_active_jobs, shared_gpu_busy, 
                           shared_models_loaded, shutdown_event, gpu_ready_event):
    """Async main function for worker process"""
    
    model = None
    active_tasks = {}  # job_id -> task
    
    try:
        # Wait for signal to start loading model
        gpu_ready_event.wait()
        
        # Initialize model on this specific GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[GPU {gpu_id}] Loading async vLLM model...")
        
        try:
            model = AsyncVLLMGenerationModel(
                model_name=model_name,
                device=f"cuda:{gpu_id}",
                enable_prefix_caching=True,
                max_context_length=6000
            )
            print(f"[GPU {gpu_id}] ✓ Async vLLM model loaded successfully")
            
            # Mark this GPU's model as loaded
            shared_models_loaded[gpu_id] = True
                
        except Exception as e:
            print(f"[GPU {gpu_id}] ✗ Failed to load model: {e}")
            return
        
        print(f"[GPU {gpu_id}] Worker ready to process jobs (max concurrent: {max_concurrent_jobs})")
        
        while not shutdown_event.is_set():
            try:
                # Clean up completed tasks
                completed_jobs = []
                for job_id, task in list(active_tasks.items()):
                    if task.done():
                        completed_jobs.append(job_id)
                        del active_tasks[job_id]
                
                # Update active job count
                shared_gpu_active_jobs[gpu_id] = len(active_tasks)
                shared_gpu_busy[gpu_id] = len(active_tasks) > 0
                
                # Check if we can accept more jobs
                if len(active_tasks) < max_concurrent_jobs:
                    try:
                        # Get job from queue with short timeout
                        job_id = job_queue.get(timeout=0.1)
                        
                        # Check shutdown flag after getting job
                        if shutdown_event.is_set():
                            break
                        
                        # print(f"[GPU {gpu_id}] Starting job {job_id} (active: {len(active_tasks)})")
                        
                        # Create async task for this job
                        task = asyncio.create_task(process_job_async(
                            gpu_id, job_id, model, shared_jobs
                        ))
                        active_tasks[job_id] = task
                        
                    except queue.Empty:
                        # No jobs available, continue
                        pass
                
                # Small sleep to prevent busy waiting
                await asyncio.sleep(0.01)
                
            except Exception as e:
                print(f"[GPU {gpu_id}] ✗ Worker loop error: {e}")
                await asyncio.sleep(0.1)
        
        # Wait for all active tasks to complete during shutdown
        if active_tasks:
            print(f"[GPU {gpu_id}] Waiting for {len(active_tasks)} active jobs to complete...")
            await asyncio.gather(*active_tasks.values(), return_exceptions=True)
        
    except Exception as e:
        print(f"[GPU {gpu_id}] ✗ Critical worker error: {e}")
        
    finally:
        # Cleanup
        if model is not None:
            print(f"[GPU {gpu_id}] Cleaning up model...")
            try:
                await model.shutdown()
            except Exception as e:
                print(f"[GPU {gpu_id}] Warning during model shutdown: {e}")
            del model
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            print(f"[GPU {gpu_id}] GPU memory cleared")
        
        shared_gpu_active_jobs[gpu_id] = 0
        print(f"[GPU {gpu_id}] Worker process exiting")


async def process_job_async(gpu_id: int, job_id: str, model: AsyncVLLMGenerationModel, shared_jobs):
    """Process a single job asynchronously"""
    try:
        # Update job status
        if job_id in shared_jobs:
            job_info = shared_jobs[job_id]
            job_info["status"] = "in_progress"
            job_info["gpu_id"] = gpu_id
            job_info["start_time"] = time.time()
            shared_jobs[job_id] = job_info
        
        # Get job info
        job_info = shared_jobs.get(job_id)
        if not job_info:
            return
        
        # Check job type and process accordingly
        if job_info.get("job_type") == "build_tree":
            # Execute build_tree operation
            tree = await model.build_tree(
                conversation=job_info["conversation"],
                degree=job_info.get("degree", 2),
                depth=job_info.get("depth", 3),
                temperature=job_info.get("temperature", 1.0),
                max_tokens=job_info.get("max_tokens", 1000),
                logprobs=job_info.get("logprobs", 1)
            )
            
            # Update job with results
            if job_id in shared_jobs:
                job_info = shared_jobs[job_id]
                job_info["status"] = "completed"
                job_info["tree"] = tree
                job_info["end_time"] = time.time()
                shared_jobs[job_id] = job_info
        else:
            # Execute the regular generation job using async vLLM
            responses = await model.generate_batch_async(
                conversations=[job_info["conversation"]], 
                n_responses_per_conv=job_info["n_responses"], 
                temperature=job_info["temperature"], 
                max_tokens=job_info["max_tokens"],
                **job_info.get("gen_kwargs", {})
            )
            
            # Update job with results
            if job_id in shared_jobs:
                job_info = shared_jobs[job_id]
                job_info["status"] = "completed"
                job_info["responses"] = responses[0] if responses else []  # First conversation's responses
                job_info["end_time"] = time.time()
                shared_jobs[job_id] = job_info
        
        # print(f"[GPU {gpu_id}] ✓ Completed job {job_id}")
        
    except Exception as e:
        print(f"[GPU {gpu_id}] ✗ Error processing job {job_id}: {e}")
        if job_id in shared_jobs:
            job_info = shared_jobs[job_id]
            job_info["status"] = "error"
            job_info["error"] = str(e)
            job_info["end_time"] = time.time()
            shared_jobs[job_id] = job_info


def worker_process(gpu_id: int, model_name: str, max_concurrent_jobs: int,
                   job_queue, shared_jobs, shared_gpu_active_jobs, shared_gpu_busy, 
                   shared_models_loaded, shutdown_event, gpu_ready_event):
    """Worker process entry point that runs the async event loop"""
    
    # Set unique cache directory for this GPU worker to prevent concurrency issues
    gpu_cache_dir = os.path.expanduser(f"~/.cache/vllm_gpu_{gpu_id}")
    os.environ["VLLM_CACHE_ROOT"] = gpu_cache_dir
    print(f"[GPU {gpu_id}] Using cache directory: {gpu_cache_dir}")
    
    # Create cache directory if it doesn't exist
    os.makedirs(gpu_cache_dir, exist_ok=True)
    
    # Set process title for easier identification
    try:
        setproctitle.setproctitle(f"vllm_gpu_{gpu_id}")
    except ImportError:
        pass
    
    print(f"[GPU {gpu_id}] Worker process started (PID: {os.getpid()})")
    
    # Run the async worker main
    try:
        asyncio.run(async_worker_main(
            gpu_id, model_name, max_concurrent_jobs,
            job_queue, shared_jobs, shared_gpu_active_jobs, shared_gpu_busy,
            shared_models_loaded, shutdown_event, gpu_ready_event
        ))
    except Exception as e:
        print(f"[GPU {gpu_id}] ✗ Critical error in async event loop: {e}")


class GenerationServiceVLLM:
    def __init__(self, model_name: str, num_gpus: int = torch.cuda.device_count(), 
                 max_batch_size: int = 2, max_concurrent_jobs_per_worker: int = 10):
        self.model_name = model_name
        self.num_gpus = num_gpus
        self.max_batch_size = max_batch_size  # Not used in vLLM but kept for compatibility
        self.max_concurrent_jobs_per_worker = max_concurrent_jobs_per_worker
        self.total_workers = num_gpus  # One worker per GPU
        
        # Check available GPUs
        if num_gpus > torch.cuda.device_count():
            raise ValueError(f"Requested {num_gpus} GPUs but only {torch.cuda.device_count()} available")
        
        # Initialize multiprocessing components
        self.manager = multiprocessing.Manager()
        self.shutdown_event = multiprocessing.Event()
        self.job_queue = multiprocessing.JoinableQueue()
        
        # Shared state across processes
        self.shared_jobs = self.manager.dict()  # job_id -> job_info
        self.shared_gpu_active_jobs = self.manager.list([0] * self.total_workers)  # Per-GPU active job count
        self.shared_gpu_busy = self.manager.list([False] * num_gpus)  # Per-GPU busy status
        self.shared_models_loaded = self.manager.list([False] * num_gpus)  # Per-GPU model loaded status
        
        # GPU processes and ready events
        self.gpu_processes = []
        self.gpu_ready_events = [multiprocessing.Event() for _ in range(self.total_workers)]
        
        print(f"Initializing {self.total_workers} async vLLM worker processes (1 per GPU) with model {model_name}...")
        print(f"Max concurrent jobs per worker: {max_concurrent_jobs_per_worker}")
        
        # Start worker processes
        for gpu_id in range(self.total_workers):
            process = multiprocessing.Process(
                target=worker_process,
                args=(
                    gpu_id, 
                    model_name, 
                    max_concurrent_jobs_per_worker,
                    self.job_queue, 
                    self.shared_jobs, 
                    self.shared_gpu_active_jobs,
                    self.shared_gpu_busy,
                    self.shared_models_loaded, 
                    self.shutdown_event, 
                    self.gpu_ready_events[gpu_id]
                ),
                daemon=False
            )
            process.start()
            self.gpu_processes.append(process)
        
        # Load models in parallel
        print("Starting parallel async vLLM model loading...")
        
        # Signal all workers to start loading models simultaneously
        for gpu_id in range(self.total_workers):
            self.gpu_ready_events[gpu_id].set()
        
        # Wait for all models to load
        max_wait = 300  # 5 minutes timeout for vLLM loading
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            loaded_count = sum(self.shared_models_loaded)
            if loaded_count == num_gpus:
                print(f"✓ All {num_gpus} async vLLM models loaded successfully with {self.total_workers} total workers!")
                break
            
            # Print progress every 15 seconds
            if int(time.time() - start_time) % 15 == 0:
                print(f"Loading progress: {loaded_count}/{num_gpus} GPU models loaded...")
            
            time.sleep(1)
        else:
            loaded_count = sum(self.shared_models_loaded)
            print(f"⚠ Warning: Only {loaded_count}/{num_gpus} GPU models loaded after {max_wait}s timeout")
            
            # Show which GPUs failed to load
            for gpu_id in range(num_gpus):
                if not self.shared_models_loaded[gpu_id]:
                    print(f"  - GPU {gpu_id}: Failed to load model")
        
        print("Async vLLM generation service initialized!")
        
        # Final status check
        loaded_count = sum(self.shared_models_loaded)
        print(f"Successfully loaded async vLLM models on {loaded_count}/{num_gpus} GPUs with {self.total_workers} total workers")
    
    def _get_least_busy_gpu(self) -> int:
        """Get the GPU ID with the least number of active jobs for load balancing"""
        min_jobs = float('inf')
        best_gpu = 0
        
        for gpu_id in range(self.total_workers):
            active_jobs = self.shared_gpu_active_jobs[gpu_id]
            if active_jobs < min_jobs:
                min_jobs = active_jobs
                best_gpu = gpu_id
                
                # If we find a completely free GPU, use it immediately
                if min_jobs == 0:
                    break
        
        return best_gpu
    
    def schedule_job(self, conversation: List[Dict], n_responses: int = 4, 
                    temperature: float = 1.0, max_tokens: int = 1000, gen_kwargs: Dict[str, Any] = {}) -> str:
        """Schedule a generation job and return unique job_id"""
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Create job info
        job_info = {"job_type": "generation", "conversation": conversation, "n_responses": n_responses, "temperature": temperature, "max_tokens": max_tokens, "gen_kwargs": gen_kwargs, "status": "queued", "responses": None, "gpu_id": None, "created_time": time.time(), "start_time": None, "end_time": None, "error": None}
        
        # Store job info in shared state
        self.shared_jobs[job_id] = job_info
        
        # Add to queue for load balancing
        # Note: We could implement more sophisticated load balancing here
        # by routing directly to specific workers, but using a shared queue
        # is simpler and still allows workers to self-balance
        self.job_queue.put(job_id)
        
        return job_id

    def build_tree(self, conversation: List[Dict], degree: int = 2, depth: int = 3,
                  temperature: float = 1.0, max_tokens: int = 1000, logprobs: int = 1) -> str:
        """Schedule a tree building job and return unique job_id"""
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Create job info for tree building
        job_info = {"job_type": "build_tree", "conversation": conversation, "degree": degree, "depth": depth, "temperature": temperature, "max_tokens": max_tokens, "logprobs": logprobs, "status": "queued", "tree": None, "gpu_id": None, "created_time": time.time(), "start_time": None, "end_time": None, "error": None}
        
        # Store job info in shared state
        self.shared_jobs[job_id] = job_info
        
        # Add to queue
        self.job_queue.put(job_id)
        
        return job_id

    def build_tree_sync(self, conversation: List[Dict], degree: int = 2, depth: int = 3,
                       temperature: float = 1.0, max_tokens: int = 1000, logprobs: int = 1):
        """Build a tree synchronously and return the result directly"""
        
        # Schedule the tree building job
        job_id = self.build_tree(conversation, degree, depth, temperature, max_tokens, logprobs)
        
        # Wait for completion
        while True:
            result = self.check_on_job(job_id)
            if result["status"] == "completed":
                return result["tree"]
            elif result["status"] == "error":
                raise Exception(f"Tree building failed: {result['error']}")
            
            time.sleep(0.1)
    
    def check_on_job(self, job_id: str) -> Dict[str, Any]:
        """Check status of a job"""
        if job_id not in self.shared_jobs:
            return {"status": "not_found"}
        
        job_info = dict(self.shared_jobs[job_id])  # Create a copy
        
        result = {"status": job_info["status"]}
        
        if job_info["status"] == "completed":
            if job_info.get("job_type") == "build_tree":
                result["tree"] = job_info["tree"]
            else:
                result["responses"] = job_info["responses"]
        elif job_info["status"] == "error":
            result["error"] = job_info["error"]
        
        # Add timing information if available
        if job_info.get("start_time") and job_info.get("end_time"):
            result["processing_time"] = (job_info.get("end_time") - job_info["start_time"])
        
        return result
    
    def get_worker_status(self) -> Dict[str, Any]:
        """Get detailed GPU status for debugging"""
        gpu_status = {}
        
        # Per-GPU status (combined worker and model status)
        for gpu_id in range(self.total_workers):
            total_active_jobs = self.shared_gpu_active_jobs[gpu_id]
            total_capacity = self.max_concurrent_jobs_per_worker
            
            gpu_status[f"gpu_{gpu_id}"] = {
                "process_alive": self.gpu_processes[gpu_id].is_alive() if gpu_id < len(self.gpu_processes) else False,
                "process_pid": self.gpu_processes[gpu_id].pid if gpu_id < len(self.gpu_processes) else None,
                "model_loaded": bool(self.shared_models_loaded[gpu_id]),
                "gpu_busy": bool(self.shared_gpu_busy[gpu_id]),
                "active_jobs": int(total_active_jobs),
                "max_concurrent_jobs": self.max_concurrent_jobs_per_worker,
                "capacity_used": f"{total_active_jobs}/{total_capacity}",
                "utilization": f"{total_active_jobs}/{total_capacity}"
            }
        
        return {
            "gpus": gpu_status
        }

    def get_status(self) -> Dict[str, Any]:
        """Get overall service status"""
        total_jobs = len(self.shared_jobs)
        completed_jobs = sum(1 for job in self.shared_jobs.values() if job["status"] == "completed")
        in_progress_jobs = sum(1 for job in self.shared_jobs.values() if job["status"] == "in_progress")
        queued_jobs = sum(1 for job in self.shared_jobs.values() if job["status"] == "queued")
        
        total_active_jobs = sum(self.shared_gpu_active_jobs)
        total_capacity = self.total_workers * self.max_concurrent_jobs_per_worker
        
        return {
            "num_gpus": self.num_gpus,
            "total_workers": self.total_workers,
            "max_concurrent_jobs_per_worker": self.max_concurrent_jobs_per_worker,
            "total_capacity": total_capacity,
            "total_active_jobs": total_active_jobs,
            "gpu_busy": list(self.shared_gpu_busy),
            "gpu_active_jobs": list(self.shared_gpu_active_jobs),
            "queue_size": self.job_queue.qsize(),
            "total_jobs": total_jobs,
            "completed_jobs": completed_jobs,
            "in_progress_jobs": in_progress_jobs,
            "queued_jobs": queued_jobs,
            "utilization": f"{total_active_jobs}/{total_capacity}"
        }
    
    def wait_for_completion(self):
        """Wait for all jobs in queue to complete"""
        # For async workers, we need to wait until all active jobs are done
        # This is more complex than the sync version
        while True:
            total_active = sum(self.shared_gpu_active_jobs)
            queue_size = self.job_queue.qsize()
            
            if total_active == 0 and queue_size == 0:
                break
                
            time.sleep(0.1)

    def shutdown(self):
        """Gracefully shutdown the service and clean up GPU resources"""
        print("Shutting down async vLLM GenerationServiceVLLM...")
        
        # Signal all workers to stop
        if hasattr(self, 'shutdown_event') and self.shutdown_event:
            self.shutdown_event.set()
        
        # Wait for worker processes to finish (with timeout)
        if hasattr(self, 'gpu_processes'):
            for i, process in enumerate(self.gpu_processes):
                if process and process.is_alive():
                    print(f"Waiting for worker process {i} to shutdown...")
                    process.join(timeout=15.0)  # Longer timeout for async cleanup
                    if process.is_alive():
                        print(f"Warning: Worker process {i} (PID: {process.pid}) did not shut down cleanly, terminating...")
                        process.terminate()
                        process.join(timeout=5.0)
                        if process.is_alive():
                            print(f"Warning: Force killing worker process {i} (PID: {process.pid})")
                            process.kill()
        
        # Clean up per-GPU cache directories to prevent corruption issues
        for gpu_id in range(self.num_gpus):
            gpu_cache_dir = os.path.expanduser(f"~/.cache/vllm_gpu_{gpu_id}")
            if os.path.exists(gpu_cache_dir):
                try:
                    print(f"Cleaning up GPU {gpu_id} cache directory: {gpu_cache_dir}")
                    shutil.rmtree(gpu_cache_dir)
                except Exception as e:
                    print(f"Warning: Failed to clean up GPU {gpu_id} cache: {e}")
        
        # Clean up manager
        if hasattr(self, 'manager') and self.manager:
            try:
                self.manager.shutdown()
            except Exception as e:
                print(f"Warning: Error shutting down manager: {e}")
            self.manager = None
        
        print("Async vLLM GenerationServiceVLLM shutdown complete")

    def __del__(self):
        """Clean up resources when the service is deleted"""
        try:
            if hasattr(self, 'manager') and self.manager:
                self.shutdown()
        except Exception as e:
            print(f"Error during async vLLM GenerationServiceVLLM cleanup: {e}")


if __name__ == "__main__":
    # Set multiprocessing start method (important for CUDA)
    multiprocessing.set_start_method('spawn', force=True)
    
    # Example usage
    print("Creating async vLLM GenerationServiceVLLM...")
    
    service = GenerationServiceVLLM(
        model_name="microsoft/phi-4", 
        num_gpus=2, 
        max_concurrent_jobs_per_worker=10
    )

    print("Worker status:")
    print(json.dumps(service.get_worker_status(), indent=2))
    
    # Schedule some jobs
    conversation = [{"role": "user", "content": "Write a 500-word poem about AI and creativity."}]
    
    start_time = time.time()
    job_ids = []

    N_JOBS = 40
    print(f"\nScheduling {N_JOBS} jobs...")
    for i in range(N_JOBS):
        job_id = service.schedule_job(conversation, n_responses=2, temperature=0.8, max_tokens=1000)
        job_ids.append(job_id)
        print(f"Scheduled job {i+1}: {job_id}")
    
    print(f"\nInitial service status:")
    print(json.dumps(service.get_status(), indent=2))
    
    # Check on jobs
    print("\nMonitoring jobs...")
    completed_count = 0
    while completed_count < N_JOBS:
        completed_count = 0
        for job_id in job_ids:
            result = service.check_on_job(job_id)
            if result["status"] in ["completed", "error"]:
                completed_count += 1
        
        status = service.get_status()
        print(f"Completed: {completed_count}/{N_JOBS}, Active: {status['total_active_jobs']}, Queue: {status['queue_size']}")
        
        if completed_count < N_JOBS:
            time.sleep(2)
    
    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per job: {total_time/N_JOBS:.2f} seconds")
    
    # Collect all generated poems
    print("\nCollecting generated poems...")
    all_poems = []
    for i, job_id in enumerate(job_ids):
        result = service.check_on_job(job_id)
        if result["status"] == "completed":
            job_data = {
                "job_id": job_id,
                "job_number": i + 1,
                "responses": result["responses"],
                "processing_time": result.get("processing_time", None)
            }
            all_poems.append(job_data)
        else:
            print(f"Warning: Job {i+1} ({job_id}) status: {result['status']}")
    
    output_file = "data/temp_poems.json"    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_poems, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved {len(all_poems)} completed jobs with poems to {output_file}")
    
    # Clean shutdown
    service.shutdown()
    print("Async vLLM service test completed!")