import multiprocessing, queue, uuid, json, time, os, torch, setproctitle
from model_generator_hf import GenerationModel
from typing import Dict, Any, List

def worker_process(worker_id: int, gpu_id: int, model_name: str, max_batch_size: int, job_queue, shared_jobs, 
                   shared_worker_busy, shared_gpu_busy, shared_models_loaded, shutdown_event, worker_ready_event):
    """Worker process for a specific GPU"""
    
    # Set process title for easier identification
    try:
        setproctitle.setproctitle(f"generation_worker_{worker_id}_gpu_{gpu_id}")
    except ImportError:
        pass
    
    model = None
    try:
        # print(f"[Worker {worker_id}, GPU {gpu_id}] Worker process started (PID: {os.getpid()}), waiting for load signal...")
        
        # Wait for signal to start loading model
        worker_ready_event.wait()
        
        # Initialize model on this specific GPU
        device = f"cuda:{gpu_id}"
        # print(f"[Worker {worker_id}, GPU {gpu_id}] Loading model on {device}...")
        
        try:
            model = GenerationModel(model_name, device=device, max_batch_size=max_batch_size)
            # print(f"[Worker {worker_id}, GPU {gpu_id}] ✓ Model loaded successfully on {device}")
            
            # Mark this GPU's model as loaded (only need to set once per GPU)
            shared_models_loaded[gpu_id] = True
                
        except Exception as e:
            # print(f"[Worker {worker_id}, GPU {gpu_id}] ✗ Failed to load model: {e}")
            # print(f"[Worker {worker_id}, GPU {gpu_id}] Worker exiting due to model loading failure")
            return  # Exit this worker if model loading fails
        
        # print(f"[Worker {worker_id}, GPU {gpu_id}] Worker ready to process jobs")
        
        while not shutdown_event.is_set():
            try:
                # Get job from queue with timeout to check shutdown flag periodically
                job_id = job_queue.get(timeout=1.0)
                
                # Check shutdown flag after getting job
                if shutdown_event.is_set():
                    break
                
                # print(f"[Worker {worker_id}, GPU {gpu_id}] Processing job {job_id}")
                
                # Mark worker and GPU as busy
                shared_worker_busy[worker_id] = True
                shared_gpu_busy[gpu_id] = True
                
                # Update job status
                if job_id in shared_jobs:
                    job_info = shared_jobs[job_id]
                    job_info["status"] = "in_progress"
                    job_info["worker_id"] = worker_id
                    job_info["gpu_id"] = gpu_id
                    job_info["start_time"] = time.time()
                    shared_jobs[job_id] = job_info
                
                # Execute the job
                try:
                    job_info = shared_jobs.get(job_id)
                    
                    if job_info:
                        responses = model.generate_batch(
                            conversation=job_info["conversation"], 
                            n_responses=job_info["n_responses"], 
                            temperature=job_info["temperature"], 
                            max_tokens=job_info["max_tokens"],
                            **job_info.get("gen_kwargs", {})
                        )
                        
                        # Update job with results
                        if job_id in shared_jobs:
                            job_info = shared_jobs[job_id]
                            job_info["status"] = "completed"
                            job_info["responses"] = responses
                            job_info["end_time"] = time.time()
                            shared_jobs[job_id] = job_info
                        
                        # print(f"[Worker {worker_id}, GPU {gpu_id}] ✓ Completed job {job_id}")
                
                except Exception as e:
                    # print(f"[Worker {worker_id}, GPU {gpu_id}] ✗ Error processing job {job_id}: {e}")
                    if job_id in shared_jobs:
                        job_info = shared_jobs[job_id]
                        job_info["status"] = "error"
                        job_info["error"] = str(e)
                        job_info["end_time"] = time.time()
                        shared_jobs[job_id] = job_info
                
                # Mark worker as free
                shared_worker_busy[worker_id] = False
                
                # Check if any other workers on this GPU are still busy
                workers_on_this_gpu = [i for i in range(len(shared_worker_busy)) 
                                     if i // (len(shared_worker_busy) // len(shared_gpu_busy)) == gpu_id]
                gpu_still_busy = any(shared_worker_busy[i] for i in workers_on_this_gpu)
                shared_gpu_busy[gpu_id] = gpu_still_busy
                
                # Mark task as done
                try:
                    job_queue.task_done()
                except Exception:
                    # Queue might be closed during shutdown
                    pass
                
            except queue.Empty:
                # Timeout occurred, continue to check shutdown flag
                continue
            except Exception as e:
                # Handle any unexpected errors
                # print(f"[Worker {worker_id}, GPU {gpu_id}] ✗ Worker error: {e}")
                shared_worker_busy[worker_id] = False
                
    except Exception as e:
        print(f"[Worker {worker_id}, GPU {gpu_id}] ✗ Critical worker error: {e}")
        
    finally:
        # Cleanup: delete model and clear GPU cache
        if model is not None:
            # print(f"[Worker {worker_id}, GPU {gpu_id}] Cleaning up model...")
            del model
            
            # Clear GPU cache for this device
            with torch.cuda.device(f"cuda:{gpu_id}"):
                torch.cuda.empty_cache()
                
            # print(f"[Worker {worker_id}, GPU {gpu_id}] GPU memory cleared")
        
        shared_worker_busy[worker_id] = False
        # print(f"[Worker {worker_id}, GPU {gpu_id}] Worker process exiting")


class GenerationServiceHF:
    def __init__(self, model_name: str, num_gpus: int = torch.cuda.device_count(), workers_per_gpu: int = 1, max_batch_size: int = 2):
        self.model_name = model_name
        self.num_gpus = num_gpus
        self.workers_per_gpu = workers_per_gpu
        self.max_batch_size = max_batch_size
        self.total_workers = num_gpus * workers_per_gpu
        
        # Check available GPUs
        if num_gpus > torch.cuda.device_count():
            raise ValueError(f"Requested {num_gpus} GPUs but only {torch.cuda.device_count()} available")
        
        # Initialize multiprocessing components
        self.manager = multiprocessing.Manager()
        self.shutdown_event = multiprocessing.Event()
        self.job_queue = multiprocessing.JoinableQueue()
        
        # Shared state across processes
        self.shared_jobs = self.manager.dict()  # job_id -> job_info
        self.shared_worker_busy = self.manager.list([False] * self.total_workers)  # Per-worker busy status
        self.shared_gpu_busy = self.manager.list([False] * num_gpus)  # Per-GPU busy status (any worker busy)
        self.shared_models_loaded = self.manager.list([False] * num_gpus)  # Per-GPU model loaded status
        
        # Worker processes and ready events (now per-worker, not per-GPU)
        self.gpu_processes = []
        self.worker_ready_events = [multiprocessing.Event() for _ in range(self.total_workers)]
        
        print(f"Initializing {self.total_workers} worker processes ({workers_per_gpu} per GPU) with model {model_name}...")
        
        # Start worker processes
        for worker_id in range(self.total_workers):
            gpu_id = worker_id // workers_per_gpu  # Map worker to GPU
            process = multiprocessing.Process(
                target=worker_process,
                args=(
                    worker_id,  # Add worker_id as first argument
                    gpu_id, 
                    model_name, 
                    self.max_batch_size, 
                    self.job_queue, 
                    self.shared_jobs, 
                    self.shared_worker_busy,
                    self.shared_gpu_busy,
                    self.shared_models_loaded, 
                    self.shutdown_event, 
                    self.worker_ready_events[worker_id]
                ),
                daemon=False
            )
            process.start()
            self.gpu_processes.append(process)
        
        # Load models in parallel (signal workers by GPU)
        print("Starting parallel model loading...")
        
        # Signal all workers to start loading models simultaneously
        for worker_id in range(self.total_workers):
            self.worker_ready_events[worker_id].set()
        
        # Wait for all models to load (check per-GPU)
        max_wait = 120  # 2 minutes timeout total
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            loaded_count = sum(self.shared_models_loaded)
            if loaded_count == num_gpus:
                print(f"✓ All {num_gpus} GPU models loaded successfully with {self.total_workers} total workers!")
                break
            
            # Print progress
            if int(time.time() - start_time) % 10 == 0:  # Every 10 seconds
                print(f"Loading progress: {loaded_count}/{num_gpus} GPU models loaded...")
            
            time.sleep(1)
        else:
            loaded_count = sum(self.shared_models_loaded)
            print(f"⚠ Warning: Only {loaded_count}/{num_gpus} GPU models loaded after {max_wait}s timeout")
            
            # Show which GPUs failed to load
            for gpu_id in range(num_gpus):
                if not self.shared_models_loaded[gpu_id]:
                    print(f"  - GPU {gpu_id}: Failed to load model")
        
        print("Generation service initialized!")
        
        # Final status check
        loaded_count = sum(self.shared_models_loaded)
        print(f"Successfully loaded models on {loaded_count}/{num_gpus} GPUs with {self.total_workers} total workers")
    
    def schedule_job(self, conversation: List[Dict], n_responses: int = 4, 
                    temperature: float = 1.0, max_tokens: int = 1000, gen_kwargs: Dict[str, Any] = {}) -> str:
        """Schedule a generation job and return unique job_id"""
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Create job info
        job_info = {
            "conversation": conversation,
            "n_responses": n_responses,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "gen_kwargs": gen_kwargs,
            "status": "queued",
            "responses": None,
            "gpu_id": None,
            "created_time": time.time(),
            "start_time": None,
            "end_time": None,
            "error": None
        }
        
        # Store job info in shared state
        self.shared_jobs[job_id] = job_info
        
        # Add to queue
        self.job_queue.put(job_id)
        
        return job_id
    
    def check_on_job(self, job_id: str) -> Dict[str, Any]:
        """Check status of a job"""
        if job_id not in self.shared_jobs:
            return {"status": "not_found"}
        
        job_info = dict(self.shared_jobs[job_id])  # Create a copy
        
        result = {"status": job_info["status"]}
        
        if job_info["status"] == "completed":
            result["responses"] = job_info["responses"]
        elif job_info["status"] == "error":
            result["error"] = job_info["error"]
        
        # Add timing information if available
        if job_info.get("start_time") and job_info.get("end_time"):
            result["processing_time"] = (job_info.get("end_time") - job_info["start_time"])
        
        return result
    
    def get_worker_status(self) -> Dict[str, Any]:
        """Get detailed worker status for debugging"""
        worker_status = {}
        
        # Per-worker status
        for worker_id in range(self.total_workers):
            gpu_id = worker_id // self.workers_per_gpu
            worker_status[f"worker_{worker_id}"] = {
                "gpu_id": gpu_id,
                "process_alive": self.gpu_processes[worker_id].is_alive() if worker_id < len(self.gpu_processes) else False,
                "process_pid": self.gpu_processes[worker_id].pid if worker_id < len(self.gpu_processes) else None,
                "worker_busy": bool(self.shared_worker_busy[worker_id]),
            }
        
        # Per-GPU status
        gpu_status = {}
        for gpu_id in range(self.num_gpus):
            workers_on_gpu = [i for i in range(self.total_workers) if i // self.workers_per_gpu == gpu_id]
            gpu_status[f"gpu_{gpu_id}"] = {
                "model_loaded": bool(self.shared_models_loaded[gpu_id]),
                "gpu_busy": bool(self.shared_gpu_busy[gpu_id]),
                "worker_ids": workers_on_gpu,
                "busy_workers": [i for i in workers_on_gpu if self.shared_worker_busy[i]],
                "available_workers": len([i for i in workers_on_gpu if not self.shared_worker_busy[i]])
            }
        
        return {
            "workers": worker_status,
            "gpus": gpu_status
        }

    def wait_for_workers_ready(self, timeout: float = 30.0):
        """Wait for all workers to be ready (useful for debugging)"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            alive_workers = sum(1 for process in self.gpu_processes if process.is_alive())
            if alive_workers == self.total_workers:
                print(f"All {self.total_workers} worker processes are alive and running")
                return True
            time.sleep(1)
        
        alive_workers = sum(1 for process in self.gpu_processes if process.is_alive())
        print(f"Warning: Only {alive_workers}/{self.total_workers} worker processes are alive after {timeout}s")
        return False

    def get_status(self) -> Dict[str, Any]:
        """Get overall service status"""
        total_jobs = len(self.shared_jobs)
        completed_jobs = sum(1 for job in self.shared_jobs.values() if job["status"] == "completed")
        in_progress_jobs = sum(1 for job in self.shared_jobs.values() if job["status"] == "in_progress")
        queued_jobs = sum(1 for job in self.shared_jobs.values() if job["status"] == "queued")
        
        return {
            "num_gpus": self.num_gpus,
            "workers_per_gpu": self.workers_per_gpu,
            "total_workers": self.total_workers,
            "gpu_busy": list(self.shared_gpu_busy),
            "worker_busy": list(self.shared_worker_busy),
            "available_workers": sum(1 for busy in self.shared_worker_busy if not busy),
            "queue_size": self.job_queue.qsize(),
            "total_jobs": total_jobs,
            "completed_jobs": completed_jobs,
            "in_progress_jobs": in_progress_jobs,
            "queued_jobs": queued_jobs
        }
    
    def wait_for_completion(self):
        """Wait for all jobs in queue to complete"""
        self.job_queue.join()

    def shutdown(self):
        """Gracefully shutdown the service and clean up GPU resources"""
        
        # Signal all workers to stop
        if hasattr(self, 'shutdown_event') and self.shutdown_event:
            self.shutdown_event.set()
        
        # Wait for worker processes to finish (with timeout)
        if hasattr(self, 'gpu_processes'):
            for i, process in enumerate(self.gpu_processes):
                if process and process.is_alive():
                    process.join(timeout=10.0)
                    if process.is_alive():
                        print(f"Warning: Worker process {i} (PID: {process.pid}) did not shut down cleanly, terminating...")
                        process.terminate()
                        process.join(timeout=5.0)
                        if process.is_alive():
                            print(f"Warning: Force killing worker process {i} (PID: {process.pid})")
                            process.kill()
        
        # Clean up manager
        if hasattr(self, 'manager') and self.manager:
            try:
                self.manager.shutdown()
            except Exception as e:
                print(f"Warning: Error shutting down manager: {e}")
            self.manager = None

    def __del__(self):
        """Clean up resources when the service is deleted"""
        try:
            if hasattr(self, 'manager') and self.manager:
                self.shutdown()
        except Exception as e:
            print(f"Error during GenerationServiceHF cleanup: {e}")


if __name__ == "__main__":
    # Set multiprocessing start method (important for CUDA)
    multiprocessing.set_start_method('spawn', force=True)
    
    # Example usage
    print("Creating GenerationServiceHF...")
    models = ["experiments/exp_20250530_1/best_model/"]
    for model in models:
        service = GenerationServiceHF(model_name=model, num_gpus=4, workers_per_gpu=2)  # 2 workers per GPU
 
        print("Worker status:")
        print(json.dumps(service.get_worker_status(), indent=2))
        
        # Schedule some jobs
        conversation = [{"role": "user", "content": "Hello, how are you? Tell me a paragraph-long joke about UC Berkeley. Hello, how are you? Tell me a paragraph-long joke about UC Berkeley. Hello, how are you? Tell me a paragraph-long joke about UC Berkeley. Hello, how are you? Tell me a paragraph-long joke about UC Berkeley. Hello, how are you? Tell me a paragraph-long joke about UC Berkeley. Hello, how are you? Tell me a paragraph-long joke about UC Berkeley. Hello, how are you? Tell me a paragraph-long joke about UC Berkeley. Hello, how are you? Tell me a paragraph-long joke about UC Berkeley. Hello, how are you? Tell me a paragraph-long joke about UC Berkeley. Hello, how are you? Tell me a paragraph-long joke about UC Berkeley. Hello, how are you? Tell me a paragraph-long joke about UC Berkeley. Hello, how are you? Tell me a paragraph-long joke about UC Berkeley. Hello, how are you? Tell me a paragraph-long joke about UC Berkeley. Hello, how are you? Tell me a paragraph-long joke about UC Berkeley. Hello, how are you? Tell me a paragraph-long joke about UC Berkeley. Hello, how are you? Tell me a paragraph-long joke about UC Berkeley. Hello, how are you? Tell me a paragraph-long joke about UC Berkeley. Hello, how are you? Tell me a paragraph-long joke about UC Berkeley. Hello, how are you? Tell me a paragraph-long joke about UC Berkeley. Hello, how are you? Tell me a paragraph-long joke about UC Berkeley. Hello, how are you? Tell me a paragraph-long joke about UC Berkeley. Hello, how are you? Tell me a paragraph-long joke about UC Berkeley. Hello, how are you? Tell me a paragraph-long joke about UC Berkeley. Hello, how are you? Tell me a paragraph-long joke about UC Berkeley. Hello, how are you? Tell me a paragraph-long joke about UC Berkeley. Hello, how are you? Tell me a paragraph-long joke about UC Berkeley. Hello, how are you? Tell me a paragraph-long joke about UC Berkeley. Hello, how are you? Tell me a paragraph-long joke about UC Berkeley. Hello, how are you? Tell me a paragraph-long joke about UC Berkeley. Hello, how are you? Tell me a paragraph-long joke about UC Berkeley. Hello, how are you? Tell me a paragraph-long joke about UC Berkeley. Hello, how are you? Tell me a paragraph-long joke about UC Berkeley. Hello, how are you? Tell me a paragraph-long joke about UC Berkeley. Hello, how are you? Tell me a paragraph-long joke about UC Berkeley. Hello, how are you? Tell me a paragraph-long joke about UC Berkeley. Hello, how are you? Tell me a paragraph-long joke about UC Berkeley. Hello, how are you? Tell me a paragraph-long joke about UC Berkeley. Hello, how are you? Tell me a paragraph-long joke about UC Berkeley."}]
        
        start_time = time.time()
        job_ids = []

        N_JOBS = 10
        print(f"\nScheduling {N_JOBS} jobs...")
        for i in range(N_JOBS):
            job_id = service.schedule_job(conversation, n_responses=4, temperature=1.0)
            job_ids.append(job_id)
            print(f"Scheduled job {i+1}: {job_id}")
        
        print(f"\nInitial service status:")
        print(json.dumps(service.get_status(), indent=2))
        
        # Check on jobs
        print("\nMonitoring jobs...")
        while True:
            all_completed = True
            for job_id in job_ids:
                result = service.check_on_job(job_id)
                print(f"Job {job_id}: {result['status']}")
                if result["status"] not in ["completed", "error"]:
                    all_completed = False
            
            print(f"Service status: {service.get_status()}")
            print("-" * 50)
            
            if all_completed:
                break
            
            time.sleep(2)
        
        total_time = time.time() - start_time
        print(f"Total time: {total_time} seconds")
        
        # Clean shutdown
        service.shutdown()
        del service
        torch.cuda.empty_cache()
        print("Finished deleting the service and empty cache")

        assistant_model = GenerationModel(model_name=model, device=None)
        optimizer = torch.optim.SGD(assistant_model.model.parameters(), lr=2e-5)

        time.sleep(10)
        del assistant_model, optimizer
        torch.cuda.empty_cache()
        print("Finished deleting the single model and empty cache")