import multiprocessing, queue, uuid, json, time, os, setproctitle
from system_agent import SystemAgent
from tasks import get_task, get_all_supported_task_names
from typing import Dict, Any, List

def evaluation_worker_process(worker_id: int, job_queue, shared_jobs, shared_worker_busy, 
                            shutdown_event, worker_ready_event, tasks_dict):
    """Worker process for evaluation tasks"""
    
    # Set process title for easier identification
    try:
        setproctitle.setproctitle(f"evaluation_worker_{worker_id}")
    except ImportError:
        pass
    
    try:
        # print(f"[Eval Worker {worker_id}] Worker process started (PID: {os.getpid()}), waiting for start signal...")
        
        # Wait for signal to start processing
        worker_ready_event.wait()
        
        # print(f"[Eval Worker {worker_id}] Worker ready to process evaluation jobs")
        
        while not shutdown_event.is_set():
            try:
                # Get job from queue with timeout to check shutdown flag periodically
                job_id = job_queue.get(timeout=1.0)
                
                # Check shutdown flag after getting job
                if shutdown_event.is_set():
                    break
                
                # print(f"[Eval Worker {worker_id}] Processing evaluation job {job_id}")
                
                # Mark worker as busy
                shared_worker_busy[worker_id] = True
                
                # Update job status
                if job_id in shared_jobs:
                    job_info = shared_jobs[job_id]
                    job_info["status"] = "in_progress"
                    job_info["worker_id"] = worker_id
                    job_info["start_time"] = time.time()
                    shared_jobs[job_id] = job_info
                
                # Execute the evaluation job
                try:
                    job_info = shared_jobs.get(job_id)
                    
                    if job_info:
                        conversation = job_info["conversation"]
                        task_name = job_info["task_name"]
                        sample = job_info["sample"]
                        
                        # Get task instance
                        task = get_task(task_name)
                        
                        # Create system agent for this task and sample
                        system_agent = SystemAgent(task_name, sample)
                        
                        # Classify assistant response
                        response_strategy = system_agent.classify_assistant_response(conversation)
                        
                        evaluation_result = {
                            "response_strategy": response_strategy,
                            # "extracted_answer": None,
                            # "evaluation_return": None,
                            # "is_correct": None,
                            # "score": None
                        }
                        
                        if response_strategy == "answer_attempt":
                            # Extract answer
                            extracted_answer = system_agent.extract_answer(conversation)
                            evaluation_result["extracted_answer"] = extracted_answer
                            
                            # Evaluate using task's evaluator function

                            # print(f"Extracted answer: {extracted_answer}")
                            # print(f"Sample: {sample}")

                            evaluation_return = task.evaluator_function(extracted_answer, sample)
                            evaluation_result["evaluation_return"] = evaluation_return
                            
                            is_correct = evaluation_return.get("is_correct", None)
                            score = evaluation_return.get("score", None)
                            if is_correct is not None and score is None:
                                score = 1 if is_correct else 0
                            
                            if score == 1.0 and not is_correct:
                                is_correct = True
                            
                            evaluation_result["is_correct"] = is_correct
                            evaluation_result["score"] = score
                        
                        # Update job with results
                        if job_id in shared_jobs:
                            job_info = shared_jobs[job_id]
                            job_info["status"] = "completed"
                            job_info["result"] = evaluation_result
                            job_info["end_time"] = time.time()
                            shared_jobs[job_id] = job_info
                        
                        # print(f"[Eval Worker {worker_id}] ✓ Completed evaluation job {job_id}")
                
                except Exception as e:
                    # print(f"[Eval Worker {worker_id}] ✗ Error processing evaluation job {job_id}: {e}")
                    if job_id in shared_jobs:
                        job_info = shared_jobs[job_id]
                        job_info["status"] = "error"
                        job_info["error"] = str(e)
                        job_info["end_time"] = time.time()
                        shared_jobs[job_id] = job_info
                
                # Mark worker as free
                shared_worker_busy[worker_id] = False
                
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
                # print(f"[Eval Worker {worker_id}] ✗ Worker error: {e}")
                shared_worker_busy[worker_id] = False
                
    except Exception as e:
        print(f"[Eval Worker {worker_id}] ✗ Critical worker error: {e}")
        
    finally:
        shared_worker_busy[worker_id] = False
        # print(f"[Eval Worker {worker_id}] Worker process exiting")


class EvaluationService:
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        
        # Initialize multiprocessing components
        self.manager = multiprocessing.Manager()
        self.shutdown_event = multiprocessing.Event()
        self.job_queue = multiprocessing.JoinableQueue()
        
        # Shared state across processes
        self.shared_jobs = self.manager.dict()  # job_id -> job_info
        self.shared_worker_busy = self.manager.list([False] * self.num_workers)  # Per-worker busy status
        
        # Load tasks
        self.tasks_dict = {}
        
        # Worker processes and ready events
        self.worker_processes = []
        self.worker_ready_events = [multiprocessing.Event() for _ in range(self.num_workers)]
        
        print(f"Initializing {self.num_workers} evaluation worker processes...")
        
        # Start worker processes
        for worker_id in range(self.num_workers):
            process = multiprocessing.Process(
                target=evaluation_worker_process,
                args=(
                    worker_id,
                    self.job_queue,
                    self.shared_jobs,
                    self.shared_worker_busy,
                    self.shutdown_event,
                    self.worker_ready_events[worker_id],
                    self.tasks_dict
                ),
                daemon=False
            )
            process.start()
            self.worker_processes.append(process)
        
        # Load all supported tasks automatically
        self._load_all_tasks()
        
        # Signal workers to start processing
        for worker_id in range(self.num_workers):
            self.worker_ready_events[worker_id].set()
        
        print("✓ All evaluation workers are ready!")
        print("Evaluation service initialized!")
    
    def _load_all_tasks(self):
        """Load all supported tasks automatically"""
        supported_task_names = get_all_supported_task_names()
        print(f"Loading all supported tasks: {supported_task_names}")
        
        for task_name in supported_task_names:
            try:
                task = get_task(task_name)
                self.tasks_dict[task_name] = task
                print(f"✓ Loaded task: {task_name}")
            except Exception as e:
                print(f"✗ Failed to load task {task_name}: {e}")
        
        print(f"✓ Loaded {len(self.tasks_dict)} tasks total")

    def schedule_evaluation(self, conversation: List[Dict], task_name: str, sample: Dict[str, Any]) -> str:
        """Schedule an evaluation job and return unique job_id"""
        
        if not self.tasks_dict:
            raise RuntimeError("No tasks loaded. Check task loading during initialization.")
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Create job info
        job_info = {
            "conversation": conversation,
            "task_name": task_name,
            "sample": sample,
            "status": "queued",
            "result": None,
            "worker_id": None,
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
    
    def check_evaluation_job(self, job_id: str) -> Dict[str, Any]:
        """Check status of an evaluation job"""
        if job_id not in self.shared_jobs:
            return {"status": "not_found"}
        
        job_info = dict(self.shared_jobs[job_id])  # Create a copy
        
        result = {"status": job_info["status"]}
        
        if job_info["status"] == "completed":
            result["result"] = job_info["result"]
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
        for worker_id in range(self.num_workers):
            worker_status[f"worker_{worker_id}"] = {
                "process_alive": self.worker_processes[worker_id].is_alive() if worker_id < len(self.worker_processes) else False,
                "process_pid": self.worker_processes[worker_id].pid if worker_id < len(self.worker_processes) else None,
                "worker_busy": bool(self.shared_worker_busy[worker_id]),
            }
        
        return {"workers": worker_status}

    def get_status(self) -> Dict[str, Any]:
        """Get overall service status"""
        total_jobs = len(self.shared_jobs)
        completed_jobs = sum(1 for job in self.shared_jobs.values() if job["status"] == "completed")
        in_progress_jobs = sum(1 for job in self.shared_jobs.values() if job["status"] == "in_progress")
        queued_jobs = sum(1 for job in self.shared_jobs.values() if job["status"] == "queued")
        
        return {
            "tasks_loaded": bool(self.tasks_dict), # Check if tasks_dict is not empty
            "loaded_tasks": list(self.tasks_dict.keys()) if self.tasks_dict else [],
            "total_samples": 0, # No dataset samples to report
            "num_workers": self.num_workers,
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
        """Gracefully shutdown the service"""
        # print("Shutting down EvaluationService...")
        
        # Signal all workers to stop
        if hasattr(self, 'shutdown_event') and self.shutdown_event:
            self.shutdown_event.set()
        
        # Wait for worker processes to finish (with timeout)
        if hasattr(self, 'worker_processes'):
            for i, process in enumerate(self.worker_processes):
                if process and process.is_alive():
                    process.join(timeout=10.0)
                    if process.is_alive():
                        print(f"Warning: Evaluation worker process {i} (PID: {process.pid}) did not shut down cleanly, terminating...")
                        process.terminate()
                        process.join(timeout=5.0)
                        if process.is_alive():
                            print(f"Warning: Force killing evaluation worker process {i} (PID: {process.pid})")
                            process.kill()
        
        # Clean up manager
        if hasattr(self, 'manager') and self.manager:
            try:
                self.manager.shutdown()
            except Exception as e:
                print(f"Warning: Error shutting down manager: {e}")
            self.manager = None
        
        # print("EvaluationService shutdown complete")

    def __del__(self):
        """Clean up resources when the service is deleted"""
        try:
            if hasattr(self, 'manager') and self.manager:
                self.shutdown()
        except Exception as e:
            print(f"Error during EvaluationService cleanup: {e}")


if __name__ == "__main__":
    # Set multiprocessing start method
    multiprocessing.set_start_method('spawn', force=True)
    
    # Example usage - tasks are loaded automatically on initialization
    print("Creating EvaluationService...")
    service = EvaluationService(num_workers=2)
    
    print("Service status:")
    print(json.dumps(service.get_status(), indent=2))
    
    # Clean shutdown
    service.shutdown()
    print("Service shut down successfully") 
