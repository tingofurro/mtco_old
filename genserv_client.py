import requests
import json
import time
from typing import Dict, Any, List, Optional

class GenerationServiceClient:
    """Client for the Flask Generation Service"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        
    def load_model(self, model_name: str, num_gpus: Optional[int] = None, workers_per_gpu: Optional[int] = None) -> Dict[str, Any]:
        """Load a model into the service"""
        url = f"{self.base_url}/load_model"
        data = {"model_name": model_name}
        if num_gpus is not None:
            data["num_gpus"] = num_gpus
        if workers_per_gpu is not None:
            data["workers_per_gpu"] = workers_per_gpu
        response = requests.post(url, json=data)
        return response.json()

    def unload_model(self) -> Dict[str, Any]:
        """Unload the current model"""
        url = f"{self.base_url}/unload_model"
        response = requests.post(url)
        return response.json()

    def what_backend(self) -> Dict[str, Any]:
        """Return the current backend"""
        url = f"{self.base_url}/what_backend"
        response = requests.get(url)
        return response.json()

    def schedule_job(self, conversation: List[Dict], n_responses: int = 4, 
                    temperature: float = 1.0, max_tokens: int = 1000, gen_kwargs: Dict[str, Any] = {}) -> Dict[str, Any]:
        """Schedule a generation job"""
        url = f"{self.base_url}/schedule_job"
        data = {
            "conversation": conversation,
            "n_responses": n_responses,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "gen_kwargs": gen_kwargs
        }
        response = requests.post(url, json=data)
        return response.json()

    def build_tree(self, conversation: List[Dict], degree: int = 2, depth: int = 3,
                  temperature: float = 1.0, max_tokens: int = 1000, logprobs: int = 1) -> Dict[str, Any]:
        """Schedule a tree building job"""
        url = f"{self.base_url}/build_tree"
        data = {
            "conversation": conversation,
            "degree": degree,
            "depth": depth,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "logprobs": logprobs
        }
        response = requests.post(url, json=data)
        return response.json()

    def check_job(self, job_id: str) -> Dict[str, Any]:
        """Check the status of a job"""
        url = f"{self.base_url}/check_on_job"
        response = requests.get(url, params={"job_id": job_id})
        return response.json()

    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        url = f"{self.base_url}/status"
        response = requests.get(url)
        return response.json()

    def health_check(self) -> Dict[str, Any]:
        """Check if service is healthy"""
        url = f"{self.base_url}/health"
        response = requests.get(url)
        return response.json()

    def get_utilization(self, time_window: float = 30.0) -> Dict[str, Any]:
        """Get utilization statistics for the specified time window"""
        url = f"{self.base_url}/utilization"
        response = requests.get(url, params={"time_window": time_window})
        return response.json()

    def is_service_available(self) -> bool:
        """Check if the service is available and has a model loaded"""
        try:
            health = self.health_check()
            return health.get('service_loaded', False)
        except Exception:
            return False

    def wait_for_service_ready(self, timeout: float = 60) -> bool:
        """Wait for the service to be ready with a model loaded"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_service_available():
                return True
            time.sleep(1)
        return False

    def wait_for_job_completion(self, job_id: str, timeout: float = 900) -> Dict[str, Any]:
        """Wait for a job to complete and return the result"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            job_status = self.check_job(job_id)
            
            if job_status.get('status') == 'completed':
                return job_status
            elif job_status.get('status') == 'failed':
                return job_status
            
            time.sleep(1)
        
        # Timeout reached
        return {"status": "timeout", "error": f"Job timed out after {timeout} seconds"}

    def wait_for_tree_completion(self, job_id: str, timeout: float = 900) -> Dict[str, Any]:
        """Wait for a tree building job to complete and return the result"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            job_status = self.check_job(job_id)
            
            if job_status.get('status') == 'completed':
                return job_status
            elif job_status.get('status') == 'failed':
                return job_status
            
            time.sleep(1)
        
        # Timeout reached
        return {"status": "timeout", "error": f"Tree building job timed out after {timeout} seconds"}

if __name__ == "__main__":
    client = GenerationServiceClient()
    client.load_model("microsoft/phi-4", num_gpus=4, workers_per_gpu=1)
    client.wait_for_service_ready()

    # Test regular generation
    # resp = client.schedule_job([{"role": "user", "content": "Hello, how are you? Tell me a paragraph-long joke about UC Berkeley."}], n_responses=2)
    # print("Regular job:", resp)
    # job_status = client.wait_for_job_completion(resp["job_id"])
    # print("Regular job result:", job_status["status"])
    
    # Test tree building (async)
    conversation = [{"role": "user", "content": "Write a 8-word poem about AI and creativity. Just output 8 words that's it."}]
    tree_resp = client.build_tree(conversation, degree=2, depth=3)
    print("Tree job:", tree_resp)
    tree_status = client.wait_for_tree_completion(tree_resp["job_id"])
    print("Tree job result:", tree_status["status"])
    if "tree" in tree_status:
        print(f"Tree has {len(tree_status['tree'])} nodes")
        for n in tree_status["tree"]:
            print(f"[{n['subtree_id']}] {n['response_text_illustrated'].ljust(100)} {n['node_start_idx']} {n['node_end_idx']}")

    client.unload_model()