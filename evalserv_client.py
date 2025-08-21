import requests
import json
import time
from typing import Dict, Any, List, Optional

class EvaluationServiceClient:
    """Client for the Flask Evaluation Service"""
    
    def __init__(self, base_url: str = "http://localhost:5001"):
        self.base_url = base_url

    def shutdown_service(self) -> Dict[str, Any]:
        """Shutdown the evaluation service"""
        url = f"{self.base_url}/shutdown_service"
        response = requests.post(url)
        return response.json()

    def schedule_evaluation(self, conversation: List[Dict], task_name: str, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Schedule an evaluation job"""
        url = f"{self.base_url}/schedule_evaluation"
        data = {
            "conversation": conversation,
            "task_name": task_name,
            "sample": sample,
        }
        response = requests.post(url, json=data)
        return response.json()

    def check_job(self, job_id: str) -> Dict[str, Any]:
        """Check the status of an evaluation job"""
        url = f"{self.base_url}/check_evaluation_job"
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

    def is_service_available(self) -> bool:
        """Check if the service is available and has tasks loaded"""
        try:
            health = self.health_check()
            return health.get('service_initialized', False) and health.get('tasks_loaded', False)
        except Exception:
            return False

    def wait_for_service_ready(self, timeout: float = 60) -> bool:
        """Wait for the service to be ready with tasks loaded"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_service_available():
                return True
            time.sleep(1)
        return False

    def wait_for_job_completion(self, job_id: str, timeout: float = 120) -> Dict[str, Any]:
        """Wait for an evaluation job to complete and return the result"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            job_status = self.check_job(job_id)
            
            if job_status.get('status') == 'completed':
                return job_status
            elif job_status.get('status') == 'error':
                return job_status
            
            time.sleep(0.5)  # Check more frequently for evaluations
        
        # Timeout reached
        return {"status": "timeout", "error": f"Job timed out after {timeout} seconds"} 