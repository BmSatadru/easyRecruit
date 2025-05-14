from abc import ABC, abstractmethod

class LLMClient(ABC):
    @abstractmethod
    def format_content(self, text: str) -> dict:
        """Extract structured data from text content"""
        pass

    @abstractmethod
    def eligibility_check(self, job_description: str, resume: str) -> dict:
        """Check eligibility of a candidate against a job description"""
        pass

    @abstractmethod
    def analyze_jd_resume(self, job_description: str, resume: str, applicant_eligibility: str) -> dict:
        """Analyze job description and resume to extract detailed matching information"""
        pass 