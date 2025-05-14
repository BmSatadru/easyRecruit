from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import pandas as pd
from .gemini_client import GeminiLLMClient
from .document_processor import DocumentProcessor

class Applicant(BaseModel):
    """Information about a job applicant based on the job description and applicant resume."""

    name: Optional[str] = Field(default=None, description="Name of the job applicant.")
    email: Optional[str] = Field(default=None, description="Email address of the job applicant.")
    mobile: Optional[str] = Field(default=None, description="Mobile number of the job applicant.")
    skills_matching: Optional[str] = Field(
        default=None,
        description="required skills from the Job Description which the applicant possesses."
    )
    skills_missing: Optional[str] = Field(
        default=None,
        description="required skills from the Job Description which the applicant lacks."
    )
    eligible_for_role: Optional[str] = Field(
        default=None,
        description="Applicant's eligibility for the role based on the APPLICANT ELIGIBILITY."
    )
    education_background: Optional[str] = Field(
        default=None,
        description="Educational background of the job applicant."
    )
    total_work_experience: Optional[int] = Field(
        default=None,
        description="Total years of work experience of the job applicant."
    )
    relevant_work_experience: Optional[int] = Field(
        default=None,
        description="Years of relevant work experience of the job applicant."
    )
    skills: Optional[List[str]] = Field(
        default=None,
        description="List of key skills possessed by the job applicant."
    )
    certifications: Optional[List[str]] = Field(
        default=None,
        description="List of relevant certifications held by the job applicant."
    )
    languages: Optional[List[str]] = Field(
        default=None,
        description="Languages known by the job applicant."
    )
    additional_information: Optional[str] = Field(
        default=None,
        description="Any additional relevant information about the job applicant."
    )
    industry_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Industry context and domain-specific information."
    )

def extract_text(file_path: str, file_type: str, doc_type: str) -> Dict[str, Any]:
    """
    Extract and process text from a document file.
    
    Args:
        file_path: Path to the document file
        file_type: Type of the document ('pdf', 'docx', or 'txt')
        doc_type: Type of document ('resume' or 'jd')
        
    Returns:
        Dictionary containing processed document information
    """
    return DocumentProcessor.process_document(file_path, file_type, doc_type)

def format_content(document_info: Dict[str, Any]) -> dict:
    """
    Format the extracted document content using the Gemini LLM.
    
    Args:
        document_info: Processed document information
        
    Returns:
        Formatted content as a dictionary
    """
    if 'error' in document_info:
        return document_info

    # Include industry context in the formatting
    llm_client = GeminiLLMClient()
    return llm_client.format_content({
        'text': document_info['cleaned_text'],
        'sections': document_info['sections'],
        'industry_context': document_info['industry_context']
    })

def eligibility_check(jd_info: Dict[str, Any], resume_info: Dict[str, Any]) -> dict:
    """
    Check eligibility of a candidate against a job description.
    
    Args:
        jd_info: Processed job description information
        resume_info: Processed resume information
        
    Returns:
        Eligibility assessment as a dictionary
    """
    if 'error' in jd_info or 'error' in resume_info:
        return {'error': 'Invalid document information'}

    llm_client = GeminiLLMClient()
    return llm_client.eligibility_check({
        'jd_text': jd_info['cleaned_text'],
        'jd_sections': jd_info['sections'],
        'jd_industry': jd_info['industry_context'],
        'resume_text': resume_info['cleaned_text'],
        'resume_sections': resume_info['sections'],
        'resume_industry': resume_info['industry_context']
    })

def analyze_jd_resume(jd_info: Dict[str, Any], resume_info: Dict[str, Any], eligibility_result: Dict[str, Any]) -> dict:
    """
    Analyze job description and resume to extract detailed matching information.
    
    Args:
        jd_info: Processed job description information
        resume_info: Processed resume information
        eligibility_result: Eligibility assessment information
        
    Returns:
        Detailed analysis as a dictionary
    """
    if 'error' in jd_info or 'error' in resume_info:
        return {'error': 'Invalid document information'}

    llm_client = GeminiLLMClient()
    return llm_client.analyze_jd_resume({
        'jd_text': jd_info['cleaned_text'],
        'jd_sections': jd_info['sections'],
        'jd_industry': jd_info['industry_context'],
        'resume_text': resume_info['cleaned_text'],
        'resume_sections': resume_info['sections'],
        'resume_industry': resume_info['industry_context'],
        'eligibility': eligibility_result
    })

def file_create(uploaded_file, file_name: str) -> tuple[str, str]:
    """
    Create a file from uploaded content.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        file_name: Base name for the file
        
    Returns:
        Tuple of (file_path, file_type)
    """
    file_type = uploaded_file.name.split('.')[-1].lower()
    file_path = f"{file_name}.{file_type}"
    
    with open(file_path, 'wb') as file:
        file.write(uploaded_file.getvalue())
    
    return file_path, file_type

def convert_to_dataframe(applicants_data: List[dict]) -> pd.DataFrame:
    """
    Convert applicants data to a pandas DataFrame.
    
    Args:
        applicants_data: List of applicant dictionaries
        
    Returns:
        Pandas DataFrame containing applicant information
    """
    data_dict = {
        'Name': [a.name for a in applicants_data],
        'Email': [a.email for a in applicants_data],
        'Mobile': [a.mobile for a in applicants_data],
        'Skills Matching': [a.skills_matching for a in applicants_data],
        'Skills Not Matching': [a.skills_missing for a in applicants_data],
        'Eligible for role': [a.eligible_for_role for a in applicants_data],
        'Education Background': [a.education_background for a in applicants_data],
        'Total Years of Experience': [a.total_work_experience for a in applicants_data],
        'Relevant Years of Experience': [a.relevant_work_experience for a in applicants_data],
        'Skills': [a.skills for a in applicants_data],
        'Certifications': [a.certifications for a in applicants_data],
        'Languages': [a.languages for a in applicants_data],
        'Additional Information': [a.additional_information for a in applicants_data],
        'Industry Context': [a.industry_context for a in applicants_data]
    }
    return pd.DataFrame(data_dict)
