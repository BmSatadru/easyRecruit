import os
import pytest
from analyzer import (
    extract_text,
    format_content,
    eligibility_check,
    analyze_jd_resume,
    file_create
)

class TestAnalyzer:
    """Test cases for analyzer functionality"""

    @pytest.fixture
    def sample_jd(self, tmp_path):
        """Create a sample job description file"""
        jd_path = tmp_path / "test_jd.txt"
        with open(jd_path, 'w') as f:
            f.write("""
SENIOR SOFTWARE ENGINEER

ABOUT US:
Leading tech company specializing in cloud solutions.

REQUIREMENTS:
- 5+ years of software development experience
- Expert in Python, Java, and cloud technologies
- Experience with AWS and microservices
- Strong background in system design
- Master's degree in Computer Science preferred

RESPONSIBILITIES:
- Lead development of cloud-native applications
- Mentor junior developers
- Design and implement scalable solutions
- Collaborate with cross-functional teams

BENEFITS:
- Competitive salary
- Health insurance
- Remote work options
            """)
        return jd_path

    @pytest.fixture
    def sample_resume(self, tmp_path):
        """Create a sample resume file"""
        resume_path = tmp_path / "test_resume.txt"
        with open(resume_path, 'w') as f:
            f.write("""
PROFESSIONAL SUMMARY:
Senior Software Engineer with 6 years of experience in cloud computing and distributed systems.

TECHNICAL SKILLS:
- Languages: Python (8 years), Java (6 years), JavaScript
- Cloud: AWS (5 years), Azure, Google Cloud
- Tools: Docker, Kubernetes, Jenkins
- Databases: PostgreSQL, MongoDB

WORK EXPERIENCE:
Lead Engineer at Cloud Tech (2020-Present)
- Led team of 5 developers for cloud-native applications
- Implemented microservices architecture
- Reduced system latency by 40%

Senior Developer at Tech Solutions (2018-2020)
- Developed scalable backend services
- Mentored junior developers

EDUCATION:
Master's in Computer Science
Bachelor's in Software Engineering

CONTACT:
Email: test@example.com
Phone: (123) 456-7890
            """)
        return resume_path

    def test_document_extraction(self, sample_jd, sample_resume):
        """Test document text extraction"""
        print("\n=== Testing Document Extraction ===")
        
        # Test JD extraction
        jd_info = extract_text(str(sample_jd), 'txt', 'jd')
        assert 'error' not in jd_info
        assert 'cleaned_text' in jd_info
        assert 'sections' in jd_info
        print("\nJob Description Sections:")
        for section in jd_info['sections']:
            print(f"- {section}")
        
        # Test Resume extraction
        resume_info = extract_text(str(sample_resume), 'txt', 'resume')
        assert 'error' not in resume_info
        assert 'cleaned_text' in resume_info
        assert 'sections' in resume_info
        print("\nResume Sections:")
        for section in resume_info['sections']:
            print(f"- {section}")

        return jd_info, resume_info

    def test_content_formatting(self, sample_jd, sample_resume):
        """Test content formatting"""
        print("\n=== Testing Content Formatting ===")
        
        # Get document info
        jd_info = extract_text(str(sample_jd), 'txt', 'jd')
        resume_info = extract_text(str(sample_resume), 'txt', 'resume')
        
        # Test formatting
        jd_content = format_content(jd_info)
        resume_content = format_content(resume_info)
        
        assert 'error' not in jd_content
        assert 'error' not in resume_content
        
        print("\nJob Description Content Keys:")
        print(jd_content.keys())
        print("\nResume Content Keys:")
        print(resume_content.keys())
        
        return jd_content, resume_content

    def test_eligibility_check(self, sample_jd, sample_resume):
        """Test eligibility checking"""
        print("\n=== Testing Eligibility Check ===")
        
        # Get document info
        jd_info = extract_text(str(sample_jd), 'txt', 'jd')
        resume_info = extract_text(str(sample_resume), 'txt', 'resume')
        
        # Test eligibility check
        result = eligibility_check(jd_info, resume_info)
        
        assert 'error' not in result
        assert 'result' in result
        
        print("\nEligibility Result:")
        print(result['result'])
        
        return result

    def test_full_analysis(self, sample_jd, sample_resume):
        """Test complete JD-Resume analysis"""
        print("\n=== Testing Complete Analysis ===")
        
        # Get document info
        jd_info = extract_text(str(sample_jd), 'txt', 'jd')
        resume_info = extract_text(str(sample_resume), 'txt', 'resume')
        
        # Get eligibility
        eligibility_result = eligibility_check(jd_info, resume_info)
        
        # Test analysis
        result = analyze_jd_resume(jd_info, resume_info, eligibility_result)
        
        assert result is not None
        
        print("\nAnalysis Result Keys:")
        if isinstance(result, dict):
            print(result.keys())
        else:
            print("Result type:", type(result))
        
        return result

def main():
    """Run verification tests"""
    print("Running Analyzer verification tests...")
    
    # Create test instance
    test = TestAnalyzer()
    
    # Create temporary directory for test files
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create sample files
        tmp_path = pytest.Path(tmp_dir)
        sample_jd = test.sample_jd(tmp_path)
        sample_resume = test.sample_resume(tmp_path)
        
        # Run tests
        print("\n1. Testing document extraction...")
        jd_info, resume_info = test.test_document_extraction(sample_jd, sample_resume)
        
        print("\n2. Testing content formatting...")
        jd_content, resume_content = test.test_content_formatting(sample_jd, sample_resume)
        
        print("\n3. Testing eligibility check...")
        eligibility_result = test.test_eligibility_check(sample_jd, sample_resume)
        
        print("\n4. Testing complete analysis...")
        analysis_result = test.test_full_analysis(sample_jd, sample_resume)
    
    print("\nAll verification tests completed!")

if __name__ == "__main__":
    main() 