import os
import pytest
from document_processor import DocumentProcessor

class TestDocumentProcessor:
    """Test cases for DocumentProcessor class"""

    @pytest.fixture
    def sample_text(self):
        return """
PROFESSIONAL SUMMARY:
Experienced software engineer with 5 years in Python development.

SKILLS:
- Python, Java, JavaScript
- AWS, Cloud Computing
- DevOps practices

WORK EXPERIENCE:
Software Engineer at Tech Corp
- Developed cloud applications
- Led team of 3 developers

EDUCATION:
Bachelor's in Computer Science
        """

    def test_clean_text(self, sample_text):
        """Test text cleaning functionality"""
        cleaned = DocumentProcessor.clean_text(sample_text)
        
        # Verify structure preservation
        assert "PROFESSIONAL SUMMARY:" in cleaned
        assert "SKILLS:" in cleaned
        assert "WORK EXPERIENCE:" in cleaned
        assert "EDUCATION:" in cleaned
        
        # Verify whitespace handling
        assert "\n\n" in cleaned  # Should preserve paragraph breaks
        assert "  " not in cleaned  # Should not have multiple spaces
        
        print("\n=== Cleaned Text ===")
        print(cleaned)
        print("===================")

    def test_identify_sections(self, sample_text):
        """Test section identification"""
        sections = DocumentProcessor.identify_sections(
            sample_text, 
            DocumentProcessor.RESUME_SECTIONS
        )
        
        # Verify section detection
        assert 'summary' in sections
        assert 'skills' in sections
        assert 'experience' in sections
        assert 'education' in sections
        
        print("\n=== Identified Sections ===")
        for section, content in sections.items():
            print(f"\n{section.upper()}:")
            print(content)
        print("=========================")

    def test_detect_industry_context(self, sample_text):
        """Test industry context detection"""
        context = DocumentProcessor.detect_industry_context(sample_text)
        
        # Verify industry detection
        assert 'primary_industry' in context
        assert 'industry_scores' in context
        assert 'detected_terms' in context
        
        # Verify tech industry detection (given the sample text)
        assert context['primary_industry'] == 'tech'
        
        print("\n=== Industry Context ===")
        print(f"Primary Industry: {context['primary_industry']}")
        print("\nIndustry Scores:")
        for industry, score in context['industry_scores'].items():
            print(f"{industry}: {score}")
        print("\nDetected Terms:")
        for industry, terms in context['detected_terms'].items():
            if terms['keywords'] or terms['skills']:
                print(f"\n{industry.upper()}:")
                if terms['keywords']:
                    print("Keywords:", terms['keywords'])
                if terms['skills']:
                    print("Skills:", terms['skills'])
        print("=====================")

    def test_process_document_with_sample(self, tmp_path):
        """Test complete document processing with a sample file"""
        # Create a sample file
        sample_path = tmp_path / "test_resume.txt"
        with open(sample_path, 'w') as f:
            f.write("""
PROFESSIONAL SUMMARY:
Senior Software Engineer with expertise in cloud computing and AI development.

TECHNICAL SKILLS:
- Programming: Python, Java, JavaScript, C++
- Cloud: AWS, Azure, Google Cloud
- AI/ML: TensorFlow, PyTorch, Scikit-learn
- DevOps: Docker, Kubernetes, Jenkins

WORK EXPERIENCE:
AI Engineer at Tech Solutions Inc. (2020-Present)
- Developed machine learning models for production
- Led team of 5 ML engineers
- Implemented CI/CD pipelines

Software Engineer at Cloud Systems (2018-2020)
- Built cloud-native applications
- Optimized system performance

EDUCATION:
Master's in Computer Science
Bachelor's in Software Engineering
            """)

        # Process the document
        result = DocumentProcessor.process_document(str(sample_path), 'txt', 'resume')
        
        # Verify complete processing
        assert 'error' not in result
        assert 'cleaned_text' in result
        assert 'sections' in result
        assert 'industry_context' in result
        
        # Verify sections
        assert 'skills' in result['sections']
        assert 'experience' in result['sections']
        assert 'education' in result['sections']
        
        # Verify industry detection
        assert result['industry_context']['primary_industry'] == 'tech'
        
        print("\n=== Complete Document Processing Results ===")
        print("\nCleaned Text Preview:")
        print(result['cleaned_text'][:200] + "...")
        
        print("\nIdentified Sections:")
        for section, content in result['sections'].items():
            print(f"\n{section.upper()}:")
            print(content[:100] + "..." if len(content) > 100 else content)
        
        print("\nIndustry Context:")
        print(f"Primary Industry: {result['industry_context']['primary_industry']}")
        print(f"Industry Scores: {result['industry_context']['industry_scores']}")
        print("=======================================")

def main():
    """Run tests with detailed output"""
    print("Running DocumentProcessor verification tests...")
    
    # Create test instance
    test = TestDocumentProcessor()
    sample_text = test.sample_text()
    
    # Run individual tests
    print("\n1. Testing text cleaning...")
    test.test_clean_text(sample_text)
    
    print("\n2. Testing section identification...")
    test.test_identify_sections(sample_text)
    
    print("\n3. Testing industry context detection...")
    test.test_detect_industry_context(sample_text)
    
    print("\n4. Testing complete document processing...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        test.test_process_document_with_sample(tmp_dir)
    
    print("\nAll verification tests completed!")

if __name__ == "__main__":
    main() 