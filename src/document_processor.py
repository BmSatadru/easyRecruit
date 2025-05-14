import pypdfium2 as pdfium
import docx2txt
import os
import re
import json
from typing import Optional, Tuple, Dict, List
from .gemini_client import GeminiLLMClient

class DocumentProcessor:
    """Handles document processing and text extraction with structure preservation and industry context."""
    
    # Initialize LLM client as class attribute
    llm_client = GeminiLLMClient()
    
    # Common section headers in resumes and job descriptions
    RESUME_SECTIONS = {
        'education': ['education', 'academic background', 'qualifications'],
        'experience': ['experience', 'work experience', 'employment history', 'work history'],
        'skills': ['skills', 'technical skills', 'core competencies', 'expertise'],
        'projects': ['projects', 'key projects', 'project experience'],
        'certifications': ['certifications', 'certificates', 'professional certifications'],
        'summary': ['summary', 'professional summary', 'profile', 'about'],
    }

    JD_SECTIONS = {
        'requirements': ['requirements', 'qualifications', 'required skills'],
        'responsibilities': ['responsibilities', 'duties', 'job duties', 'role'],
        'company_info': ['company', 'about us', 'about company'],
        'benefits': ['benefits', 'perks', 'what we offer'],
    }

    # Industry-specific terminology patterns
    INDUSTRY_PATTERNS = {
        'tech': {
            'keywords': ['software', 'development', 'programming', 'IT', 'technology'],
            'skills': ['python', 'java', 'javascript', 'aws', 'cloud', 'devops'],
        },
        'finance': {
            'keywords': ['banking', 'finance', 'investment', 'trading', 'financial'],
            'skills': ['excel', 'financial modeling', 'analysis', 'risk management'],
        },
        'healthcare': {
            'keywords': ['healthcare', 'medical', 'clinical', 'patient'],
            'skills': ['patient care', 'medical records', 'healthcare systems'],
        },
    }

    def __init__(self):
        """Initialize document processor."""
        pass  # LLM client is now a class attribute

    @staticmethod
    def extract_from_pdf(file_path: str) -> Optional[Dict[str, str]]:
        """
        Extract text from a PDF file while preserving structure.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary with page numbers and their content, or None if extraction fails
        """
        try:
            pdf = pdfium.PdfDocument(file_path)
            text_content = {}
            
            for page_number in range(len(pdf)):
                page = pdf[page_number]
                text_page = page.get_textpage()
                text_content[f"page_{page_number + 1}"] = text_page.get_text_range()
                
            return text_content
            
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            return None

    @staticmethod
    def extract_from_docx(file_path: str) -> Optional[Dict[str, str]]:
        """
        Extract text from a DOCX file while preserving structure.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            Dictionary with document sections, or None if extraction fails
        """
        try:
            text = docx2txt.process(file_path)
            # Split by double newlines to preserve paragraph structure
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            return {'content': text, 'paragraphs': paragraphs}
        except Exception as e:
            print(f"Error extracting text from DOCX: {str(e)}")
            return None

    @staticmethod
    def extract_from_text(file_path: str) -> Optional[str]:
        """
        Extract text from a plain text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            File contents as string or None if reading fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading text file: {str(e)}")
            return None

    @staticmethod
    def identify_sections(text: str, section_patterns: Dict[str, List[str]]) -> Dict[str, str]:
        """
        Identify document sections based on common headers.
        
        Args:
            text: Document text
            section_patterns: Dictionary of section names and their possible headers
            
        Returns:
            Dictionary of identified sections and their content
        """
        sections = {}
        lines = text.split('\n')
        current_section = 'unknown'
        current_content = []

        for line in lines:
            line = line.strip().lower()
            if not line:
                continue

            # Check if line matches any section header
            matched_section = None
            for section, patterns in section_patterns.items():
                if any(pattern in line.lower() for pattern in patterns):
                    matched_section = section
                    break

            if matched_section:
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                current_section = matched_section
                current_content = []
            else:
                current_content.append(line)

        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content)

        return sections

    @classmethod
    def detect_industry_context_llm(cls, text: str, doc_type: str) -> Dict[str, any]:
        """
        Use LLM to detect industry context and extract relevant information through targeted questions.
        
        Args:
            text: Document text
            doc_type: Type of document ('resume' or 'jd')
            
        Returns:
            Dictionary containing detailed industry analysis
        """
        try:
            # First, identify sections to process them separately
            sections = cls.identify_sections(
                text,
                cls.RESUME_SECTIONS if doc_type.lower() == 'resume' else cls.JD_SECTIONS
            )

            # Define question sets based on document type and sections
            questions = {
                'industry_context': [
                    "What is the primary industry this role or candidate belongs to?",
                    "What are the related or secondary industries relevant to this role/experience?",
                    "How confident are you about the industry classification (0-100)?"
                ],
                'technical_analysis': [
                    "What technical terms, tools, or technologies are mentioned?",
                    "What industry-specific terminology or jargon is used?",
                    "What methodologies or approaches are mentioned?"
                ],
                'role_analysis': [
                    "What is the seniority level of this role/candidate?",
                    "What is the main category or function of this role?",
                    "What are the key responsibilities or achievements mentioned?",
                    "What areas of expertise are required or demonstrated?"
                ],
                'qualifications': [
                    "What educational qualifications are mentioned?",
                    "What certifications or professional qualifications are listed?",
                    "What specific domain knowledge or expertise is demonstrated?"
                ]
            }

            # Initialize results dictionary
            results = {}
            
            # Process each question category using relevant sections
            for category, category_questions in questions.items():
                print(f"\nProcessing {category}...")
                category_results = []
                
                # Select relevant sections for each category
                relevant_sections = []
                if category == 'industry_context':
                    relevant_sections = [sections.get('summary', ''), sections.get('experience', '')]
                elif category == 'technical_analysis':
                    relevant_sections = [sections.get('skills', ''), sections.get('experience', '')]
                elif category == 'role_analysis':
                    relevant_sections = [sections.get('experience', ''), sections.get('summary', '')]
                elif category == 'qualifications':
                    relevant_sections = [sections.get('education', ''), sections.get('certifications', ''), sections.get('skills', '')]
                
                # Combine relevant sections
                section_text = '\n\n'.join(filter(None, relevant_sections))
                if not section_text:
                    section_text = text  # Fallback to full text if no relevant sections found
                
                for question in category_questions:
                    prompt = f"""
Analyze this {'job description' if doc_type == 'jd' else 'resume'} excerpt and answer the following question:
{question}

Please provide a clear, specific answer based only on the information present in the text.
If the information is not available, respond with "Not specified".

Text:
{section_text}
"""
                    response = cls.llm_client.model.generate_content(prompt)
                    category_results.append(response.text.strip())
                
                # Store results for this category
                results[category] = category_results

            # Process results into structured format
            industry_context = {
                "primary_industry": results['industry_context'][0],
                "secondary_industries": [ind.strip() for ind in results['industry_context'][1].split(',') if ind.strip() and 'not specified' not in ind.lower()],
                "industry_confidence": int(re.search(r'\d+', results['industry_context'][2] or '0').group() or 0),
                "detected_terms": {
                    "technical_terms": [term.strip() for term in results['technical_analysis'][0].split(',') if term.strip() and 'not specified' not in term.lower()],
                    "industry_specific_terms": [term.strip() for term in results['technical_analysis'][1].split(',') if term.strip() and 'not specified' not in term.lower()],
                    "methodologies": [term.strip() for term in results['technical_analysis'][2].split(',') if term.strip() and 'not specified' not in term.lower()]
                },
                "role_analysis": {
                    "seniority_level": results['role_analysis'][0],
                    "role_category": results['role_analysis'][1],
                    "key_responsibilities": [resp.strip() for resp in results['role_analysis'][2].split(',') if resp.strip() and 'not specified' not in resp.lower()],
                    "required_expertise": [exp.strip() for exp in results['role_analysis'][3].split(',') if exp.strip() and 'not specified' not in exp.lower()]
                },
                "qualifications": {
                    "education": [edu.strip() for edu in results['qualifications'][0].split(',') if edu.strip() and 'not specified' not in edu.lower()],
                    "certifications": [cert.strip() for cert in results['qualifications'][1].split(',') if cert.strip() and 'not specified' not in cert.lower()],
                    "domain_knowledge": [domain.strip() for domain in results['qualifications'][2].split(',') if domain.strip() and 'not specified' not in domain.lower()]
                }
            }

            # Generate summary based on collected information
            summary_prompt = f"""
Based on the following analysis, provide a brief summary of the industry context and role:

Primary Industry: {industry_context['primary_industry']}
Secondary Industries: {', '.join(industry_context['secondary_industries'])}
Role Level: {industry_context['role_analysis']['seniority_level']}
Role Category: {industry_context['role_analysis']['role_category']}

Please provide a concise 2-3 sentence summary.
"""
            summary_response = cls.llm_client.model.generate_content(summary_prompt)
            industry_context["analysis_summary"] = summary_response.text.strip()

            return industry_context

        except Exception as e:
            print(f"Error in LLM industry context detection: {str(e)}")
            # Fallback to basic context
            return {
                "primary_industry": "unknown",
                "secondary_industries": [],
                "industry_confidence": 0,
                "detected_terms": {
                    "technical_terms": [],
                    "industry_specific_terms": [],
                    "methodologies": []
                },
                "role_analysis": {
                    "seniority_level": "unknown",
                    "role_category": "unknown",
                    "key_responsibilities": [],
                    "required_expertise": []
                },
                "qualifications": {
                    "education": [],
                    "certifications": [],
                    "domain_knowledge": []
                },
                "analysis_summary": "Failed to analyze industry context"
            }

    @classmethod
    def process_document(cls, file_path: str, file_type: str, doc_type: str = 'unknown') -> Dict[str, any]:
        """
        Process a document with structure preservation and industry context.
        
        Args:
            file_path: Path to the document
            file_type: Type of the document ('pdf', 'docx', or 'txt')
            doc_type: Type of document ('resume' or 'jd')
            
        Returns:
            Dictionary containing processed document information
        """
        if not os.path.exists(file_path):
            return {'error': f"File not found: {file_path}"}

        # Extract raw text based on file type
        if file_type.lower() == 'pdf':
            raw_content = cls.extract_from_pdf(file_path)
        elif file_type.lower() in ['docx', 'doc']:
            raw_content = cls.extract_from_docx(file_path)
        elif file_type.lower() == 'txt':
            raw_content = {'content': cls.extract_from_text(file_path)}
        else:
            return {'error': f"Unsupported file type: {file_type}"}

        if not raw_content:
            return {'error': 'Failed to extract text from document'}

        # Get the main text content
        text_content = raw_content.get('content', 
                                     '\n'.join(raw_content.values()) if isinstance(raw_content, dict) else '')

        # Clean the text
        cleaned_text = cls.clean_text(text_content)

        # Identify document sections
        sections = cls.identify_sections(
            cleaned_text,
            cls.RESUME_SECTIONS if doc_type.lower() == 'resume' else cls.JD_SECTIONS
        )

        # Use LLM for industry context detection
        industry_context = cls.detect_industry_context_llm(cleaned_text, doc_type)

        # Debug output for verification
        print("\nDocument Processing Debug:")
        print(f"File: {file_path}")
        print(f"Type: {file_type}")
        print("\nDetected Sections:")
        for section, content in sections.items():
            print(f"\n{section.upper()}:")
            print(f"Length: {len(content)} chars")
            print(f"Preview: {content[:100]}...")
        
        print("\nIndustry Analysis:")
        print(f"Primary: {industry_context.get('primary_industry')}")
        print(f"Confidence: {industry_context.get('industry_confidence')}%")
        print("Role Analysis:", json.dumps(industry_context.get('role_analysis', {}), indent=2))

        return {
            'raw_content': raw_content,
            'cleaned_text': cleaned_text,
            'sections': sections,
            'industry_context': industry_context,
            'document_type': doc_type,
            'file_type': file_type
        }

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text while preserving important structure.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Preserve section headers by adding newlines around them
        text = re.sub(r'([A-Z][A-Z\s]+:)', r'\n\1\n', text)
        
        # Replace multiple spaces with single space
        text = ' '.join(text.split())
        
        # Normalize newlines while preserving paragraph structure
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove unnecessary whitespace while preserving structure
        text = '\n'.join(line.strip() for line in text.splitlines())
        
        return text 