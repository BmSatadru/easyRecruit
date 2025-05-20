import logging
import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import spacy
from .gemini_client import GeminiLLMClient
import json
from .db.mongodb_client import MongoDBClient

logger = logging.getLogger(__name__)

class JDAnalyzer:
    # Common section headers in job descriptions
    SECTION_HEADERS = {
        'responsibilities': [
            'responsibilities', 'duties', 'what you\'ll do', 'role overview',
            'job description', 'position description', 'key responsibilities'
        ],
        'qualifications': [
            'qualifications', 'requirements', 'what you\'ll need', 
            'required skills', 'minimum requirements', 'skills required'
        ],
        'education': [
            'education', 'educational requirements', 'academic requirements',
            'qualifications', 'degree requirements'
        ],
        'experience': [
            'experience', 'work experience', 'professional experience',
            'required experience', 'experience requirements'
        ],
        'benefits': [
            'benefits', 'perks', 'what we offer', 'compensation',
            'benefits & perks', 'why join us'
        ],
        'company': [
            'about us', 'company overview', 'who we are', 'about the company',
            'our company', 'company description'
        ]
    }
    
    def __init__(self):
        """Initialize the JD Analyzer with necessary models and clients."""
        self.llm_client = GeminiLLMClient()
        self.db_client = MongoDBClient()
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Successfully loaded spaCy model")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            self.nlp = None
            
        # Verification counters
        self.stats = defaultdict(int)
    
    def analyze_structure(self, text: str) -> Dict[str, any]:
        """Analyze the structure of the job description text.
        
        Args:
            text: The cleaned job description text
            
        Returns:
            Dict containing identified sections and structural analysis
        """
        logger.info("Starting structure analysis")
        self.stats.clear()
        
        # Split text into potential sections
        sections = self._identify_sections(text)
        
        # Analyze each section's content
        analyzed_sections = {}
        for section_type, content in sections.items():
            # Verify section content
            word_count = len(content.split())
            self.stats[f'section_words_{section_type}'] = word_count
            
            if word_count < 10:  # Arbitrary minimum for a valid section
                logger.warning(f"Section {section_type} seems too short ({word_count} words)")
                continue
                
            analyzed_sections[section_type] = {
                'content': content,
                'word_count': word_count,
                'key_phrases': self._extract_key_phrases(content)
            }
        
        # Generate verification report
        verification = self._generate_structure_verification()
        
        return {
            'sections': analyzed_sections,
            'verification': verification,
            'stats': dict(self.stats)
        }
    
    def _identify_sections(self, text: str) -> Dict[str, str]:
        """Identify and extract sections from the text."""
        sections = {}
        current_section = 'unclassified'
        current_content = []
        
        # Split text into lines
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line is a section header
            section_type = self._detect_section_type(line)
            if section_type:
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                    self.stats['sections_identified'] += 1
                
                current_section = section_type
                current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content)
            
        return sections
    
    def _detect_section_type(self, line: str) -> Optional[str]:
        """Detect if a line is a section header and return its type."""
        line_lower = line.lower()
        
        # Check if line looks like a header (e.g., all caps, ends with :, etc.)
        if not (line.isupper() or line.endswith(':') or len(line.split()) <= 5):
            return None
        
        for section_type, patterns in self.SECTION_HEADERS.items():
            if any(pattern in line_lower for pattern in patterns):
                self.stats[f'section_header_{section_type}'] = 1
                return section_type
        
        return None
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from section content using spaCy."""
        if not self.nlp:
            return []
            
        doc = self.nlp(text)
        phrases = []
        
        # Extract noun phrases
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 4:  # Limit phrase length
                phrases.append(chunk.text)
                
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'SKILL', 'GPE']:
                phrases.append(ent.text)
        
        return list(set(phrases))  # Remove duplicates
    
    def _generate_structure_verification(self) -> List[str]:
        """Generate verification messages for structure analysis."""
        messages = [
            f"✓ Identified {self.stats['sections_identified']} sections",
        ]
        
        # Add section-specific messages
        for section_type in self.SECTION_HEADERS.keys():
            if self.stats.get(f'section_header_{section_type}'):
                messages.append(f"✓ Found {section_type} section")
                words = self.stats.get(f'section_words_{section_type}', 0)
                messages.append(f"  - {words} words in section")
        
        return messages
    
    def process_content(self, text: str, sections: Dict[str, any]) -> Dict[str, any]:
        """Process the content using Gemini LLM for detailed extraction."""
        logger.info("\n" + "="*50)
        logger.info("JD_ANALYZER PROCESSING STARTED")
        logger.info("="*50)
        
        try:
            # Log the input sections
            logger.info("\nInput Sections:")
            for section_name, content in sections.items():
                logger.info(f"\n{section_name.upper()}:")
                logger.info("-"*30)
                logger.info(content[:200] + "..." if len(content) > 200 else content)
            
            # Prepare structured prompt for the LLM
            prompt = self._create_extraction_prompt(text, sections)
            
            # Log the created prompt
            logger.info("\nCreated Prompt:")
            logger.info("-"*30)
            logger.info(prompt)
            
            # Get LLM response using _handle_api_call
            logger.info("\nSending to GeminiClient...")
            response = self.llm_client._handle_api_call(prompt)
            
            # Log the raw response
            logger.info("\nRaw LLM Response:")
            logger.info("-"*30)
            logger.info(response)
            
            # Parse and validate response
            extracted_data = self._parse_llm_response(response)
            
            # Log the parsed data
            logger.info("\nParsed and Validated Data:")
            logger.info("-"*30)
            logger.info(json.dumps(extracted_data, indent=2))
            
            # Verify extraction quality
            verification = self._verify_extraction(extracted_data)
            
            logger.info("\nVerification Results:")
            logger.info("-"*30)
            for msg in verification:
                logger.info(msg)
            
            logger.info("\n" + "="*50)
            logger.info("JD_ANALYZER PROCESSING COMPLETED")
            logger.info("="*50 + "\n")
            
            return {
                'extracted_data': extracted_data,
                'verification': verification
            }
            
        except Exception as e:
            logger.error(f"\nError in content processing: {e}")
            return {
                'error': str(e),
                'verification': ["❌ Content processing failed"]
            }
    
    def _create_extraction_prompt(self, text: str, sections: Dict[str, any]) -> str:
        """Create a structured prompt for the LLM."""
        prompt = [
            "Extract key information from this job description and return it in valid JSON format.",
            "Use double quotes for JSON keys and string values.",
            "Expected structure:",
            "{",
            '  "core_metadata": {',
            '    "company": "string",',
            '    "job_title": "string",',
            '    "location": "string",',
            '    "employment_type": "string"',
            '  },',
            '  "compensation": {',
            '    "salary_range": "string",',
            '    "benefits": ["string"],',
            '    "perks": ["string"]',
            '  },',
            '  "requirements": {',
            '    "skills": {',
            '      "technical": ["string"],',
            '      "soft": ["string"]',
            '    },',
            '    "experience": "string",',
            '    "education": "string"',
            '  },',
            '  "job_details": {',
            '    "responsibilities": ["string"],',
            '    "team_structure": "string",',
            '    "growth_opportunities": "string"',
            '  },',
            '  "additional_info": {',
            '    "industry": "string",',
            '    "seniority_level": "string",',
            '    "travel_requirements": "string",',
            '    "remote_policy": "string"',
            '  }',
            "}",
            "\nJob Description:\n",
            text,
            "\nImportant: Ensure the response is a valid JSON object with all the fields shown above. Use null for missing values."
        ]
        return "\n".join(prompt)
    
    def _parse_llm_response(self, response: str) -> Dict:
        """Parse and validate the LLM response."""
        try:
            if not response or not response.strip():
                logger.error("Empty response from LLM")
                return self._get_default_response("Empty response from LLM")

            # Clean up the response
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()
            
            # Attempt to parse JSON
            try:
                data = json.loads(response)
            except json.JSONDecodeError as je:
                logger.error(f"JSON parse error: {je}")
                return self._get_default_response(f"JSON parse error: {je}")

            # Validate and ensure required structure
            return self._validate_and_structure_data(data)
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return self._get_default_response(str(e))

    def _get_default_response(self, error_msg: str) -> Dict:
        """Get default structured response with error message."""
        return {
            'core_metadata': {
                "company": "Not specified",
                "job_title": "Not specified",
                "location": "Not specified",
                "employment_type": "Not specified"
            },
            'requirements': {
                "skills": {
                    "technical": [],
                    "soft": []
                },
                "experience": "Not specified",
                "education": "Not specified"
            },
            'error': error_msg
        }

    def _validate_and_structure_data(self, data: Dict) -> Dict:
        """Validate and ensure proper data structure."""
        default_structure = self._get_default_response("")
        
        # Merge with default structure to ensure all fields exist
        for key in default_structure:
            if key not in data:
                data[key] = default_structure[key]
            elif isinstance(default_structure[key], dict):
                for subkey in default_structure[key]:
                    if subkey not in data[key]:
                        data[key][subkey] = default_structure[key][subkey]
        
        return data
    
    def _verify_extraction(self, data: Dict) -> List[str]:
        """Verify the quality of extracted information."""
        verification = []
        
        # Check core metadata
        if 'core_metadata' in data:
            meta = data['core_metadata']
            for field in ['company', 'job_title', 'location']:
                if field in meta and meta[field]:
                    verification.append(f"✓ Found {field}: {meta[field]}")
                else:
                    verification.append(f"⚠️ Missing {field}")
        
        # Check requirements
        if 'requirements' in data:
            reqs = data['requirements']
            if 'skills' in reqs:
                skills = reqs['skills']
                verification.append(f"✓ Found {len(skills.get('technical', []))} technical skills")
                verification.append(f"✓ Found {len(skills.get('soft', []))} soft skills")
        
        return verification

    def process_and_store_jd(self, doc_id: str, text: str, file_metadata: Dict = None) -> Dict[str, any]:
        """Process and store job description data.
        
        Args:
            doc_id: Unique identifier for the job description
            text: The job description text
            file_metadata: Optional metadata about the original file
            
        Returns:
            Dict containing processing results and storage status
        """
        try:
            # Analyze structure
            structure_results = self.analyze_structure(text)
            
            # Process content
            content_results = self.process_content(text, structure_results['sections'])
            
            # Verify content results
            if 'extracted_data' not in content_results:
                content_results['extracted_data'] = {}
            
            if 'core_metadata' not in content_results['extracted_data']:
                content_results['extracted_data']['core_metadata'] = {
                    "company": "Not specified",
                    "job_title": "Not specified",
                    "location": "Not specified",
                    "employment_type": "Not specified"
                }
            
            # Prepare data for storage
            jd_data = {
                **content_results['extracted_data'],
                'stats': structure_results['stats']
            }
            
            # Add file metadata if provided
            if file_metadata:
                jd_data.update({
                    'original_filename': file_metadata.get('original_filename'),
                    'file_size': file_metadata.get('file_size')
                })
            
            # Store in MongoDB
            storage_success = self.db_client.store_jd_data(doc_id, jd_data)
            
            return {
                'success': True,
                'doc_id': doc_id,
                'structure': structure_results,
                'content': content_results,
                'storage': {'success': storage_success}
            }
            
        except Exception as e:
            logger.error(f"Error in JD processing and storage: {e}")
            return {
                'error': str(e),
                'verification': ["❌ Processing failed"]
            }

def test_analyzer():
    """Test function to verify analyzer functionality."""
    analyzer = JDAnalyzer()
    logger.info("Starting analyzer test")
    
    # Test with sample text
    sample_text = """
    Senior Software Engineer
    
    About Us:
    We're a leading tech company...
    
    Responsibilities:
    - Design and implement scalable solutions
    - Lead technical discussions
    
    Requirements:
    - 5+ years experience in Python
    - Bachelor's degree in Computer Science
    
    Benefits:
    - Competitive salary
    - Health insurance
    """
    
    # Test structure analysis
    structure_results = analyzer.analyze_structure(sample_text)
    logger.info("Structure Analysis Results:")
    for msg in structure_results['verification']:
        logger.info(msg)
    
    # Test content processing
    if 'sections' in structure_results:
        content_results = analyzer.process_content(sample_text, structure_results['sections'])
        logger.info("Content Processing Results:")
        for msg in content_results['verification']:
            logger.info(msg)

if __name__ == "__main__":
    test_analyzer() 