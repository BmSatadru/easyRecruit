import os
import uuid
import logging
from typing import Dict, Optional, Tuple, List
import pypdfium2 as pdfium
from pathlib import Path
import re
import spacy
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class JDProcessor:
    def __init__(self, upload_dir: str = "uploads/jd"):
        """Initialize the JD Processor with a directory for storing uploaded files.
        
        Args:
            upload_dir (str): Directory path for storing uploaded PDFs
        """
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Load spaCy model for company and job title extraction
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Successfully loaded spaCy model")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            self.nlp = None
    
    def _verify_pdf_structure(self, pdf: pdfium.PdfDocument) -> Tuple[bool, List[str]]:
        """Verify PDF structure and content.
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list of verification messages)
        """
        messages = []
        try:
            # Check number of pages
            page_count = len(pdf)
            messages.append(f"✓ PDF has {page_count} pages")
            
            # Check if pages are readable
            for i in range(page_count):
                page = pdf[i]
                text = page.get_textpage().get_text_range()
                char_count = len(text)
                messages.append(f"✓ Page {i+1}: {char_count} characters extracted")
                
                # Verify minimum content
                if char_count < 50:  # Arbitrary minimum for a job description page
                    messages.append(f"⚠️ Warning: Page {i+1} has very little text ({char_count} chars)")
            
            return True, messages
        except Exception as e:
            messages.append(f"❌ Error verifying PDF structure: {str(e)}")
            return False, messages

    def validate_pdf(self, file_content: bytes) -> Tuple[bool, str, List[str]]:
        """Validate if the uploaded file is a readable PDF.
        
        Returns:
            Tuple[bool, str, List[str]]: (is_valid, error_message, verification_messages)
        """
        verification_msgs = []
        try:
            # Try to load the PDF
            pdf = pdfium.PdfDocument(file_content)
            verification_msgs.append("✓ Successfully loaded PDF")
            
            # Verify PDF structure
            is_valid, struct_msgs = self._verify_pdf_structure(pdf)
            verification_msgs.extend(struct_msgs)
            
            if not is_valid:
                return False, "Invalid PDF structure", verification_msgs
            
            return True, "", verification_msgs
        except Exception as e:
            error_msg = f"Invalid PDF file: {str(e)}"
            verification_msgs.append(f"❌ {error_msg}")
            return False, error_msg, verification_msgs

    def save_pdf(self, file_content: bytes, original_filename: str) -> Tuple[str, str, Dict]:
        """Save the PDF file with a unique ID and extract basic metadata.
        
        Returns:
            Tuple[str, str, Dict]: (unique_id, file_path, metadata)
        """
        # Generate a unique ID
        unique_id = str(uuid.uuid4())
        
        # Extract timestamp and create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{unique_id}_{os.path.basename(original_filename)}"
        file_path = self.upload_dir / filename
        
        # Save the file
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        # Extract basic metadata
        metadata = {
            'original_filename': original_filename,
            'storage_path': str(file_path),
            'upload_timestamp': timestamp,
            'file_size': len(file_content),
            'doc_id': unique_id
        }
        
        logger.info(f"Saved PDF: {metadata}")
        return unique_id, str(file_path), metadata

    def extract_text(self, file_path: str) -> Dict[str, any]:
        """Extract and clean text from the PDF file with verification steps."""
        pdf = pdfium.PdfDocument(file_path)
        extraction_info = {
            'pages': {},
            'metadata': {},
            'verification': []
        }
        
        total_chars = 0
        total_words = 0
        
        for page_number in range(len(pdf)):
            page = pdf[page_number]
            textpage = page.get_textpage()
            raw_text = textpage.get_text_range()
            
            # Clean the text
            cleaned_text = self._clean_text(raw_text)
            
            # Collect page statistics
            chars = len(cleaned_text)
            words = len(cleaned_text.split())
            total_chars += chars
            total_words += words
            
            # Store page info
            extraction_info['pages'][f'page_{page_number + 1}'] = {
                'text': cleaned_text,
                'statistics': {
                    'characters': chars,
                    'words': words,
                    'lines': len(cleaned_text.split('\n'))
                }
            }
            
            # Extract potential company name and job title if first page
            if page_number == 0 and self.nlp:
                doc = self.nlp(cleaned_text[:1000])  # Process first 1000 chars
                # Extract potential company names
                orgs = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
                if orgs:
                    extraction_info['metadata']['potential_company'] = orgs[0]
                
                # Extract potential job title using patterns
                title_patterns = [
                    r"(?i)position:\s*(.*?)(?:\n|$)",
                    r"(?i)job title:\s*(.*?)(?:\n|$)",
                    r"(?i)role:\s*(.*?)(?:\n|$)"
                ]
                for pattern in title_patterns:
                    match = re.search(pattern, cleaned_text[:1000])
                    if match:
                        extraction_info['metadata']['potential_job_title'] = match.group(1).strip()
                        break
        
        # Add verification information
        extraction_info['verification'] = [
            f"✓ Processed {len(pdf)} pages",
            f"✓ Extracted {total_chars} characters",
            f"✓ Found {total_words} words",
            f"✓ Average words per page: {total_words // len(pdf)}",
        ]
        
        if 'potential_company' in extraction_info['metadata']:
            extraction_info['verification'].append(
                f"✓ Detected company: {extraction_info['metadata']['potential_company']}"
            )
        if 'potential_job_title' in extraction_info['metadata']:
            extraction_info['verification'].append(
                f"✓ Detected job title: {extraction_info['metadata']['potential_job_title']}"
            )
        
        logger.info(f"Extraction verification: {extraction_info['verification']}")
        return extraction_info

    def _clean_text(self, text: str) -> str:
        """Clean the extracted text with verification steps."""
        original_length = len(text)
        lines = text.split('\n')
        cleaned_lines = []
        
        # Track cleaning statistics
        stats = {
            'headers_removed': 0,
            'empty_lines_removed': 0,
            'whitespace_normalized': 0
        }
        
        for line in lines:
            # Skip headers/footers
            if self._is_header_footer(line):
                stats['headers_removed'] += 1
                continue
            
            # Clean the line
            cleaned_line = line.strip()
            if not cleaned_line:
                stats['empty_lines_removed'] += 1
                continue
                
            # Normalize whitespace
            original_spaces = len(re.findall(r'\s+', cleaned_line))
            cleaned_line = re.sub(r'\s+', ' ', cleaned_line)
            if original_spaces != len(re.findall(r'\s+', cleaned_line)):
                stats['whitespace_normalized'] += 1
            
            # Remove non-ASCII chars
            cleaned_line = re.sub(r'[^\x00-\x7F]+', '', cleaned_line)
            
            if cleaned_line:
                cleaned_lines.append(cleaned_line)
        
        # Join lines with proper spacing
        cleaned_text = '\n'.join(cleaned_lines)
        
        logger.debug(f"Text cleaning stats: {stats}")
        return cleaned_text

    def _is_header_footer(self, line: str) -> bool:
        """Identify headers/footers with common patterns."""
        patterns = [
            r'^\s*Page \d+\s*$',
            r'^\s*\d+/\d+\s*$',
            r'^\s*[-_]{3,}\s*$',
            r'^\s*\d{1,2}/\d{1,2}/\d{2,4}\s*$',  # Date patterns
            r'^\s*[©Cc]opyright\s+.*$',  # Copyright notices
            r'^\s*[Cc]onfidential\s*$',  # Confidential markers
        ]
        
        return any(re.match(pattern, line) for pattern in patterns)

# Example usage and verification code
def test_jd_processor():
    """Test function to verify JD processor functionality."""
    processor = JDProcessor()
    logger.info("Starting JD processor test")
    
    # Test with a sample PDF
    test_file = "path/to/test.pdf"  # Replace with actual test file
    if os.path.exists(test_file):
        with open(test_file, 'rb') as f:
            content = f.read()
            
        # Test validation with verification
        is_valid, error, verification = processor.validate_pdf(content)
        logger.info("PDF Validation Results:")
        for msg in verification:
            logger.info(msg)
        
        if is_valid:
            # Test saving with metadata
            doc_id, path, metadata = processor.save_pdf(content, "test.pdf")
            logger.info(f"Saved PDF with metadata: {metadata}")
            
            # Test text extraction with verification
            extraction_results = processor.extract_text(path)
            logger.info("Extraction Results:")
            for msg in extraction_results['verification']:
                logger.info(msg)
            
            # Log metadata if found
            if 'metadata' in extraction_results:
                logger.info(f"Extracted metadata: {extraction_results['metadata']}")
    else:
        logger.error("Test file not found")

if __name__ == "__main__":
    test_jd_processor() 