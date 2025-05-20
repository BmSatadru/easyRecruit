import streamlit as st
import logging
import tempfile
from pathlib import Path
from datetime import datetime
from src.jd_processor import JDProcessor
from src.gemini_client import GeminiLLMClient
from src.jd_analyzer import JDAnalyzer
import time
import json

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize processors
jd_processor = JDProcessor()
jd_analyzer = JDAnalyzer()
llm_client = None

# Page configuration
st.set_page_config(
    page_title="JD Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Job Description Analyzer")
st.write("Upload a Job Description to analyze its key components.")

# Verify Gemini API configuration
try:
    logger.info("Verifying Gemini API configuration...")
    GeminiLLMClient.verify_api_configuration()
    llm_client = GeminiLLMClient()
    logger.info("Gemini API configuration verified successfully")
except Exception as e:
    logger.error(f"Failed to initialize Gemini API: {str(e)}")
    st.error(f"❌ Failed to initialize Gemini API: {str(e)}")
    st.error("Please check your API key configuration in the .env file.")
    st.stop()

def process_jd(uploaded_file) -> dict:
    """Process the uploaded JD file with verification steps."""
    try:
        logger.info(f"Processing JD file: {uploaded_file.name}")
        
        # Step 1: Validate PDF
        with st.spinner("🔍 Validating PDF..."):
            is_valid, error, verification_msgs = jd_processor.validate_pdf(uploaded_file.getvalue())
            
            # Show validation results
            with st.expander("📋 PDF Validation Results", expanded=True):
                for msg in verification_msgs:
                    if "❌" in msg:
                        st.error(msg)
                    elif "⚠️" in msg:
                        st.warning(msg)
                    else:
                        st.success(msg)
            
            if not is_valid:
                st.error(f"Invalid PDF: {error}")
                return None
        
        # Step 2: Save and Process PDF
        with st.spinner("💾 Saving and processing PDF..."):
            # Save with metadata
            doc_id, file_path, metadata = jd_processor.save_pdf(uploaded_file.getvalue(), uploaded_file.name)
            
            # Show metadata
            with st.expander("📌 Document Metadata"):
                st.json(metadata)
            
            # Extract and process text
            extraction_results = jd_processor.extract_text(file_path)
            
            # Show extraction verification
            with st.expander("📊 Text Extraction Statistics", expanded=True):
                for msg in extraction_results['verification']:
                    st.write(msg)
            
            # Combine results
            results = {
                'metadata': metadata,
                'extraction': extraction_results,
                'doc_id': doc_id,
                'file_path': file_path
            }
            
            return results
            
    except Exception as e:
        logger.error(f"Error processing JD: {str(e)}", exc_info=True)
        st.error(f"Error processing JD: {str(e)}")
        return None

def analyze_jd_content(results: dict) -> dict:
    try:
        logger.info(f"Analyzing JD content for document {results['doc_id']}")
        
        # Get the full text from all pages
        full_text = "\n".join(
            page['text'] for page in results['extraction']['pages'].values()
        )
        
        # Debug log the text being processed
        logger.debug("Text being processed:")
        logger.debug(full_text[:500] + "..." if len(full_text) > 500 else full_text)
        
        # Get file metadata from results
        file_metadata = results.get('metadata', {})
        logger.debug("File metadata:")
        logger.debug(json.dumps(file_metadata, indent=2))
        
        # Process and store JD data with retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1} to process JD")
                analysis_results = jd_analyzer.process_and_store_jd(
                    results['doc_id'],
                    full_text,
                    file_metadata
                )
                logger.debug("Analysis results:")
                logger.debug(json.dumps(analysis_results, indent=2))
                break
            except Exception as e:
                logger.error(f"Error in attempt {attempt + 1}: {str(e)}")
                if "quota" in str(e).lower() and attempt < max_retries - 1:
                    st.warning(f"Rate limit hit. Retrying in 15 seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(15)
                    continue
                raise
        
        # Safe display of results
        if analysis_results and 'content' in analysis_results:
            content_data = analysis_results['content']
            extracted_data = content_data.get('extracted_data', {})
            
            # Always ensure core_metadata exists
            core_metadata = extracted_data.get('core_metadata', {
                "company": "Not specified",
                "job_title": "Not specified",
                "location": "Not specified",
                "employment_type": "Not specified"
            })
            
            # Display with error handling
            try:
                st.write("#### Core Metadata")
                st.json(core_metadata)
            except Exception as e:
                st.error(f"Error displaying metadata: {str(e)}")
        
        # Show structure analysis results
        with st.expander("📊 Structure Analysis", expanded=True):
            # Show verification messages
            for msg in analysis_results['structure']['verification']:
                if "✓" in msg:
                    st.success(msg)
                elif "⚠️" in msg:
                    st.warning(msg)
                else:
                    st.info(msg)
            
            # Show statistics
            if 'stats' in analysis_results['structure']:
                st.write("Section Statistics:")
                st.json(analysis_results['structure']['stats'])
        
        # Show content processing results
        with st.expander("🎯 Content Analysis", expanded=True):
            if 'content' in analysis_results:
                content_data = analysis_results['content']
                
                # Safely get extracted data
                extracted_data = content_data.get('extracted_data', {})
                
                # Safely display core metadata
                core_metadata = extracted_data.get('core_metadata', {})
                st.write("#### Core Metadata")
                st.json({
                    "company": core_metadata.get('company', 'Not specified'),
                    "job_title": core_metadata.get('job_title', 'Not specified'),
                    "location": core_metadata.get('location', 'Not specified'),
                    "employment_type": core_metadata.get('employment_type', 'Not specified')
                })
                
                # Display other sections with safe gets
                sections = ['requirements', 'job_details', 'additional_info']
                for section in sections:
                    if section in extracted_data:
                        st.write(f"#### {section.replace('_', ' ').title()}")
                        st.json(extracted_data[section])
        
        # Show storage status
        if analysis_results.get('storage', {}).get('success'):
            st.success("✅ Job Description data stored successfully in database")
        else:
            st.error("❌ Failed to store Job Description data in database")
        
        return analysis_results
            
    except Exception as e:
        logger.error(f"Error in JD analysis: {str(e)}", exc_info=True)
        st.error(f"Error analyzing JD content: {str(e)}")
        return None

# Main interface
st.write("### Step 1: Upload Job Description")
uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])

if uploaded_file:
    # Process the JD
    results = process_jd(uploaded_file)
    
    if results:
        st.success(f"✅ Job Description processed successfully (ID: {results['doc_id']})")
        
        # Analyze the content
        analysis_results = analyze_jd_content(results)
        
        if analysis_results:
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["Overview", "Detailed Analysis", "Raw Content"])
            
            with tab1:
                st.subheader("📑 Document Overview")
                
                # Show basic document info
                st.write("### 📊 Document Statistics")
                total_words = sum(page['statistics']['words'] 
                                for page in results['extraction']['pages'].values())
                total_chars = sum(page['statistics']['characters'] 
                                for page in results['extraction']['pages'].values())
                
                # Add debug logging
                logger.debug("Extraction Results:")
                logger.debug(f"Total Words: {total_words}")
                logger.debug(f"Total Characters: {total_chars}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Pages", len(results['extraction']['pages']))
                with col2:
                    st.metric("Total Words", total_words)
                with col3:
                    st.metric("Total Characters", total_chars)
                
                # Show key metadata if available
                if 'content' in analysis_results and 'extracted_data' in analysis_results['content']:
                    data = analysis_results['content']['extracted_data']
                    # Add debug logging for extracted data
                    logger.debug("Extracted Data:")
                    logger.debug(json.dumps(data, indent=2))
                    
                    if 'core_metadata' in data:
                        st.write("### 🎯 Key Information")
                        meta = data['core_metadata']
                        logger.debug("Core Metadata:")
                        logger.debug(json.dumps(meta, indent=2))
                        
                        if meta.get('job_title'):
                            st.write(f"**Position:** {meta['job_title']}")
                        if meta.get('company'):
                            st.write(f"**Company:** {meta['company']}")
                        if meta.get('location'):
                            st.write(f"**Location:** {meta['location']}")
                        
                        # Add additional key information
                        if 'requirements' in data:
                            st.write("### 📋 Key Requirements")
                            if 'skills' in data['requirements']:
                                skills = data['requirements']['skills']
                                if skills.get('technical'):
                                    st.write("**Technical Skills:**")
                                    st.write(", ".join(skills['technical'][:5]) + ("..." if len(skills['technical']) > 5 else ""))
                                if skills.get('soft'):
                                    st.write("**Soft Skills:**")
                                    st.write(", ".join(skills['soft'][:5]) + ("..." if len(skills['soft']) > 5 else ""))
                        
                        # Add experience and education if available
                        if data['requirements'].get('experience'):
                            st.write("**Experience Required:**", data['requirements']['experience'])
                        if data['requirements'].get('education'):
                            st.write("**Education Required:**", data['requirements']['education'])
            
            with tab2:
                st.subheader("🔍 Analysis Results")
                
                # Show structure analysis
                if 'structure' in analysis_results:
                    with st.expander("📊 Document Structure"):
                        st.json(analysis_results['structure'])
                
                # Show content analysis
                if 'content' in analysis_results:
                    with st.expander("📑 Extracted Content"):
                        st.json(analysis_results['content'])
            
            with tab3:
                st.subheader("📄 Raw Content")
                for page_num, page_data in results['extraction']['pages'].items():
                    with st.expander(f"📃 {page_num}"):
                        st.write("Statistics:")
                        st.json(page_data['statistics'])
                        st.write("Content:")
                        st.text_area("", page_data['text'], height=200)
else:
    st.info("👆 Please upload a Job Description PDF to begin analysis.")

# Show debug log viewer
with st.expander("🔍 Debug Log"):
    try:
        with open('debug.log', 'r') as log_file:
            st.code(log_file.read(), language='text')
    except FileNotFoundError:
        st.info("No debug log file found yet. It will be created when you process a document.")
    except Exception as e:
        st.error(f"Error reading debug log: {str(e)}")