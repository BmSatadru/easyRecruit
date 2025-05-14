import streamlit as st
import pandas as pd
from src.analyzer import (
    extract_text,
    format_content,
    eligibility_check,
    analyze_jd_resume,
    file_create
)
from src.gemini_client import GeminiLLMClient

# Verify API configuration at startup
try:
    GeminiLLMClient.verify_api_configuration()
except Exception as e:
    st.error(f"❌ Failed to initialize Gemini API: {str(e)}")
    st.error("Please check your API key configuration in the .env file.")
    st.stop()

st.set_page_config(
    page_title="JD-Resume Matcher",
    page_icon="📄",
    layout="wide"
)

st.title("📄 JD-Resume Matching Analyzer")
st.write("Upload a Job Description and Resume to analyze the match!")

# File upload section
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Job Description")
    uploaded_jd = st.file_uploader("Choose JD file", type=['pdf', 'docx', 'txt'])

with col2:
    st.subheader("Upload Resume")
    uploaded_resume = st.file_uploader("Choose Resume file", type=['pdf', 'docx', 'txt'])

if uploaded_jd is not None and uploaded_resume is not None:
    with st.spinner('Processing files...'):
        try:
            # Process JD
            jd_file, jd_type = file_create(uploaded_jd, "job_description")
            jd_info = extract_text(jd_file, jd_type, "jd")
            
            if 'error' in jd_info:
                st.error(f"Failed to process Job Description: {jd_info['error']}")
                st.stop()
                
            # Process Resume
            resume_file, resume_type = file_create(uploaded_resume, "resume")
            resume_info = extract_text(resume_file, resume_type, "resume")
            
            if 'error' in resume_info:
                st.error(f"Failed to process Resume: {resume_info['error']}")
                st.stop()

            # Show extracted text and sections in expander
            with st.expander("View Extracted Content"):
                # Job Description tab
                st.subheader("Job Description")
                st.text_area("Extracted Text", jd_info['cleaned_text'], height=200)
                
                # Show sections
                st.write("Identified Sections:")
                for section, content in jd_info['sections'].items():
                    with st.expander(f"📑 {section.title()}"):
                        st.write(content)
                
                # Show industry context
                st.write("Industry Context:")
                st.json(jd_info['industry_context'])
                
                st.divider()
                
                # Resume tab
                st.subheader("Resume")
                st.text_area("Extracted Text", resume_info['cleaned_text'], height=200)
                
                # Show sections
                st.write("Identified Sections:")
                for section, content in resume_info['sections'].items():
                    with st.expander(f"📑 {section.title()}"):
                        st.write(content)
                
                # Show industry context
                st.write("Industry Context:")
                st.json(resume_info['industry_context'])

            # Format content
            with st.spinner('Analyzing content...'):
                jd_content = format_content(jd_info)
                resume_content = format_content(resume_info)

                if "error" in jd_content or "error" in resume_content:
                    st.error("Error in content analysis. Please check the extracted text.")
                    st.stop()

                # Check eligibility
                eligibility_result = eligibility_check(jd_info, resume_info)

                if "error" in eligibility_result:
                    st.error("Error in eligibility check. Please try again.")
                    st.stop()

                # Analyze JD and Resume
                analysis_result = analyze_jd_resume(jd_info, resume_info, eligibility_result)

                if not analysis_result:
                    st.error("Error in final analysis. Please try again.")
                    st.stop()

            # Display results
            st.subheader("Analysis Results")

            # Create tabs for different sections
            tab1, tab2, tab3 = st.tabs(["Match Analysis", "Detailed Content", "Raw Data"])

            with tab1:
                try:
                    # Parse eligibility result
                    eligibility_data = eval(eligibility_result["result"])
                    
                    # Display industry match
                    st.subheader("Industry Alignment")
                    jd_industry = jd_info['industry_context']['primary_industry']
                    resume_industry = resume_info['industry_context']['primary_industry']
                    
                    if jd_industry == resume_industry:
                        st.success(f"✅ Perfect industry match: {jd_industry}")
                    else:
                        st.warning(f"⚠️ Industry mismatch: JD={jd_industry}, Resume={resume_industry}")
                    
                    # Display match percentage with a progress bar
                    st.metric("Match Percentage", f"{eligibility_data['match_percentage']}%")
                    st.progress(float(eligibility_data['match_percentage']) / 100)
                    
                    # Display overall eligibility
                    if eligibility_data['overall_eligibility']:
                        st.success("✅ Candidate is Eligible")
                    else:
                        st.error("❌ Candidate is Not Eligible")
                    
                    # Display key matches and gaps
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("✅ Key Matches")
                        for match in eligibility_data['key_matches']:
                            st.write(f"- {match}")
                    
                    with col2:
                        st.write("❌ Key Gaps")
                        for gap in eligibility_data['key_gaps']:
                            st.write(f"- {gap}")
                    
                    # Display recommendations
                    st.write("📋 Recommendations")
                    st.info(eligibility_data['recommendations'])
                except Exception as e:
                    st.error(f"Error displaying match analysis: {str(e)}")

            with tab2:
                try:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Job Description Analysis")
                        st.json(jd_content)
                    
                    with col2:
                        st.write("Resume Analysis")
                        st.json(resume_content)
                except Exception as e:
                    st.error(f"Error displaying detailed content: {str(e)}")

            with tab3:
                try:
                    st.write("Detailed Analysis Data")
                    st.json(analysis_result)
                except Exception as e:
                    st.error(f"Error displaying raw data: {str(e)}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please try again with different files or contact support if the issue persists.")

else:
    st.info("Please upload both a Job Description and a Resume to begin the analysis.") 