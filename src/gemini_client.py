import json
import google.generativeai as genai
import os
import re
from dotenv import load_dotenv
import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)

def rate_limit_decorator(max_retries=3, retry_delay=15):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if "quota" in str(e).lower():
                        if attempt < max_retries - 1:
                            logger.warning(f"Rate limit hit. Waiting {retry_delay} seconds. Attempt {attempt + 1}/{max_retries}")
                            time.sleep(retry_delay)
                            continue
                    raise
            return func(*args, **kwargs)
        return wrapper
    return decorator

class GeminiLLMClient:
    """Client for handling JD data extraction using Google's Gemini API."""
    
    def __init__(self):
        """Initialize the Gemini client."""
        self.verify_api_configuration()
        self._initialize_client()

    @staticmethod
    def verify_api_configuration():
        """Verify and set up API configuration."""
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "❌ API Key Not Found\n\n"
                "Please set up your Gemini API key:\n"
                "1. Get an API key from: https://makersuite.google.com/app/apikey\n"
                "2. Create a .env file in your project root\n"
                "3. Add the line: GEMINI_API_KEY=your_api_key_here"
            )
        
        genai.configure(api_key=api_key)

    def _initialize_client(self):
        """Initialize the Gemini model with fallbacks."""
        logger.info("Initializing Gemini model")
        
        # Updated model names to match the latest Gemini API versions
        models = [
            'gemini-2.0-flash',  # Latest Flash model
            'gemini-2.0-flash-001',  # Stable Flash model
            'gemini-2.0-flash-lite',  # Lightweight Flash model
            'gemini-1.5-pro-001',  # Fallback Pro model
            'gemini-1.5-pro'  # Last resort fallback
        ]
        
        last_error = None
        for model_name in models:
            try:
                logger.info(f"Attempting to initialize model: {model_name}")
                self.model = genai.GenerativeModel(model_name)
                
                # Test the model with a minimal prompt to verify connectivity
                test_response = self.model.generate_content("Test.")
                if test_response and hasattr(test_response, 'text'):
                    logger.info(f"Successfully initialized model: {model_name}")
                    return
                
            except Exception as e:
                last_error = str(e)
                if "quota" in last_error.lower():
                    logger.warning(f"Rate limit hit for {model_name}. Waiting 60 seconds before trying next model...")
                    time.sleep(60)  # Wait before trying next model
                elif "not found" in last_error.lower():
                    logger.warning(f"Model {model_name} not available: {last_error}")
                else:
                    logger.warning(f"Failed to initialize {model_name}: {last_error}")
                continue
        
        error_msg = (
            "Failed to initialize any Gemini model. "
            "Please check:\n"
            "1. Your API key is valid and has sufficient quota\n"
            "2. You have network connectivity\n"
            "3. The API service is available\n\n"
            f"Last error: {last_error}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    @rate_limit_decorator(max_retries=3, retry_delay=15)
    def _handle_api_call(self, prompt: str) -> str:
        """Handle API calls with error handling and retries."""
        logger.info("\n" + "*"*50)
        logger.info("GEMINI CLIENT PROCESSING STARTED")
        logger.info("*"*50)
        
        try:
            logger.info("\nReceived Prompt:")
            logger.info("-"*30)
            logger.info(prompt[:500] + "..." if len(prompt) > 500 else prompt)
            
            # Add safety check for model initialization
            if not hasattr(self, 'model'):
                logger.info("\nInitializing model...")
                self._initialize_client()
            
            logger.info("\nMaking API call...")
            response = self.model.generate_content(prompt)
            
            if not response or not response.text:
                logger.error("\nEmpty response from Gemini API")
                raise ValueError("Empty response from API")
                
            logger.info("\nAPI Response Received:")
            logger.info("-"*30)
            logger.info(response.text[:500] + "..." if len(response.text) > 500 else response.text)
            
            # Try to parse as JSON to verify structure
            try:
                json_response = json.loads(response.text)
                logger.info("\nValid JSON structure verified")
                logger.info("-"*30)
                logger.info(json.dumps(json_response, indent=2))
            except json.JSONDecodeError:
                logger.warning("\nResponse is not JSON format")
            
            logger.info("\n" + "*"*50)
            logger.info("GEMINI CLIENT PROCESSING COMPLETED")
            logger.info("*"*50 + "\n")
            
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"\nGemini API call error: {str(e)}")
            # Return a structured error response
            error_response = json.dumps({
                "core_metadata": {
                    "company": "Not specified",
                    "job_title": "Not specified",
                    "location": "Not specified",
                    "employment_type": "Not specified"
                },
                "error": str(e)
            })
            logger.info("\nReturning error response:")
            logger.info("-"*30)
            logger.info(error_response)
            return error_response

    def process_jd_content(self, text: str) -> dict:
        """Process job description content and extract structured information.
        
        Args:
            text: The job description text to analyze
            
        Returns:
            Dict containing structured JD information
        """
        prompt = """
Extract key information from this job description and return it in valid JSON format:
{
    "core_metadata": {
        "company": "string",
        "job_title": "string",
        "location": "string",
        "employment_type": "string"
    },
    "requirements": {
        "skills": {
            "technical": ["list of technical skills"],
            "soft": ["list of soft skills"]
        },
        "experience": "string",
        "education": "string"
    },
    "job_details": {
        "responsibilities": ["list of responsibilities"],
        "team_structure": "string",
        "growth_opportunities": "string"
    },
    "additional_info": {
        "industry": "string",
        "seniority_level": "string",
        "travel_requirements": "string",
        "remote_policy": "string"
    }
}

Job Description:
{text}

Important: Ensure the response is a valid JSON object with all the fields shown above. Use null for missing values.
"""
        try:
            response = self._handle_api_call(prompt)
            parsed_data = json.loads(response)
            
            # Add default core_metadata if missing
            if 'core_metadata' not in parsed_data:
                parsed_data['core_metadata'] = {
                    "company": "Not specified",
                    "job_title": "Not specified",
                    "location": "Not specified",
                    "employment_type": "Not specified"
                }
            
            return parsed_data
        except Exception as e:
            logger.error(f"Error in process_jd_content: {str(e)}")
            return {
                "core_metadata": {
                    "company": "Not specified",
                    "job_title": "Not specified",
                    "location": "Not specified",
                    "employment_type": "Not specified"
                },
                "error": str(e)
            }

    def format_content(self, text: str) -> dict:
        """Format and structure the content of a resume or job description."""
        
        # First determine if it's a JD or resume
        doc_type_prompt = """
Analyze this text and determine if it's a Job Description (JD) or Resume. Look for key indicators:
- JD: Requirements, responsibilities, qualifications needed, company offering a role
- Resume: Personal experience, achievements, education history, work history

Text:
{text}

Respond with ONLY 'JD' or 'RESUME'.
"""
        doc_type_response = self._handle_api_call(doc_type_prompt.format(text=text))
        is_jd = doc_type_response.strip().upper() == 'JD'

        print(f"\nDocument Type Detection: {'JD' if is_jd else 'RESUME'}")
        
        # Define section patterns exactly as in DocumentProcessor
        section_patterns = {
            'requirements': ['requirements', 'qualifications', 'required skills'],
            'responsibilities': ['responsibilities', 'duties', 'job duties', 'role'],
            'company_info': ['company', 'about us', 'about company'],
            'benefits': ['benefits', 'perks', 'what we offer']
        } if is_jd else {
            'education': ['education', 'academic background', 'qualifications'],
            'experience': ['experience', 'work experience', 'employment history', 'work history'],
            'skills': ['skills', 'technical skills', 'core competencies', 'expertise'],
            'summary': ['summary', 'professional summary', 'profile', 'about']
        }

        print("\nIdentifying sections...")
        
        # Use DocumentProcessor's section identification logic
        sections = {}
        lines = text.split('\n')
        current_section = 'unknown'
        current_content = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if line matches any section header
            matched_section = None
            for section, patterns in section_patterns.items():
                if any(pattern in line.lower() for pattern in patterns):
                    matched_section = section
                    break

            if matched_section:
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                current_section = matched_section
                current_content = []
            else:
                current_content.append(line)

        if current_content:
            sections[current_section] = '\n'.join(current_content)

        print(f"Found sections: {list(sections.keys())}")
        
        # If no sections found, treat entire text as one section
        if not sections or (len(sections) == 1 and 'unknown' in sections):
            print("\nNo standard sections found, analyzing full text...")
            if is_jd:
                sections = {
                    'requirements': text,
                    'responsibilities': text
                }
            else:
                sections = {
                    'experience': text,
                    'skills': text
                }

        # Process each section with type-specific prompts
        results = {}
        for section, content in sections.items():
            print(f"\nProcessing section: {section}")
            if not content.strip():
                print(f"Skipping empty section: {section}")
                continue

            try:
                if is_jd:
                    if section == 'requirements':
                        prompt = f"""
Extract specific requirements from this job description section. Focus on:
- Required qualifications
- Technical skills needed
- Experience requirements
- Educational requirements

Text:
{content}

Format as JSON with:
{{
    "qualifications": ["list of required qualifications"],
    "technical_skills": ["list of technical skills"],
    "experience": ["list of experience requirements"],
    "education": ["list of educational requirements"]
}}
"""
                    elif section == 'responsibilities':
                        prompt = f"""
Extract key responsibilities from this job description section. Focus on:
- Main duties
- Key accountabilities
- Project responsibilities
- Team interactions

Text:
{content}

Format as JSON with:
{{
    "main_duties": ["list of main duties"],
    "key_accountabilities": ["list of accountabilities"],
    "project_responsibilities": ["list of project responsibilities"],
    "team_interactions": ["list of team interactions"]
}}
"""
                    elif section == 'company_info':
                        prompt = f"""
Extract company information from this section. Focus on:
- Industry type
- Company size/scale
- Company culture
- Work environment

Text:
{content}

Format as JSON with:
{{
    "industry": "primary industry",
    "company_details": ["list of company details"],
    "culture_points": ["list of culture points"],
    "work_environment": ["list of environment details"]
}}
"""
                    else:  # benefits or unknown
                        prompt = f"""
Extract key information from this section. Focus on:
- Benefits offered
- Perks
- Growth opportunities
- Additional advantages

Text:
{content}

Format as JSON with:
{{
    "benefits": ["list of benefits"],
    "perks": ["list of perks"],
    "opportunities": ["list of opportunities"],
    "additional": ["list of additional points"]
}}
"""
                else:
                    if section == 'education':
                        prompt = f"""
Extract education details from this section. Focus on:
- Degrees earned
- Institutions
- Graduation dates
- Academic achievements

Text:
{content}

Format as JSON with:
{{
    "degrees": ["list of degrees"],
    "institutions": ["list of institutions"],
    "graduation_dates": ["list of dates"],
    "achievements": ["list of achievements"]
}}
"""
                    elif section == 'experience':
                        prompt = f"""
Extract experience details from this section. Focus on:
- Job titles
- Companies
- Dates
- Key achievements
- Responsibilities

Text:
{content}

Format as JSON with:
{{
    "job_titles": ["list of titles"],
    "companies": ["list of companies"],
    "dates": ["list of dates"],
    "achievements": ["list of achievements"],
    "responsibilities": ["list of responsibilities"]
}}
"""
                    elif section == 'skills':
                        prompt = f"""
Extract skills information from this section. Focus on:
- Technical skills
- Soft skills
- Tools/Technologies
- Methodologies

Text:
{content}

Format as JSON with:
{{
    "technical_skills": ["list of technical skills"],
    "soft_skills": ["list of soft skills"],
    "tools": ["list of tools/technologies"],
    "methodologies": ["list of methodologies"]
}}
"""
                    else:  # summary or unknown
                        prompt = f"""
Extract key information from this section. Focus on:
- Professional summary
- Key strengths
- Career objectives
- Notable points

Text:
{content}

Format as JSON with:
{{
    "summary": ["list of summary points"],
    "strengths": ["list of strengths"],
    "objectives": ["list of objectives"],
    "notable_points": ["list of notable points"]
}}
"""

                print(f"Sending prompt for section: {section}")
                section_response = self._handle_api_call(prompt)
                print(f"Got response for section: {section}")
                
                try:
                    section_data = json.loads(section_response)
                    results[section] = section_data
                    print(f"Successfully parsed section: {section}")
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON for section {section}: {str(e)}")
                    print("Response preview:")
                    print(section_response[:200] + "..." if len(section_response) > 200 else section_response)
                    continue
                
            except Exception as e:
                print(f"Error processing section {section}: {str(e)}")
                continue

        print("\nCombining results...")
        
        # Combine results into final format
        try:
            # Extract information
            industry_info = self._extract_industry_info(results, is_jd)
            role_info = self._extract_role_info(results, is_jd)
            qual_info = self._extract_qualification_info(results, is_jd)

            final_result = {
                "primary_industry": industry_info["primary"],
                "secondary_industries": industry_info["secondary"],
                "industry_confidence": industry_info["confidence"],
                "detected_terms": {
                    "technical_terms": role_info["technical_terms"],
                    "industry_specific_terms": industry_info["terms"],
                    "methodologies": role_info["methodologies"]
                },
                "role_analysis": {
                    "seniority_level": role_info["seniority"],
                    "role_category": role_info["category"],
                    "key_responsibilities": role_info["responsibilities"],
                    "required_expertise": role_info["expertise"]
                },
                "qualifications": {
                    "education": qual_info["education"],
                    "certifications": qual_info["certifications"],
                    "domain_knowledge": qual_info["domain_knowledge"]
                }
            }

            # Generate summary last to include all gathered information
            final_result["analysis_summary"] = self._generate_summary(
                industry_info["primary"],
                role_info["expertise"][:3],
                role_info["responsibilities"][:5]
            )

            print("\nAnalysis complete!")
            return {"result": json.dumps(final_result, indent=2)}

        except Exception as e:
            print(f"\nError combining results: {str(e)}")
            print("Available results:", results.keys())
            return {"error": "Failed to combine section results", "raw_results": results}

    def _extract_industry_info(self, results, is_jd):
        """Extract industry information from section results."""
        if is_jd:
            company_info = results.get('company_info', {})
            primary = company_info.get('industry', 'Unknown')
            terms = company_info.get('company_details', [])
            confidence = 90 if primary != 'Unknown' else 70
        else:
            experience = results.get('experience', {})
            companies = experience.get('companies', [])
            primary = self._determine_primary_industry(companies)
            terms = experience.get('responsibilities', [])
            confidence = 80 if companies else 60

        return {
            "primary": primary,
            "secondary": self._determine_secondary_industries(terms),
            "confidence": confidence,
            "terms": terms
        }

    def _extract_role_info(self, results, is_jd):
        """Extract role information from section results."""
        if is_jd:
            resp = results.get('responsibilities', {})
            reqs = results.get('requirements', {})
            return {
                "technical_terms": reqs.get('technical_skills', []),
                "methodologies": reqs.get('methodologies', []),
                "seniority": self._extract_seniority(resp.get('main_duties', [])),
                "category": self._extract_role_category(resp.get('main_duties', [])),
                "responsibilities": resp.get('main_duties', []) + resp.get('key_accountabilities', []),
                "expertise": reqs.get('technical_skills', []) + reqs.get('experience', [])
            }
        else:
            exp = results.get('experience', {})
            skills = results.get('skills', {})
            return {
                "technical_terms": skills.get('technical_skills', []),
                "methodologies": skills.get('methodologies', []),
                "seniority": self._extract_seniority(exp.get('job_titles', [])),
                "category": self._extract_role_category(exp.get('responsibilities', [])),
                "responsibilities": exp.get('responsibilities', []),
                "expertise": skills.get('technical_skills', []) + skills.get('tools', [])
            }

    def _extract_qualification_info(self, results, is_jd):
        """Extract qualification information from section results."""
        if is_jd:
            reqs = results.get('requirements', {})
            return {
                "education": reqs.get('education', []),
                "certifications": reqs.get('qualifications', []),
                "domain_knowledge": reqs.get('experience', [])
            }
        else:
            edu = results.get('education', {})
            skills = results.get('skills', {})
            return {
                "education": edu.get('degrees', []),
                "certifications": edu.get('achievements', []),
                "domain_knowledge": skills.get('methodologies', []) + skills.get('tools', [])
            }

    def _determine_primary_industry(self, companies):
        """Determine primary industry from company names."""
        # Add industry detection logic here
        return "Technology" if any('tech' in c.lower() or 'software' in c.lower() for c in companies) else "Unknown"

    def _determine_secondary_industries(self, terms):
        """Determine secondary industries from terms."""
        industries = []
        industry_keywords = {
            "Finance": ["banking", "financial", "investment"],
            "Healthcare": ["medical", "health", "clinical"],
            "Manufacturing": ["manufacturing", "production", "industrial"],
            "Retail": ["retail", "commerce", "sales"],
            "Technology": ["software", "IT", "tech"]
        }
        
        terms_text = " ".join(terms).lower()
        for industry, keywords in industry_keywords.items():
            if any(keyword in terms_text for keyword in keywords):
                industries.append(industry)
        
        return industries

    def _extract_seniority(self, points):
        """Extract seniority level from points."""
        seniority_indicators = {
            'senior': ['senior', 'lead', 'principal', 'head'],
            'mid': ['mid', 'intermediate', '3-5 years', '2-4 years'],
            'entry': ['entry', 'junior', 'graduate', '0-2 years', '1-2 years']
        }
        
        for level, indicators in seniority_indicators.items():
            if any(ind in ' '.join(points).lower() for ind in indicators):
                return level.title()
        return "Not Specified"

    def _extract_role_category(self, points):
        """Extract role category from points."""
        categories = {
            'Engineering': ['engineer', 'developer', 'programmer', 'technical'],
            'Marketing': ['marketing', 'brand', 'digital', 'seo'],
            'Sales': ['sales', 'business development', 'account'],
            'Management': ['manager', 'director', 'lead', 'head'],
            'Design': ['designer', 'ux', 'ui', 'creative'],
            'Operations': ['operations', 'logistics', 'supply chain'],
            'HR': ['hr', 'human resources', 'recruitment', 'talent'],
            'Finance': ['finance', 'accounting', 'financial', 'analyst']
        }
        
        points_text = ' '.join(points).lower()
        for category, indicators in categories.items():
            if any(ind in points_text for ind in indicators):
                return category
        return "Other"

    def _generate_summary(self, industry, expertise, points):
        """Generate a brief summary based on extracted information."""
        summary_prompt = f"""
Generate a brief 2-3 sentence summary based on:
Industry: {industry}
Expertise Areas: {', '.join(expertise[:3])}
Key Points: {', '.join(points[:5])}

Keep the summary concise and focused on the main role and requirements.
"""
        try:
            return self._handle_api_call(summary_prompt)
        except:
            return "Summary generation failed"

    def eligibility_check(self, job_description: str, resume: str) -> dict:
        prompt = f"""
Analyze the following job description and resume to determine the eligibility of the applicant. Provide a comprehensive assessment covering:

1. Key Requirements Analysis:
- Required skills and qualifications from JD
- Required experience levels
- Essential responsibilities

2. Candidate Qualifications Analysis:
- Matching skills and qualifications
- Relevant experience
- Education alignment

3. Detailed Match Analysis:
- Direct requirement matches
- Equivalent qualifications (e.g., BTech/B.E. considered same)
- Additional beneficial skills

4. Gap Analysis:
- Missing required skills
- Experience shortfalls
- Qualification gaps

Provide a detailed assessment in JSON format with:
- overall_eligibility (boolean)
- match_percentage (0-100)
- detailed_analysis (string)
- key_matches (array)
- key_gaps (array)
- recommendations (string)

Job Description:
{job_description}

Resume:
{resume}
"""
        response = self._handle_api_call(prompt)
        return {"result": response}

    def analyze_jd_resume(self, job_description: str, resume: str, applicant_eligibility: str) -> dict:
        prompt = f"""
As an advanced hiring manager algorithm, analyze the job description and resume to extract detailed matching information. Consider the provided eligibility assessment.

Generate a response in valid JSON format with the following structure:
{{
    "applicants": [
        {{
            "name": "extracted from resume",
            "email": "extracted from resume",
            "mobile": "extracted from resume",
            "skills_matching": "skills from JD that applicant has",
            "skills_missing": "skills from JD that applicant lacks",
            "eligible_for_role": "based on provided eligibility assessment",
            "education_background": "from resume",
            "total_work_experience": "total years",
            "relevant_work_experience": "relevant years",
            "skills": ["array of skills"],
            "certifications": ["array of certifications"],
            "languages": ["array of languages"],
            "additional_information": "any other relevant details"
        }}
    ]
}}

Job Description:
{job_description}

Resume:
{resume}

Applicant Eligibility Assessment:
{applicant_eligibility}
"""
        response = self._handle_api_call(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"error": "Failed to parse JSON response", "raw_response": response} 