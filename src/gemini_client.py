import json
import google.generativeai as genai
import os
import re
from dotenv import load_dotenv
from .llm_client import LLMClient
from time import sleep

class GeminiLLMClient(LLMClient):
    _instance = None
    _is_initialized = False
    model_name = None  # Class variable to store the model name
    MAX_RETRIES = 3
    RETRY_DELAY = 30  # seconds
    
    # Track quota exceeded models to avoid retrying them
    _quota_exceeded_models = set()

    def __new__(cls):
        if cls._instance is None:
            # Verify configuration before creating instance
            cls.verify_api_configuration()
            cls._instance = super(GeminiLLMClient, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the Gemini client only once."""
        if not self._is_initialized:
            self._initialize_client()
            GeminiLLMClient._is_initialized = True

    @staticmethod
    def _extract_retry_delay(error_message):
        """Extract retry delay from error message."""
        match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)\s*}', error_message)
        if match:
            return int(match.group(1))
        return 30  # default delay

    @classmethod
    def _handle_rate_limit_error(cls, error_message, current_model=None):
        """Handle rate limit errors with appropriate messages and model fallback."""
        if "429" in error_message:
            if "exceeded your current quota" in error_message:
                if current_model:
                    cls._quota_exceeded_models.add(current_model)
                    # Try to verify configuration again with remaining models
                    try:
                        print(f"Quota exceeded for {current_model}, attempting to switch models...")
                        cls.verify_api_configuration(exclude_models=cls._quota_exceeded_models)
                        return None  # Indicate successful fallback
                    except Exception as e:
                        if len(cls._quota_exceeded_models) == len(cls.get_model_candidates()):
                            # All models have exceeded quota
                            raise RuntimeError(
                                "⚠️ API Quota Exceeded for all available models\n\n"
                                "The free tier quota for all Gemini models has been exceeded. To resolve this:\n"
                                "1. Wait for the quota to reset (usually within an hour)\n"
                                "2. Consider upgrading to a paid tier\n"
                                "3. Check your usage at: https://ai.google.dev/pricing\n\n"
                                "For more information on quotas and limits, visit:\n"
                                "https://ai.google.dev/gemini-api/docs/rate-limits"
                            )
                        else:
                            raise e
            elif "rate limit" in error_message.lower():
                retry_delay = cls._extract_retry_delay(error_message)
                raise RuntimeError(
                    f"⚠️ Rate Limit Reached\n\n"
                    f"Please wait {retry_delay} seconds before trying again.\n"
                    "The API has temporary rate limits to ensure fair usage."
                )
        return error_message

    @staticmethod
    def get_model_candidates():
        """Get list of model candidates in order of preference."""
        return [
            "models/gemini-1.5-pro-latest",     # Latest 1.5 Pro
            "models/gemini-1.5-pro",            # Generic 1.5 Pro
            "models/gemini-2.5-flash-preview-04-17",  # Latest Flash Preview - faster, lower quota usage
            "models/gemini-2.0-flash",          # Stable Flash model
            "models/gemini-1.5-flash",          # 1.5 Flash model
            "models/gemini-2.0-flash-lite",     # Lightweight Flash model
            "models/gemini-1.5-flash-8b",       # 8B Flash model - for high volume tasks
            "models/gemini-1.5-pro-002",        # Specific 1.5 Pro version
            "models/gemini-1.5-pro-001",        # Specific 1.5 Pro version
            "models/gemini-pro",                # Legacy name
            "models/gemini-2.0-pro-exp"         # Experimental 2.0
        ]

    @classmethod
    def verify_api_configuration(cls, exclude_models=None):
        """
        Verify API configuration and model availability.
        Raises exception if configuration is invalid.
        
        Args:
            exclude_models: Set of models to exclude from consideration (e.g., due to quota limits)
        """
        # Load environment variables
        load_dotenv()

        # Check API key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "❌ API Key Not Found\n\n"
                "Please set up your Gemini API key:\n"
                "1. Get an API key from: https://makersuite.google.com/app/apikey\n"
                "2. Create a .env file in your project root\n"
                "3. Add the line: GEMINI_API_KEY=your_api_key_here"
            )

        # Configure Gemini API
        genai.configure(api_key=api_key)
        
        # Get available models
        try:
            available_models = [m.name for m in genai.list_models()]
        except Exception as e:
            print(f"Warning: Could not fetch available models: {str(e)}")
            available_models = [f"models/{m}" for m in cls.get_model_candidates()]

        # Initialize exclude_models if None
        exclude_models = exclude_models or set()
        
        # Try each model in order until one works
        last_error = None
        model_candidates = [m for m in cls.get_model_candidates() if m not in exclude_models]
        
        if not model_candidates:
            raise RuntimeError(
                "⚠️ API Quota Exceeded for all available models\n\n"
                "The free tier quota for all Gemini models has been exceeded. To resolve this:\n"
                "1. Wait for the quota to reset (usually within an hour)\n"
                "2. Consider upgrading to a paid tier\n"
                "3. Check your usage at: https://ai.google.dev/pricing\n\n"
                "For more information on quotas and limits, visit:\n"
                "https://ai.google.dev/gemini-api/docs/rate-limits"
            )

        for model_name in model_candidates:
            if model_name not in available_models:
                print(f"Warning: Model {model_name} not available, skipping...")
                continue

            try:
                # Store the selected model name as a class variable
                cls.model_name = model_name.replace('models/', '')  # Remove 'models/' prefix
                
                # Test model access with minimal content
                model = genai.GenerativeModel(cls.model_name)
                test_response = model.generate_content("Test")
                if not test_response:
                    raise ValueError("Failed to generate test content")
                
                print(f"✅ Gemini API configuration verified successfully using model: {cls.model_name}")
                return True
                
            except Exception as e:
                error_message = str(e)
                try:
                    # Handle rate limit errors
                    if "429" in error_message and "quota" in error_message.lower():
                        print(f"Quota exceeded for {model_name}, attempting to switch models...")
                        exclude_models.add(model_name)
                        last_error = e
                        continue
                    elif "rate limit" in error_message.lower():
                        retry_delay = cls._extract_retry_delay(error_message)
                        print(f"Rate limit reached. Waiting {retry_delay} seconds before retry...")
                        sleep(retry_delay)
                        # Retry the same model after delay
                        continue
                    else:
                        last_error = e
                        print(f"Error with model {model_name}: {error_message}")
                        continue
                except Exception as handle_error:
                    last_error = handle_error
                    continue

        # If we've exhausted all models, raise the last error
        cls.model_name = None  # Reset model name on failure
        if last_error:
            raise RuntimeError(f"Failed to configure Gemini API with any available model: {str(last_error)}")

    def _initialize_client(self):
        """Initialize the client with the verified configuration."""
        try:
            if not self.__class__.model_name:
                raise ValueError("Model name not set. Please verify API configuration first.")
            
            self.model = genai.GenerativeModel(self.__class__.model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini client: {str(e)}")

    def _handle_api_call(self, prompt: str) -> str:
        """Helper method to handle API calls with proper error handling."""
        retries = 0
        while retries < self.MAX_RETRIES:
            try:
                response = self.model.generate_content(prompt)
                if not response or not response.text:
                    raise ValueError("Empty response from API")
                return response.text.strip()
            except Exception as e:
                error_msg = str(e)
                print(f"Error in API call: {error_msg}")
                
                try:
                    # Handle rate limit errors
                    error_msg = self._handle_rate_limit_error(error_msg)
                except RuntimeError as rate_error:
                    # If it's a quota error, raise it immediately
                    if "quota" in str(rate_error):
                        return f"Error: {str(rate_error)}"
                    # For rate limits, wait and retry
                    retry_delay = self._extract_retry_delay(str(e))
                    print(f"Rate limit reached. Waiting {retry_delay} seconds before retry...")
                    sleep(retry_delay)
                    retries += 1
                    continue
                
                # For model-specific errors, try to reinitialize
                if "model" in error_msg.lower():
                    try:
                        print("Attempting to reinitialize with verified model...")
                        self.__class__.verify_api_configuration()
                        self._initialize_client()
                        continue
                    except Exception as reinit_error:
                        print(f"Reinitialization failed: {str(reinit_error)}")
                
                retries += 1
                if retries < self.MAX_RETRIES:
                    print(f"Retrying API call... (Attempt {retries + 1}/{self.MAX_RETRIES})")
                    sleep(self.RETRY_DELAY)
            
        return "Error: Failed to generate response after multiple attempts"

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