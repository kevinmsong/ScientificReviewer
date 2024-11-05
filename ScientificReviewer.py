import streamlit as st
import logging
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import fitz
import io
from PIL import Image
import base64
from typing import List, Dict, Any, Tuple, Union

logging.basicConfig(level=logging.INFO)

# Initialize OpenAI client
api_key = st.secrets["openai_api_key"]
client = OpenAI(api_key=api_key)

def create_review_agents(num_agents: int, review_type: str = "paper", include_moderator: bool = False) -> List[ChatOpenAI]:
    """Create review agents including a moderator if specified."""
    # Select model based on review type
    model = "gpt-4-turbo" if review_type == "poster" else "gpt-4"
    
    # Create regular review agents
    agents = [ChatOpenAI(temperature=0.1, openai_api_key=api_key, model=model) 
             for _ in range(num_agents)]
    
    # Add moderator agent if requested and multiple reviewers
    if include_moderator and num_agents > 1:
        moderator_agent = ChatOpenAI(temperature=0.1, openai_api_key=api_key, 
                                   model="gpt-4")
        agents.append(moderator_agent)
    
    return agents

def extract_content(response: Union[str, Any], default_value: str) -> str:
    """Extract content from various response types."""
    if isinstance(response, str):
        return response
    elif hasattr(response, 'content'):
        return response.content
    elif isinstance(response, list) and len(response) > 0:
        return response[0].content
    else:
        logging.warning(f"Unexpected response type: {type(response)}")
        return default_value

def extract_pdf_content(pdf_file) -> Tuple[str, List[Image.Image]]:
    """Extract text and images from a PDF file."""
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text_content = ""
    images = []
    
    for page in pdf_document:
        text_content += page.get_text()
        for img in page.get_images():
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)
    
    return text_content, images

def get_moderator_prompt(review_results: List[Dict[str, Any]], review_type: str) -> str:
    """Generate the prompt for the moderator agent."""
    prompt = """As a senior scientific moderator, analyze the following peer reviews and:
    1. Evaluate the scientific rigor and objectivity of each review
    2. Identify the most valid and well-supported points from each review
    3. Compile a comprehensive summary incorporating the best insights
    4. Assign a final score (1-9) based on the collective reviews
    5. Provide clear strengths and suggestions for improvement

    Previous reviews to analyze:
    """
    
    for idx, review in enumerate(review_results, 1):
        prompt += f"\nReview {idx} (by {review['expertise']}):\n{review['review']}\n"
    
    prompt += "\nPlease provide your analysis in the following format:\n"
    prompt += """
    REVIEW ANALYSIS:
    [Analysis of each review's scientific rigor and key points]
    
    COMPREHENSIVE SUMMARY:
    [Synthesis of the most valid insights]
    
    FINAL SCORE: [1-9]
    
    STRENGTHS:
    - [Key strength points]
    
    SUGGESTIONS FOR IMPROVEMENT:
    - [Key improvement points]
    """
    
    return prompt

def process_reviews(content: str, agents: List[ChatOpenAI], expertises: List[str], 
                   custom_prompts: List[str], review_type: str) -> Dict[str, Any]:
    """Process reviews from multiple agents and moderator."""
    review_results = []
    
    # Get reviews from each agent
    for i, (agent, expertise, prompt) in enumerate(zip(agents[:-1], expertises, custom_prompts)):
        try:
            # Structure the review prompt more explicitly
            full_prompt = f"""As an expert in {expertise}, please review the following {review_type}.

{prompt}

Here is the content to review:

{content}

Please structure your review with the following sections:
1. Overview and Summary (2-3 sentences)
2. Technical Analysis
3. Methodology Assessment
4. Strengths
5. Weaknesses
6. Suggestions for Improvement
7. Scores (1-9) for:
   - Scientific Merit
   - Technical Rigor
   - Clarity
   - Impact
   - Overall Score

Please be specific and provide justification for your assessments."""

            # Make the API call with proper error handling
            try:
                response = agent.invoke([HumanMessage(content=full_prompt)])
                review_text = extract_content(response, f"[Error: Unable to extract response for {expertise}]")
            except Exception as api_error:
                logging.error(f"API Error for {expertise}: {str(api_error)}")
                review_text = f"API Error occurred while getting review from {expertise}. Please try again."
            
            review_results.append({
                "expertise": expertise,
                "review": review_text,
                "success": True
            })
            
        except Exception as e:
            logging.error(f"Error in review process for {expertise}: {str(e)}")
            review_results.append({
                "expertise": expertise,
                "review": f"An error occurred while processing review from {expertise}. Error: {str(e)}",
                "success": False
            })
    
    # If we have multiple reviews and a moderator agent, get moderation
    if len(review_results) > 1 and len(agents) > len(expertises):
        try:
            # Enhanced moderator prompt with better structure
            moderator_prompt = """As a senior scientific moderator, please analyze the following reviews:

"""
            # Add only successful reviews to the moderator's analysis
            successful_reviews = [r for r in review_results if r.get("success", False)]
            
            for idx, review in enumerate(successful_reviews, 1):
                moderator_prompt += f"\nREVIEW {idx} (by {review['expertise']}):\n{review['review']}\n"
            
            moderator_prompt += """
Please provide your analysis in the following structured format:

1. REVIEW ANALYSIS
For each review, assess:
- Scientific rigor and methodology
- Objectivity and evidence-based reasoning
- Key valid points and insights
- Any potential biases or limitations

2. SYNTHESIS OF KEY POINTS
- Areas of consensus
- Important disagreements
- Most substantiated criticisms
- Most valuable suggestions

3. FINAL ASSESSMENT
- Overall score (1-9): [Score]
- Key strengths: [List 3-5 main strengths]
- Key weaknesses: [List 3-5 main weaknesses]
- Priority improvements: [List 3-5 main suggestions]
- Final recommendation: [Accept/Major Revision/Minor Revision/Reject]

Please be specific and provide justification for your assessments."""

            # Make the API call for moderation
            try:
                moderator_response = agents[-1].invoke([HumanMessage(content=moderator_prompt)])
                moderation_result = extract_content(moderator_response, "[Error: Unable to extract moderator response]")
            except Exception as mod_error:
                logging.error(f"Moderator API Error: {str(mod_error)}")
                moderation_result = "Error occurred during moderation. Please try again."
            
        except Exception as e:
            logging.error(f"Error in moderation process: {str(e)}")
            moderation_result = f"An error occurred during moderation. Error: {str(e)}"
    else:
        moderation_result = None
    
    return {
        "individual_reviews": review_results,
        "moderation": moderation_result
    }

def display_review_results(results: Dict[str, Any]) -> None:
    """Display review results in Streamlit with enhanced error handling."""
    try:
        # Display individual reviews with better formatting
        st.subheader("Individual Reviews")
        for review in results["individual_reviews"]:
            with st.expander(f"Review by {review['expertise']}", expanded=True):
                if review.get("success", False):
                    # Split the review into sections for better readability
                    sections = review['review'].split('\n\n')
                    for section in sections:
                        st.write(section.strip())
                        st.markdown("---")
                else:
                    st.error(review['review'])
        
        # Display moderation if available
        if results["moderation"]:
            st.subheader("Moderator Analysis")
            
            # Check if moderation was successful
            if not results["moderation"].startswith("[Error"):
                # Split moderator analysis into sections for better readability
                sections = results["moderation"].split('\n\n')
                for section in sections:
                    st.write(section.strip())
                    st.markdown("---")
            else:
                st.error(results["moderation"])
    
    except Exception as e:
        st.error(f"Error displaying results: {str(e)}")
        logging.exception("Error in display_review_results:")

def get_default_prompt(review_type: str, expertise: str) -> str:
    """Get default prompt based on review type."""
    prompts = {
        "Paper": f"""As an expert in {expertise}, please provide a comprehensive review of this scientific paper considering:
            Strengths and Weaknesses
            
            1. Scientific Merit and Novelty
            2. Methodology and Technical Rigor
            3. Data Analysis and Interpretation
            4. Clarity and Presentation
            5. Impact and Significance
            
            Suggestions for Improvement
            
            Please provide scores (1-9) for each aspect and an overall score.""",
        
        "Grant": f"""As an expert in {expertise}, please evaluate this grant proposal considering:
            Strengths and Weaknesses
            
            1. Innovation and Significance
            2. Approach and Methodology
            3. Feasibility and Timeline
            4. Budget Justification
            5. Expected Impact

            Suggestions for Improvement
            
            Please provide scores (1-9) for each aspect and an overall score.""",
        
        "Poster": f"""As an expert in {expertise}, please review this scientific poster considering:
            Strengths and Weaknesses
            
            1. Visual Appeal and Organization
            2. Scientific Content
            3. Methodology Presentation
            4. Results and Conclusions
            5. Impact and Relevance

            Suggestions for Improvement
            
            Please provide scores (1-9) for each aspect and an overall score."""
    }
    return prompts.get(review_type, "Please provide a thorough review of this submission.")

def scientific_review_page():
    st.header("Scientific Review System")
    
    # Review type selection
    review_type = st.selectbox(
        "Select Review Type",
        ["Paper", "Grant", "Poster"]
    )
    
    # Number of reviewers
    num_reviewers = st.number_input(
        "Number of Reviewers",
        min_value=1,
        max_value=10,
        value=2
    )
    
    # Option for moderator when multiple reviewers
    use_moderator = False
    if num_reviewers > 1:
        use_moderator = st.checkbox("Include Moderator/Judge Review", value=True)
    
    # Collect expertise and custom prompts for each reviewer
    expertises = []
    custom_prompts = []
    
    with st.expander("Configure Reviewers"):
        for i in range(num_reviewers):
            col1, col2 = st.columns(2)
            with col1:
                expertise = st.text_input(f"Expertise for Reviewer {i+1}", 
                                        value=f"Scientific Expert {i+1}")
                expertises.append(expertise)
            
            with col2:
                default_prompt = get_default_prompt(review_type, expertise)
                prompt = st.text_area(
                    f"Custom Prompt for Reviewer {i+1}",
                    value=default_prompt,
                    height=200
                )
                custom_prompts.append(prompt)
    
    # File upload
    uploaded_file = st.file_uploader(f"Upload {review_type} (PDF)", type=["pdf"])
    
    if uploaded_file and st.button("Start Review"):
        try:
            st.write("Starting review process...")
            
            # Extract content
            content = extract_pdf_content(uploaded_file)[0]
            
            # Create agents including moderator if selected
            agents = create_review_agents(num_reviewers, review_type.lower(), use_moderator)
            
            # Process reviews
            results = process_reviews(content, agents, expertises, custom_prompts, review_type.lower())
            
            # Display results
            display_review_results(results)
            
            st.write("Review process completed.")
            
        except Exception as e:
            st.error(f"An error occurred during the review process: {str(e)}")
            logging.exception("Error in review process:")

def main():
    st.set_page_config(
        page_title="Scientific Review System",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS
    st.markdown("""
        <style>
        .stButton>button {
            width: 100%;
            margin-top: 1rem;
        }
        .stExpander {
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-bottom: 1rem;
        }
        .streamlit-expanderHeader {
            background-color: #f8f9fa;
        }
        .stTextArea>div>div>textarea {
            font-family: monospace;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.title("Enhanced Scientific Review System")
    
    # Add version number and info to sidebar
    st.sidebar.text("Version 2.0.0")
    st.sidebar.markdown("### Model Information")
    st.sidebar.markdown("- Review Model: GPT-4")
    st.sidebar.markdown("- Moderator Model: GPT-4")
    
    scientific_review_page()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logging.exception("Unexpected error in main application:")
