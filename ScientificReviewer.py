import streamlit as st
import logging
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import fitz
import io
from PIL import Image
import base64
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)

def create_review_agents(num_agents: int, review_type: str = "paper", include_moderator: bool = False) -> List[ChatOpenAI]:
    """Create review agents including a moderator if specified."""
    # Select model based on review type
    model = "gpt-4-turbo" if review_type == "poster" else "gpt-4"
    
    # Create regular review agents
    agents = [ChatOpenAI(temperature=0.1, openai_api_key=st.secrets["openai_api_key"], model=model) 
             for _ in range(num_agents)]
    
    # Add moderator agent if requested and multiple reviewers
    if include_moderator and num_agents > 1:
        moderator_agent = ChatOpenAI(temperature=0.1, openai_api_key=st.secrets["openai_api_key"], 
                                   model="gpt-4")
        agents.append(moderator_agent)
    
    return agents

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
    for agent, expertise, prompt in zip(agents[:-1], expertises, custom_prompts):
        try:
            full_prompt = prompt + f"\n\nContent to review:\n{content}"
            response = agent.invoke([HumanMessage(content=full_prompt)])
            review_text = extract_content(response, f"[Error: Unable to extract response for {expertise}]")
            review_results.append({
                "expertise": expertise,
                "review": review_text
            })
        except Exception as e:
            logging.error(f"Error getting response from {expertise}: {str(e)}")
            review_results.append({
                "expertise": expertise,
                "review": f"[Error: Issue with review from {expertise}]"
            })
    
    # If we have multiple reviews and a moderator agent, get moderation
    if len(review_results) > 1 and len(agents) > len(expertises):
        try:
            moderator_prompt = get_moderator_prompt(review_results, review_type)
            moderator_response = agents[-1].invoke([HumanMessage(content=moderator_prompt)])
            moderation_result = extract_content(moderator_response, "[Error: Unable to extract moderator response]")
        except Exception as e:
            logging.error(f"Error getting moderator response: {str(e)}")
            moderation_result = "[Error: Issue with moderation]"
    else:
        moderation_result = None
    
    return {
        "individual_reviews": review_results,
        "moderation": moderation_result
    }

def display_review_results(results: Dict[str, Any]) -> None:
    """Display review results in Streamlit."""
    # Display individual reviews
    st.subheader("Individual Reviews")
    for review in results["individual_reviews"]:
        with st.expander(f"Review by {review['expertise']}"):
            st.write(review["review"])
    
    # Display moderation if available
    if results["moderation"]:
        st.subheader("Moderator Analysis")
        st.write(results["moderation"])

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
        max_value=5,
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
                prompt = st.text_area(
                    f"Custom Prompt for Reviewer {i+1}",
                    value=f"As an expert in {expertise}, please review this {review_type.lower()} "
                          "considering scientific merit, methodology, and impact."
                )
                custom_prompts.append(prompt)
    
    # File upload
    uploaded_file = st.file_uploader(f"Upload {review_type} (PDF)", type=["pdf"])
    
    if uploaded_file and st.button("Start Review"):
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

# Add to main()
def main():
    st.title("Enhanced Scientific Review System")
    scientific_review_page()

if __name__ == "__main__":
    main()
