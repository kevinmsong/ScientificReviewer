import streamlit as st
import logging
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import fitz
import io
from PIL import Image
import base64

logging.basicConfig(level=logging.INFO)

# Initialize OpenAI client
api_key = st.secrets["openai_api_key"]
client = OpenAI(api_key=api_key)

def create_review_agents(num_agents, review_type="paper"):
    # Select model based on review type
    if review_type == "poster":
        model = "gpt-4-turbo"
    else:
        model = "gpt-4o"
        
    return [ChatOpenAI(temperature=0.1, openai_api_key=api_key, model=model) for _ in range(num_agents)]

def extract_content(response, default_value):
    if isinstance(response, str):
        return response
    elif hasattr(response, 'content'):
        return response.content
    elif isinstance(response, list) and len(response) > 0:
        return response[0].content
    else:
        logging.warning(f"Unexpected response type: {type(response)}")
        return default_value

def extract_pdf_content(pdf_file):
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

def load_default_prompts():
    return {
        "poster_review": """
        Please review the research poster thoroughly and provide the following:
        
        1. Summary:
        {summary_section}
        
        2. Scoring Criteria:
        {scoring_section}
        
        3. Detailed Evaluation:
        {evaluation_section}
        
        4. Figure Analysis:
        {figure_section}
        
        5. Constructive Suggestions:
        {suggestions_section}
        """,
        "grant_review": """
        Please review, as a {expertise} for a postdoctoral scientific audience, the following {grant_type} project proposal considering these main criteria:
        
        {criteria}
        
        1. Summary:
        {summary_section}
        
        2. Scoring Criteria:
        {scoring_section}
        
        3. Detailed Evaluation:
        {evaluation_section}
        
        4. Constructive Suggestions:
        {suggestions_section}
        """,
        "paper_review": """
        You are an expert in {expertise}. Please review the following scientific paper for peer-reviewed publication.
        
        1. Summary:
        {summary_section}
        
        2. Scoring Criteria:
        {scoring_section}
        
        3. Detailed Evaluation:
        {evaluation_section}
        
        4. Figure Analysis:
        {figure_section}
        
        5. Constructive Suggestions:
        {suggestions_section}
        """
    }

def scientific_poster_review_page():
    st.header("Scientific Poster Review")
    
    # Load default prompts
    default_prompts = load_default_prompts()
    
    # Allow customization of prompt sections
    st.subheader("Customize Review Prompt")
    with st.expander("Edit Prompt Sections"):
        summary_section = st.text_area(
            "Summary Section",
            value="Start with a concise summary of the poster's overall strengths and weaknesses.\n"
                  "Highlight key areas where the poster excels and areas that need significant improvement.",
            height=100
        )
        
        scoring_section = st.text_area(
            "Scoring Section",
            value="Provide an overall score for the poster as well as a breakdown of scores for each section. "
                  "Use a 1-9 scale, where:\n1 is extremely poor,\n5 is satisfactory, and\n9 is flawless.",
            height=100
        )
        
        evaluation_section = st.text_area(
            "Evaluation Section",
            value="For each section, provide specific feedback and suggestions for improvement",
            height=100
        )
        
        figure_section = st.text_area(
            "Figure Analysis Section",
            value="Analyze each figure in the poster. Describe what the figure shows, its relevance to the research, "
                  "and any improvements that could be made.",
            height=100
        )
        
        suggestions_section = st.text_area(
            "Suggestions Section",
            value="For each section, offer specific, actionable recommendations on how to improve the content, "
                  "presentation, or overall quality.",
            height=100
        )

    uploaded_file = st.file_uploader("Upload your poster (PDF)", type=["pdf"])

    if uploaded_file is not None and st.button("Start Analysis"):
        st.write("Starting poster analysis process...")
        
        agent = create_review_agents(1, review_type="poster")[0]
        
        try:
            text_content, images = extract_pdf_content(uploaded_file)
            
            # Construct custom prompt
            custom_prompt = default_prompts["poster_review"].format(
                summary_section=summary_section,
                scoring_section=scoring_section,
                evaluation_section=evaluation_section,
                figure_section=figure_section,
                suggestions_section=suggestions_section
            )
            
            analysis_result = analyze_poster(text_content, images, agent, custom_prompt)

            st.write("Analysis Result:")
            st.write(analysis_result)

            st.write("Poster analysis completed.")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")
    else:
        st.info("Please upload a poster (PDF) and click 'Start Analysis'.")

def analyze_poster(text_content, images, agent, prompt_template):
    prompt = prompt_template + f"\n\nHere's the text content extracted from the poster:\n{text_content}"

    try:
        message_content = [{"type": "text", "text": prompt}]
        for i, img in enumerate(images):
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_str}"
                }
            })

        response = agent.invoke([HumanMessage(content=message_content)])
        analysis = extract_content(response, "[Error: Unable to extract response for poster analysis]")
    except Exception as e:
        logging.error(f"Error getting response for poster analysis: {str(e)}")
        analysis = f"[Error: Issue with poster analysis: {str(e)}]"

    return analysis

def grant_review_page():
    st.header("Grant Proposal Review")
    
    # Load default prompts
    default_prompts = load_default_prompts()
    
    # Allow customization of prompt sections
    st.subheader("Customize Review Prompt")
    with st.expander("Edit Prompt Sections"):
        summary_section = st.text_area(
            "Summary Section",
            value="Start with a concise summary of the proposal's overall strengths and weaknesses.\n"
                  "Highlight key areas where the proposal excels and areas that need significant improvement.",
            height=100
        )
        
        scoring_section = st.text_area(
            "Scoring Section",
            value="Provide an overall score for the proposal and scores for each section (1-9 scale).",
            height=100
        )
        
        evaluation_section = st.text_area(
            "Evaluation Section",
            value="For each section, provide specific feedback and suggestions for improvement.",
            height=100
        )
        
        suggestions_section = st.text_area(
            "Suggestions Section",
            value="For each section, offer specific, actionable recommendations for improvement.",
            height=100
        )
    
    uploaded_file = st.file_uploader("Upload your project proposal (PDF)", type="pdf")

    if uploaded_file is not None:
        content = extract_pdf_content(uploaded_file)[0]
        
        review_type = st.radio("Select review type:", ("NIH Proposal", "NSF Proposal"))
        
        num_agents = st.number_input("Enter the number of reviewer agents:", min_value=1, max_value=5, value=3)
        
        expertises = []
        for i in range(num_agents):
            expertise = st.text_input(f"Enter expertise for agent {i+1}:", f"Scientific Expert {i+1}")
            expertises.append(expertise)

        if st.button("Start Review"):
            st.write("Starting the review process...")

            agents = create_review_agents(num_agents, review_type="grant")
            
            # Construct custom prompt
            custom_prompt = default_prompts["grant_review"].format(
                expertise="{expertise}",
                grant_type=review_type,
                criteria="{criteria}",
                summary_section=summary_section,
                scoring_section=scoring_section,
                evaluation_section=evaluation_section,
                suggestions_section=suggestions_section
            )
            
            review_log = review_proposal(content, agents, expertises, review_type, custom_prompt)

            for review in review_log:
                st.write(f"Review by {review['expertise']}:")
                st.write(review['review'])

            st.write("Review process completed.")

def review_proposal(content, agents, expertises, review_type, prompt_template):
    criteria = ["Significance", "Investigator(s)", "Innovation", "Approach", "Environment"] if review_type == "NIH Proposal" else ["Intellectual Merit", "Broader Impacts"]
    review_log = []

    for agent, expertise in zip(agents, expertises):
        prompt = prompt_template.format(
            expertise=expertise,
            criteria=", ".join(criteria)
        ) + f"\n\nProposal content:\n{content}"

        try:
            response = agent.invoke([HumanMessage(content=prompt)])
            review_text = extract_content(response, "[Error: Unable to extract response]")
        except Exception as e:
            logging.error(f"Error getting response: {str(e)}")
            review_text = "[Error: Issue with review]"

        review_log.append({"review": review_text, "expertise": expertise})

    return review_log

def scientific_paper_review_page():
    st.header("Scientific Paper Review")
    
    # Load default prompts
    default_prompts = load_default_prompts()
    
    # Allow customization of prompt sections
    st.subheader("Customize Review Prompt")
    with st.expander("Edit Prompt Sections"):
        summary_section = st.text_area(
            "Summary Section",
            value="Start with a concise summary of the paper's overall strengths and weaknesses.\n"
                  "Highlight key areas where the paper excels and areas that need significant improvement.",
            height=100
        )
        
        scoring_section = st.text_area(
            "Scoring Section",
            value="Provide an overall score for the paper and scores for each section (1-9 scale).",
            height=100
        )
        
        evaluation_section = st.text_area(
            "Evaluation Section",
            value="For each section, provide specific feedback and suggestions for improvement.",
            height=100
        )
        
        figure_section = st.text_area(
            "Figure Analysis Section",
            value="Analyze each figure in the paper. Describe what the figure shows, its relevance, "
                  "and suggested improvements.",
            height=100
        )
        
        suggestions_section = st.text_area(
            "Suggestions Section",
            value="For each section, offer specific, actionable recommendations for improvement.",
            height=100
        )

    uploaded_file = st.file_uploader("Upload your scientific paper (PDF)", type="pdf")

    if uploaded_file is not None:
        content = extract_pdf_content(uploaded_file)[0]

        num_agents = st.number_input("Enter the number of reviewer agents:", min_value=1, max_value=5, value=3)
        
        expertises = []
        for i in range(num_agents):
            expertise = st.text_input(f"Enter expertise for agent {i+1}:", f"Scientific Expert {i+1}")
            expertises.append(expertise)

        if st.button("Start Review"):
            st.write("Starting peer review process...")
            
            agents = create_review_agents(num_agents, review_type="paper")
            
            # Construct custom prompt
            custom_prompt = default_prompts["paper_review"].format(
                expertise="{expertise}",
                summary_section=summary_section,
                scoring_section=scoring_section,
                evaluation_section=evaluation_section,
                figure_section=figure_section,
                suggestions_section=suggestions_section
            )
            
            review_log = review_scientific_paper(content, agents, expertises, custom_prompt)

            for review in review_log:
                st.write(f"Review by {review['reviewer']}:")
                st.write(review['review'])

            average_rating = calculate_average_rating(review_log)
            if average_rating:
                st.write(f"\nAverage Rating: {average_rating:.2f}")
                decision = get_editorial_decision(average_rating)
                st.write(f"Recommended Editorial Decision: {decision}")
            else:
                st.write("Unable to calculate average rating.")

            st.write("Peer review process completed.")

def review_scientific_paper(content, agents, expertises, prompt_template):
    review_log = []

    for agent, expertise in zip(agents, expertises):
        prompt = prompt_template.format(
            expertise=expertise
        ) + f"\n\nContent to review:\n{content}"

        try:
            response = agent.invoke([HumanMessage(content=prompt)])
            review_text = extract_content(response, f"[Error: Unable to extract response for Reviewer {expertise}]")
        except Exception as e:
            logging.error(f"Error getting response from Reviewer {expertise}: {str(e)}")
            review_text = f"[Error: Issue with Reviewer {expertise}]"

        review_log.append({"reviewer": expertise, "review": review_text})

    return review_log

def calculate_average_rating(review_log):
    ratings = []
    for review in review_log:
        review_text = review["review"]
        try:
            rating = int(review_text.split("Rating:")[-1].split("/")[0].strip())
            ratings.append(rating)
        except ValueError:
            st.warning(f"Could not extract rating from {review['reviewer']}'s review.")
    
    if ratings:
        average_rating = sum(ratings) / len(ratings)
        return average_rating
    else:
        return None

def get_editorial_decision(average_rating):
    if average_rating is None:
        return "Unable to determine"
    elif average_rating >= 7:
        return "Accept"
    elif 5 <= average_rating < 7:
        return "Minor Revision"
    elif 3 <= average_rating < 5:
        return "Major Revision"
    else:
        return "Reject"

def main():
    st.title("Scientific Reviewer Application")
    
    # Add version number to sidebar
    st.sidebar.text("Version 1.4.0")
    
    # Add model information to sidebar
    st.sidebar.markdown("### Model Information")
    st.sidebar.markdown("- Poster Review: GPT-4 Turbo")
    st.sidebar.markdown("- Paper Review: GPT-4o")
    st.sidebar.markdown("- Grant Review: GPT-4o")
    
    # Show settings expander in sidebar
    with st.sidebar.expander("Settings"):
        st.markdown("### Default Settings")
        # Temperature settings
        default_temp = st.slider(
            "Default Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            help="Controls randomness in the response. Lower values are more focused."
        )
        
        # Maximum tokens
        max_tokens = st.number_input(
            "Maximum Response Length",
            min_value=100,
            max_value=4000,
            value=2000,
            step=100,
            help="Maximum number of tokens in the response"
        )
    
    # Navigation
    page = st.sidebar.selectbox(
        "Choose a review type",
        ["Grant Proposal Review", "Scientific Paper Review", "Scientific Poster Review"]
    )
    
    if page == "Grant Proposal Review":
        grant_review_page()
    elif page == "Scientific Paper Review":
        scientific_paper_review_page()
    elif page == "Scientific Poster Review":
        scientific_poster_review_page()

# Add configuration for the Streamlit page
st.set_page_config(
    page_title="Scientific Reviewer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve the appearance
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

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logging.exception("Unexpected error in main application:")
