import streamlit as st
import logging
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import fitz  # PyMuPDF
import io

logging.basicConfig(level=logging.INFO)

# Initialize OpenAI client
api_key = st.secrets["openai_api_key"]
client = OpenAI(api_key=api_key)

def create_review_agents(num_agents, model="gpt-4o"):
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
    for page in pdf_document:
        text_content += page.get_text()
    return text_content

def scientific_poster_review_page():
    st.header("Scientific Poster Review")

    uploaded_file = st.file_uploader("Upload your poster (PDF)", type=["pdf"])

    if uploaded_file is not None and st.button("Start Analysis"):
        st.write("Starting poster analysis process...")
        
        agent = create_review_agents(1)[0]
        
        try:
            text_content = extract_pdf_content(uploaded_file)
            analysis_result = analyze_poster(text_content, agent)

            st.write("Analysis Result:")
            st.write(analysis_result)

            st.write("Poster analysis completed.")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")
    else:
        st.info("Please upload a poster (PDF) and click 'Start Analysis'.")

def analyze_poster(text_content, agent):
    prompt = f"""
    Please analyze this scientific poster based on the following text content. Consider these points:

    1. What is the main problem/challenge being addressed by this project?
    2. How is this project innovative? What methods does it use to address the problem/challenge?
    3. Evaluate the scientific rigor of the poster based on the available information.
    4. Are the results meaningful and well-presented?
    5. How are the results benchmarked or compared to existing work?
    6. Based on the text, evaluate the structure and organization of the poster. Is it effective in communicating the research?
    7. Analyze any mentions of figures, graphs, or visual elements in the text. Are they described clearly and informatively?

    Please be technical, elaborate, and critically analyze the content. You may be harsh in your review. Suggest concrete improvements for each section of the poster based on the text content.

    Here's the text content extracted from the poster:
    {text_content}

    Please provide your analysis:
    """

    try:
        response = agent.invoke([HumanMessage(content=prompt)])
        analysis = extract_content(response, "[Error: Unable to extract response for poster analysis]")
    except Exception as e:
        logging.error(f"Error getting response for poster analysis: {str(e)}")
        analysis = f"[Error: Issue with poster analysis: {str(e)}]"

    return analysis

def grant_review_page():
    st.header("Grant Proposal Review")
    
    uploaded_file = st.file_uploader("Upload your project proposal (PDF)", type="pdf")

    if uploaded_file is not None:
        content = extract_pdf_content(uploaded_file)
        
        review_type = st.radio("Select review type:", ("NIH Proposal", "NSF Proposal"))
        
        num_agents = st.number_input("Enter the number of reviewer agents:", min_value=1, max_value=5, value=3)
        
        expertises = []
        for i in range(num_agents):
            expertise = st.text_input(f"Enter expertise for agent {i+1}:", f"Scientific Expert {i+1}")
            expertises.append(expertise)

        if st.button("Start Review"):
            st.write("Starting the review process...")

            agents = create_review_agents(num_agents)
            
            review_log = review_proposal(content, agents, expertises, review_type)

            for review in review_log:
                st.write(f"Review by {review['expertise']}:")
                st.write(review['review'])

            st.write("Review process completed.")

def review_proposal(content, agents, expertises, review_type):
    criteria = ["Significance", "Investigator(s)", "Innovation", "Approach", "Environment"] if review_type == "NIH Proposal" else ["Intellectual Merit", "Broader Impacts"]
    review_log = []

    for agent, expertise in zip(agents, expertises):
        prompt = f"""
        Please review, as a {expertise} for a postdoctoral scientific audience, the following {'NIH' if review_type == "NIH Proposal" else 'NSF'} project proposal considering these main criteria:
        
        {", ".join(criteria)}

        Additional review principles:
        - Focus on the highest quality and potential to advance or transform knowledge frontiers
        - Consider broader contributions to societal goals
        - Assess based on appropriate metrics, considering project size and resources

        For each criterion, provide a harsh and critical review, focusing on weaknesses. Be technical, elaborate, and extremely critical in your assessment.

        End your review for each criterion with a clear numerical rating from 1 to 9 (1 being the lowest, 9 being the highest) in the following format:
        
        [Criterion Name] Rating: X/9

        Provide a brief summary for each rating, highlighting the main weaknesses and suggesting concrete details for improvement.

        Proposal content:
        {content}

        Your review:
        """

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

    uploaded_file = st.file_uploader("Upload your scientific paper (PDF)", type="pdf")

    if uploaded_file is not None:
        content = extract_pdf_content(uploaded_file)

        num_agents = st.number_input("Enter the number of reviewer agents:", min_value=1, max_value=5, value=3)
        
        expertises = []
        for i in range(num_agents):
            expertise = st.text_input(f"Enter expertise for agent {i+1}:", f"Scientific Expert {i+1}")
            expertises.append(expertise)

        if st.button("Start Review"):
            st.write("Starting peer review process...")
            
            agents = create_review_agents(num_agents)
            
            review_log = review_scientific_paper(content, agents, expertises)

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

def review_scientific_paper(content, agents, expertises):
    review_log = []

    for agent, expertise in zip(agents, expertises):
        prompt = f"""
        You are an expert in {expertise}. Please review the following scientific paper for peer-reviewed publication.
        
        Focus on significance, innovation, and comprehensive evaluation of approaches (rigor and reproducibility, clarity, evaluation, etc.)
        
        Please be technical, elaborate, and extremely critical. Make the review harsh, and focus on weaknesses and specific areas of the paper, section by section.

        Content to review:
        {content}

        Please provide your review, addressing the following points:
        1. Significance of the work
        2. Innovation in the approach
        3. Rigor and reproducibility
        4. Clarity of presentation
        5. Evaluation methods

        End your review with a rating from 1 to 9 (1 being the lowest, 9 being the highest) and a brief summary.

        Your review:
        """

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
    st.sidebar.text("Version 1.6.0")
    
    # Navigation
    page = st.sidebar.selectbox("Choose a review type", ["Grant Proposal Review", "Scientific Paper Review", "Scientific Poster Review"])
    
    if page == "Grant Proposal Review":
        grant_review_page()
    elif page == "Scientific Paper Review":
        scientific_paper_review_page()
    elif page == "Scientific Poster Review":
        scientific_poster_review_page()

if __name__ == "__main__":
    main()
