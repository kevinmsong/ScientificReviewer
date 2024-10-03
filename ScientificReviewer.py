import streamlit as st
import logging
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import PyPDF2
import io
from PIL import Image
import base64
import asyncio
import re
from pdf2image import convert_from_bytes
from pdf2image.exceptions import PDFInfoNotInstalledError
import tempfile

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

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Grant Review Functions
async def review_proposal(content, agents, expertises, review_type):
    criteria = ["Significance", "Investigator(s)", "Innovation", "Approach", "Environment"] if review_type == "NIH" else ["Intellectual Merit", "Broader Impacts"]
    review_log = []

    for agent, expertise in zip(agents, expertises):
        review_prompt = f"""
        Please review, as a {expertise} for a postdoctoral scientific audience, the following {'NIH' if review_type == 'NIH' else 'NSF'} project proposal considering these main criteria:
        
        {"1. Significance\n2. Investigator(s)\n3. Innovation\n4. Approach\n5. Environment" if review_type == "NIH" else "1. Intellectual Merit: The potential to advance knowledge\n2. Broader Impacts: The potential to benefit society and contribute to desired societal outcomes"}

        {"" if review_type == "NIH" else '''
        For both criteria, consider:
        a) Potential to advance knowledge/benefit society
        b) Creativity, originality, and transformative potential
        c) Soundness of the plan and assessment mechanism
        d) Qualifications of the team/individual
        e) Adequacy of resources
        '''}

        Additional review principles:
        Focus on the highest quality and potential to advance or transform knowledge frontiers
        Consider broader contributions to societal goals
        Assess based on appropriate metrics, considering project size and resources

        Please provide your review, addressing the following points for each criterion:
        1. Significance of the work
        2. Innovation in the approach
        3. Rigor and reproducibility
        4. Clarity of presentation
        5. Evaluation methods

        For each criterion, provide a harsh and critical review, focusing on weaknesses. Be technical, elaborate, and extremely critical in your assessment.

        Highlight any specific paragraphs that need significant correction by referring to them using their starting words.
        
        If needed, you may use block quotes to point at specific areas that need improvement, and provide concrete suggestions for each quoted section.

        End your review for each criterion with a clear numerical rating from 1 to 9 (1 being the lowest, 9 being the highest) in the following format:
        
        [Criterion Name] Rating: X/9

        Provide a brief summary for each rating, highlighting the main weaknesses and suggesting concrete details for improvement.

        Proposal content:
        {content}

        Your review:
        """

        try:
            response = await agent.ainvoke([HumanMessage(content=review_prompt)])
            review_text = extract_content(response, "[Error: Unable to extract response]")
        except Exception as e:
            logging.error(f"Error getting response: {str(e)}")
            review_text = "[Error: Issue with review]"

        review_log.append({"review": review_text, "expertise": expertise})
        st.write(f"Review by {expertise}:\n\n{review_text}\n\n")

    return review_log

def extract_ratings(review_text, criteria):
    ratings = {}
    for criterion in criteria:
        match = re.search(rf"{criterion} Rating:\s*(\d+)/9", review_text)
        if match:
            ratings[criterion] = int(match.group(1))
        else:
            ratings[criterion] = None
    return ratings

def calculate_lowest_ratings(review_log, criteria):
    lowest_ratings = {criterion: 9 for criterion in criteria}
    lowest_rationales = {criterion: "" for criterion in criteria}

    for review in review_log:
        review_text = review["review"]
        ratings = extract_ratings(review_text, criteria)
        
        for criterion in criteria:
            if ratings[criterion] is not None and ratings[criterion] < lowest_ratings[criterion]:
                lowest_ratings[criterion] = ratings[criterion]
                rationale_match = re.search(rf"{criterion} Rating:.*?\n(.*?)(?=\n\n|\Z)", review_text, re.DOTALL)
                if rationale_match:
                    lowest_rationales[criterion] = rationale_match.group(1).strip()

    return lowest_ratings, lowest_rationales

# Scientific Paper Review Functions
async def review_article(content, agents, expertises):
    review_log = []

    for i, (agent, expertise) in enumerate(zip(agents, expertises)):
        prompt = f"""
        You are an expert in {expertise}. Please review the following abstract/article for peer-reviewed publication.
        
        Focus on significance, innovation, and comprehensive evaluation of approaches (rigor and reproducibility, clarity, evaluation, etc.)
        
        Please be technical, elaborate, and extremely critical. Make the reviews harsher, and focus on weaknesses and specific areas of the paper, section by section.

        If needed, you may use block quotes to point at specific areas that need improvement, and provide concrete suggestions for each quoted section.

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
            response = await agent.ainvoke([HumanMessage(content=prompt)])
            review_text = extract_content(response, f"[Error: Unable to extract response for Reviewer {i+1}]")
        except Exception as e:
            logging.error(f"Error getting response from Reviewer {i+1}: {str(e)}")
            review_text = f"[Error: Issue with Reviewer {i+1}]"

        review_log.append({"reviewer": expertise, "review": review_text})
        st.write(f"Reviewer ({expertise}):\n\n{review_text}\n\n")

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

# Scientific Poster Review Functions
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def scientific_poster_review_page():
    st.header("Scientific Poster Review")

    uploaded_file = st.file_uploader("Upload your poster (PDF or Image)", type=["pdf", "png", "jpg", "jpeg"])

    if uploaded_file is not None and st.button("Start Analysis"):
        st.write("Starting poster analysis process...")
        
        agent = create_review_agents(1)[0]
        
        try:
            if uploaded_file.type == "application/pdf":
                # Extract text from PDF
                text_content = extract_text_from_pdf(uploaded_file)
                analysis_result = asyncio.run(analyze_poster_text(text_content, agent))
            else:
                # Process image
                image = Image.open(uploaded_file)
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                analysis_result = asyncio.run(analyze_poster(img_str, agent))

            st.write("Analysis Result:")
            st.write(analysis_result)

            st.write("Poster analysis completed.")
        except UnidentifiedImageError:
            st.error("The uploaded file could not be identified as an image. Please ensure you're uploading a valid image file.")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")
    else:
        st.info("Please upload a poster (either PDF or image) and click 'Start Analysis'.")

async def analyze_poster_text(text_content, agent):
    prompt = f"""
    This is the text content extracted from a scientific poster. Please analyze it considering the following points:

    1. What is the main problem/challenge being addressed by this project?
    2. How is this project innovative? What methods does it use to address the problem/challenge?
    3. Evaluate the scientific rigor of the poster based on the available information.
    4. Are the results meaningful and well-presented?
    5. How are the results benchmarked or compared to existing work?

    Please be technical, elaborate, and critically analyze the content. You may be harsh in your review. Suggest concrete improvements for each section of the poster.

    Text content:
    {text_content}

    Please provide your analysis:
    """

    try:
        response = await agent.ainvoke([HumanMessage(content=prompt)])
        analysis = extract_content(response, "[Error: Unable to extract response for poster analysis]")
    except Exception as e:
        logging.error(f"Error getting response for poster analysis: {str(e)}")
        analysis = f"[Error: Issue with poster analysis: {str(e)}]"

    return analysis
    
# Main Application
def main():
    st.title("Scientific Reviewer Application")
    
    # Add version number to sidebar
    st.sidebar.text("Version 1.3.0")
    
    # Navigation
    page = st.sidebar.selectbox("Choose a review type", ["Grant Proposal Review", "Scientific Paper Review", "Scientific Poster Review"])
    
    if page == "Grant Proposal Review":
        grant_review_page()
    elif page == "Scientific Paper Review":
        scientific_paper_review_page()
    elif page == "Scientific Poster Review":
        scientific_poster_review_page()

def grant_review_page():
    st.header("Grant Proposal Review")
    
    uploaded_file = st.file_uploader("Upload your project proposal (PDF)", type="pdf")

    if uploaded_file is not None:
        content = extract_text_from_pdf(uploaded_file)

        review_type = st.radio("Select review type:", ("NIH Proposal", "NSF Proposal"))
        
        num_agents = st.number_input("Enter the number of reviewer agents:", min_value=1, max_value=5, value=3)
        
        expertises = []
        for i in range(num_agents):
            expertise = st.text_input(f"Enter expertise for agent {i+1}:", f"Scientific Expert {i+1}")
            expertises.append(expertise)

        if st.button("Start Review"):
            st.write("Starting the review process...")

            agents = create_review_agents(num_agents)
            
            if review_type == "NIH Proposal":
                review_log = asyncio.run(review_proposal(content, agents, expertises, "NIH"))
                criteria = ["Significance", "Investigator(s)", "Innovation", "Approach", "Environment"]
            else:  # NSF Proposal
                review_log = asyncio.run(review_proposal(content, agents, expertises, "NSF"))
                criteria = ["Intellectual Merit", "Broader Impacts"]

            lowest_ratings, lowest_rationales = calculate_lowest_ratings(review_log, criteria)

            st.write("\nLowest Ratings and Rationales:")
            for criterion in criteria:
                st.write(f"{criterion}: {lowest_ratings[criterion]}/9")
                st.write(f"Rationale: {lowest_rationales[criterion]}\n")

            st.write("Review process completed.")

def scientific_paper_review_page():
    st.header("Scientific Paper Review")

    input_method = st.radio("Choose input method:", ("Paste Abstract", "Paste Full Text", "Upload PDF"))

    content = ""
    if input_method == "Paste Abstract":
        content = st.text_area("Paste the abstract here:", height=300)
    elif input_method == "Paste Full Text":
        content = st.text_area("Paste the full text of your paper here:", height=500)
    else:
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file is not None:
            content = extract_text_from_pdf(uploaded_file)

    num_agents = st.number_input("Enter the number of reviewer agents:", min_value=1, max_value=5, value=3)
    
    expertises = []
    for i in range(num_agents):
        expertise = st.text_input(f"Enter expertise for agent {i+1}:", f"Scientific Expert {i+1}")
        expertises.append(expertise)

    if st.button("Start Review"):
        if content:
            st.write("Starting peer review process...")
            
            agents = create_review_agents(num_agents)
            
            review_log = asyncio.run(review_article(content, agents, expertises))

            average_rating = calculate_average_rating(review_log)
            if average_rating:
                st.write(f"\nAverage Rating: {average_rating:.2f}")
                decision = get_editorial_decision(average_rating)
                st.write(f"Recommended Editorial Decision: {decision}")
            else:
                st.write("Unable to calculate average rating.")

            st.write("Peer review process completed.")
        else:
            st.warning("Please provide content for review (either paste an abstract, full text, or upload a PDF).")

if __name__ == "__main__":
    main()
