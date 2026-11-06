# ScientificReviewer

Streamlit app that runs multi-expert AI peer review on uploaded scientific PDFs (papers, grants, or posters). Reviewers can debate across iterations, optionally with a moderator, and you can download a PDF summary.

## Features

- Upload a PDF and extract text with PyMuPDF
- Configure 1–10 reviewers with custom expertise names, review guidelines, and critique style
- Choose GPT-4o or Gemini 2.0 Flash per reviewer
- Review types: Paper, Grant, or Poster (with default guideline prompts)
- Rating scales: Paper Score (−2 to 2), Star Rating (1–5), or NIH Scale (1–9)
- Multi-iteration discussion with expert dialogue between reviewers
- Optional GPT-4o moderator summaries (interim and final)
- Final analysis with average score (when scores are found in reviews) and downloadable PDF report via ReportLab

## Requirements

See `requirements.txt`:

- streamlit
- openai
- langchain
- langchain-community
- PyMuPDF
- Pillow
- tiktoken
- google-generativeai
- reportlab

## Setup

1. Clone the repo and create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure Streamlit secrets (e.g. `.streamlit/secrets.toml`):

```toml
openai_api_key = "YOUR_OPENAI_API_KEY"
gemini_api_key = "YOUR_GEMINI_API_KEY"
```

## Usage

```bash
streamlit run ScientificReviewer.py
```

Then:

1. Pick a rating scale and review type.
2. Set the number of reviewers and discussion iterations.
3. Optionally enable the moderator (when more than one reviewer).
4. Configure each reviewer’s name, model, guidelines, and critique style.
5. Upload a PDF and click **Start Review**.

## Notes

- Caption in-app: v2.3.0 – Enhanced Dialogue Mode.
- API keys are read from `st.secrets`; they are not taken from environment variables in this script.
