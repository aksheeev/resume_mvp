# AI-Powered Resume Intelligence System 🚀

A full-stack AI ranking engine designed to parse, analyze, and rank resumes based on job requirements. Unlike simple keyword matching, this system uses **TF-IDF**, **Contextual Chunking**, and a novel **"Fair Transferability"** algorithm to identify the best candidates based on demonstrated experience.

## 🌟 Key Features
* **Contextual Scoring:** Rewards skills found in "Experience" sections (3.0x) over simple lists.
* **Fair Transferability:** Awards bonus points for relevant, non-required skills (e.g., AWS expertise for a Docker role).
* **Length Normalization:** Prevents keyword-stuffing by penalizing ultra-short resumes.
* **96.36% Accuracy:** Benchmarked against the Kaggle Resume Dataset for DevOps roles.

## 🛠️ Tech Stack
* **Python**
* **Streamlit** (Frontend/UI)
* **scikit-learn** (TF-IDF Model)
* **spaCy** (NLP & Skill Matching)

## 🚀 How to Run Locally
1. Clone the repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate it and install dependencies: `pip install -r requirements.txt`
4. Run the app: `streamlit run app.py`