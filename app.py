# app.py

import streamlit as st
import spacy
from spacy.matcher import PhraseMatcher
import json
import PyPDF2
import docx
import io
import pandas as pd
from pathlib import Path
from collections import defaultdict
import re # Import Regex for chunking
from sklearn.feature_extraction.text import TfidfVectorizer

# --- CONFIGURATION & PATHS ---
BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"

# --- SCORING CONSTANTS (v4.5) ---
CONTEXT_MULTIPLIERS = {
    "experience": 3.0,
    "projects": 2.0,
    "skills": 1.0, # Skills section is base
    "education": 1.0,
    "other": 1.0
}
# (v4.5) This is the "fair" bonus weight
TRANSFERABLE_BONUS_WEIGHT = 0.25 

# (v4.4) Keywords for the "Segmentation" Chunker
SECTION_KEYWORDS = {
    "experience": ['experience', 'employment', 'work history', 'professional experience', 'career history', 'work experience', 'relevant experience'],
    "projects": ['projects', 'personal projects', 'portfolio', 'academic projects', 'relevant projects'],
    "education": ['education', 'academic background', 'academic qualifications', 'certifications', 'courses'],
    "skills": ['skills', 'technical skills', 'proficiencies', 'technologies', 'technical expertise', 'core competencies', 'summary', 'objective']
}

# --- RESOURCE LOADING (CACHED FOR PERFORMANCE) ---

@st.cache_resource
def load_nlp_model():
    """Loads the spaCy 'en_core_web_sm' model."""
    nlp = spacy.load("en_core_web_sm")
    return nlp

@st.cache_resource
def load_json_file(file_path):
    """Utility function to load a JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"FATAL ERROR: Config file not found at {file_path}")
        return None

def normalize_skill(skill):
    """(BUG FIX) Normalize skill names for consistent matching"""
    return skill.lower().strip()

@st.cache_resource
def load_all_configs():
    """Loads all configuration files from the 'config' directory."""
    skills_raw = load_json_file(CONFIG_DIR / "skills_base.json")
    ontology = load_json_file(CONFIG_DIR / "skill_ontology.json")
    
    if skills_raw is None or ontology is None:
        st.stop()

    # (BUG FIX) Normalize all skills on load
    skills = sorted(list(set([normalize_skill(s) for s in skills_raw])))

    skill_to_categories = defaultdict(list)
    for category, skills_in_category in ontology.items():
        for skill in skills_in_category:
            skill_to_categories[normalize_skill(skill)].append(category)

    return skills, ontology, skill_to_categories

@st.cache_resource
def create_skill_matcher(_nlp, skills_list):
    """(BUG FIX) Creates a spaCy PhraseMatcher with all skills consistently lowercased."""
    matcher = PhraseMatcher(_nlp.vocab, attr="LOWER")
    patterns = [_nlp.make_doc(skill) for skill in skills_list] 
    matcher.add("SKILL", patterns)
    return matcher

# --- FILE PARSING FUNCTIONS ---
def parse_pdf(file_bytes):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception: return None

def parse_docx(file_bytes):
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception: return None

def parse_txt(file_bytes):
    try:
        return file_bytes.decode("utf-8")
    except Exception: return None

# --- NEW ML & SCORING LOGIC (v4.5) ---

def chunk_resume_text(text):
    """(v4.4) Enhanced chunking that segments text by finding all headers first."""
    chunks = {
        "experience": "",
        "projects": "", 
        "education": "",
        "skills": "",
        "other": ""
    }
    
    text_lower = text.lower()
    section_starts = {} # This will store {start_index: section_name}
    
    # 1. Find the start index of all section headers
    for section, keywords in SECTION_KEYWORDS.items():
        for keyword in keywords:
            try:
                for match in re.finditer(r'\b' + re.escape(keyword) + r'\b', text_lower):
                    line_start = text_lower.rfind('\n', 0, match.start()) + 1
                    line_end = text_lower.find('\n', match.end())
                    if line_end == -1: line_end = len(text_lower)
                    
                    line_length = line_end - line_start
                    if line_length < 100: 
                        section_starts[match.start()] = section
                        break 
            except re.error:
                continue 

    if not section_starts:
        chunks["other"] = text
        return chunks

    # 2. Sort the found sections by their start index
    sorted_starts = sorted(section_starts.items())
    
    # 3. Assign text before the first header to "other"
    first_section_start = sorted_starts[0][0]
    chunks["other"] = text[:first_section_start]
    
    # 4. Segment the text based on these start points
    for i, (start_idx, section_name) in enumerate(sorted_starts):
        end_idx = len(text) 
        if i + 1 < len(sorted_starts):
            end_idx = sorted_starts[i+1][0] 
        
        chunks[section_name] += text[start_idx:end_idx] + "\n"
    
    return chunks

def normalize_tfidf_by_resume_length(tfidf_scores_dict, text):
    """Prevent short resumes from getting inflated scores"""
    word_count = len(text.split())
    
    if word_count < 100:
        length_factor = 0.3
    elif word_count < 300:
        length_factor = 0.7
    else:
        length_factor = 1.0
    
    normalized_scores = {}
    for skill, score in tfidf_scores_dict.items():
        normalized_scores[skill] = score * length_factor
    
    return normalized_scores, length_factor

def extract_skills_from_chunks(chunks, matcher, nlp):
    """(BUG FIX) Improved skill extraction with normalization"""
    skill_locations = defaultdict(set)
    
    for zone, text in chunks.items():
        if not text.strip():
            continue
            
        doc = nlp(text)
        matches = matcher(doc)
        
        for match_id, start, end in matches:
            skill_found = doc[start:end].text.lower().strip()
            skill_locations[skill_found].add(zone)
    
    return skill_locations

def get_context_multiplier(skill, skill_locations):
    """Applies the multiplier based on the best zone a skill was found in."""
    zones = skill_locations.get(skill, {"other"})
    
    if "experience" in zones:
        return CONTEXT_MULTIPLIERS["experience"]
    if "projects" in zones:
        return CONTEXT_MULTIPLIERS["projects"]
    if "skills" in zones:
        return CONTEXT_MULTIPLIERS["skills"] 
    
    return CONTEXT_MULTIPLIERS["other"]

def calculate_skill_score(skill, skill_locations, tfidf_scores_dict):
    """Calculates the final depth score for a single skill."""
    base_score = tfidf_scores_dict.get(skill, 0) 
    if base_score == 0:
        return 0, "N/A (Base: 0.0)"
        
    multiplier = get_context_multiplier(skill, skill_locations)
    final_score = base_score * multiplier
    remark = f"(Score: {final_score:.2f} = Base: {base_score:.2f} * Mult: {multiplier}x)"
    return final_score, remark

# --- DEBUGGING FUNCTIONS ---
def debug_skill_matching(filename, text, required_skills, skill_locations, chunks):
    """Debug function to see what skills are being matched"""
    st.markdown(f"--- \n ### 🔍 Debug Info: `{filename}`")
    st.write(f"**Required skills:** `{', '.join(required_skills)}`")
    
    found_skill_names = list(skill_locations.keys())
    st.write(f"**All found skills ({len(found_skill_names)}):** `{', '.join(found_skill_names)}`")
    
    st.markdown("#### Section Analysis")
    for skill in required_skills:
        if skill in skill_locations:
            st.success(f"**{skill}**: found in **{skill_locations[skill]}**")
        else:
            st.error(f"**{skill}**: not found")
    
    with st.expander("Show Text Chunks"):
        st.json(chunks)

def validate_tfidf_scores(tfidf_scores_dict, required_sks, all_found_skills, length_factor):
    """Validate TF-IDF scores for required skills"""
    st.markdown(f"#### 📊 TF-IDF Score Analysis (Length Factor: {length_factor})")
    for skill in required_sks:
        score = tfidf_scores_dict.get(skill, 0)
        if skill in all_found_skills:
            st.success(f"**{skill}**: {score:.4f}")
        else:
            st.error(f"**{skill}**: {score:.4f} (Not found in text)")

# --- LOAD ALL RESOURCES ---
nlp = load_nlp_model()
skills_list, ontology_data, skill_to_categories_map = load_all_configs()
matcher = create_skill_matcher(nlp, skills_list)

# --- STREAMLIT UI ---

st.set_page_config(layout="wide")
st.title("✅✅✅ v4.5 (FAIR MODEL) HAS LOADED! ✅✅✅")
st.write("This model uses the new 'Related Skill Bonus' logic.")

st.error("🎯 **DEBUG CONTROLS SECTION - LOOK HERE!**")
debug_mode = st.checkbox("🎯 ENABLE DEBUG MODE (Shows detailed analysis for first 5 resumes)", value=False)
st.session_state.debug_mode = debug_mode
st.markdown("---")

# --- UI: Inputs ---
st.subheader("1. Define Required Skills")
required_skills_raw = st.multiselect(
    "Select the skills you are hiring for:",
    options=skills_list,
    default=["docker", "kubernetes", "aws"]
)
required_skills = [normalize_skill(s) for s in required_skills_raw]

st.subheader("2. Upload Resumes")
uploaded_files = st.file_uploader(
    "Upload candidate resumes (PDF, DOCX, or TXT) in a batch:",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

st.divider()

# --- UI: Processing & Output ---
if st.button("Rank Candidates", type="primary") and uploaded_files:
    if not required_skills:
        st.error("Please select at least one required skill.")
    else:
        results = []
        resume_texts = []
        filenames = []
        
        # --- First Pass: Parse all resumes ---
        with st.spinner("Parsing all resumes..."):
            for file in uploaded_files:
                file_bytes = file.getvalue()
                text = ""
                if file.type == "application/pdf": text = parse_pdf(file_bytes)
                elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document": text = parse_docx(file_bytes)
                elif file.type == "text/plain": text = parse_txt(file_bytes)
                
                if text:
                    resume_texts.append(text)
                    filenames.append(file.name)
                else:
                    results.append({"Candidate (File)": file.name, "Total Score": 0, "Remarks": "Error: Could not read file.", "Direct Score": 0, "Transferable Score": 0, "Direct Details": "N/A", "Transferable Details": "N/A"})
        
        if not resume_texts:
            st.error("Could not parse any of the uploaded files.")
            st.stop()
            
        # --- Second Pass: Fit the TF-IDF Model (v4.2 FIX) ---
        with st.spinner("Fitting the Statistical Model (TF-IDF)..."):
            vectorizer = TfidfVectorizer(
                vocabulary=skills_list, 
                lowercase=True, 
                token_pattern=r'(?u)\b\w+\b', 
                use_idf=True,                  
                smooth_idf=True
            )
            tfidf_matrix = vectorizer.fit_transform(resume_texts)
            tfidf_feature_names = vectorizer.get_feature_names_out()

        # --- Third Pass: Score each resume (v4.5 LOGIC) ---
        with st.spinner(f"Scoring {len(resume_texts)} resumes..."):
            progress_bar = st.progress(0.0)
            tfidf_scores_dense = tfidf_matrix.toarray()
            
            for i, text in enumerate(resume_texts):
                filename = filenames[i]
                
                # 1. Get all data for this resume
                resume_tfidf_scores_array = tfidf_scores_dense[i]
                
                raw_tfidf_scores = {}
                for j, skill in enumerate(tfidf_feature_names):
                    raw_tfidf_scores[skill] = resume_tfidf_scores_array[j]
                
                tfidf_scores_dict, length_factor = normalize_tfidf_by_resume_length(raw_tfidf_scores, text)
                
                chunks = chunk_resume_text(text)
                doc = nlp(text)
                skill_locations = extract_skills_from_chunks(chunks, matcher, nlp)
                all_found_skills = set(skill_locations.keys())
                
                if st.session_state.debug_mode and i < 5:
                    debug_skill_matching(filename, text, required_skills, skill_locations, chunks)
                    validate_tfidf_scores(tfidf_scores_dict, required_skills, all_found_skills, length_factor)

                # 2. Initialize scores
                total_direct_score, total_transfer_score = 0.0, 0.0
                direct_remarks, transfer_remarks = [], []
                
                # 3. Calculate Direct Score
                for req_skill in required_skills:
                    if req_skill in all_found_skills:
                        score, remark = calculate_skill_score(req_skill, skill_locations, tfidf_scores_dict)
                        total_direct_score += score
                        direct_remarks.append(f"{req_skill} (Direct): {remark}")
                
                # 4. Calculate Transferable Score (NEW v4.5 LOGIC)
                target_categories = set()
                for req_skill in required_skills:
                    target_categories.update(skill_to_categories_map.get(req_skill, []))
                
                bonus_skills = all_found_skills - set(required_skills)
                
                total_bonus_score = 0.0
                for bonus_skill in bonus_skills:
                    if any(cat in skill_to_categories_map.get(bonus_skill, []) for cat in target_categories):
                        score, remark = calculate_skill_score(bonus_skill, skill_locations, tfidf_scores_dict)
                        total_bonus_score += score
                        transfer_remarks.append(f"{bonus_skill} [Related]: {remark}")
                
                total_transfer_score = total_bonus_score * TRANSFERABLE_BONUS_WEIGHT
                
                # 5. Final Score Calculation
                final_direct_score = total_direct_score * 100
                final_transfer_score = total_transfer_score * 100
                final_total_score = final_direct_score + final_transfer_score

                results.append({
                    "Candidate (File)": filename,
                    "Total Score": final_total_score,
                    "Direct Score": final_direct_score,
                    "Transferable Score": final_transfer_score,
                    "Remarks": f"Found {len(all_found_skills)} total skills. (LenFactor: {length_factor})",
                    "Direct Details": ", ".join(direct_remarks) if direct_remarks else "No direct skills found.",
                    "Transferable Details": ", ".join(transfer_remarks) if transfer_remarks else "N/A"
                })
                
                progress_bar.progress((i + 1) / len(resume_texts))
        
        # 6. DISPLAY RESULTS
        st.success("Ranking complete!")
        
        df = pd.DataFrame(results)
        df = df.sort_values(by="Total Score", ascending=False)
        
        display_cols = ["Candidate (File)", "Total Score", "Direct Score", "Transferable Score", "Remarks"]
        st.dataframe(df[display_cols].set_index("Candidate (File)"), use_container_width=True)
        
        @st.cache_data
        def convert_df_to_csv(df_to_convert):
            return df_to_convert.to_csv(index=True).encode('utf-8')

        csv_data = convert_df_to_csv(df)
        st.download_button(
            label="⬇️ Download Results as CSV",
            data=csv_data,
            file_name="resume_ranking_results_v4_5_FAIR.csv",
            mime="text/csv",
        )
        
        with st.expander("Show Detailed Scoring Breakdown"):
            st.dataframe(df.set_index("Candidate (File)"), use_container_width=True)

elif not uploaded_files:
    st.info("Please upload resumes to begin ranking.")