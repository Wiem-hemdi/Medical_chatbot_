import pandas as pd
import re
from rapidfuzz import process
import pdfplumber

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("datasets/comprehensive_medical_dataset.csv")
df['term'] = df['term'].str.lower()
df['abbreviation'] = df['abbreviation'].fillna("").str.lower()

# Known labs for fuzzy matching
KNOWN_LABS = [lab for lab in df['term'].tolist() + df['abbreviation'].tolist() if lab]

# -----------------------------
# Extract lab values from text
# -----------------------------
def extract_lab_values(text):
    """
    Extract lab values from text, e.g., 'My Hb is 11.0 and HbA1c 6.2'.
    Returns list of (lab_name, value)
    """
    text = text.lower()
    labs_pattern = "|".join([re.escape(name) for name in KNOWN_LABS])
    pattern = rf"({labs_pattern})\s*(?:is|=|:)?\s*([0-9]+\.?[0-9]*)"
    matches = re.findall(pattern, text)
    
    # Also look for patterns like "Glucose: 6.5 mmol/L"
    pattern2 = r"([a-zà-ÿ\s]+)\s*[:=]\s*([0-9]+\.?[0-9]*)\s*(?:mg/dl|g/l|mmol/l|ui/l|%|μmol/l)?"
    matches2 = re.findall(pattern2, text)
    
    all_matches = matches + matches2
    
    results = []
    for match in all_matches:
        lab_name = match[0].strip()
        try:
            value = float(match[1])
            # Vérifier si c'est un nom de labo connu ou similaire
            if len(lab_name) > 2:  # Éviter les mots trop courts
                results.append((lab_name, value))
        except ValueError:
            continue
    
    return list(set(results))  # Supprimer les doublons

# -----------------------------
# Fuzzy match lab name
# -----------------------------
def match_lab_name(user_lab):
    if not user_lab or len(user_lab) < 2:
        return None
    
    match_data = process.extractOne(user_lab, KNOWN_LABS)
    if match_data is None:
        return None
    match, score, _ = match_data
    if score > 60:
        return match
    return None

# -----------------------------
# Interpret lab value
# -----------------------------
def interpret_lab(test_name, value):
    # Chercher d'abord le terme exact
    row = df[(df['term'] == test_name) | (df['abbreviation'] == test_name)]
    
    # Si pas trouvé, chercher avec fuzzy matching
    if row.empty:
        matched_name = match_lab_name(test_name)
        if matched_name:
            row = df[(df['term'] == matched_name) | (df['abbreviation'] == matched_name)]
    
    if row.empty:
        return f"<b>{test_name}</b>: Test non trouvé dans le dataset."
    
    row = row.iloc[0]
    
    if pd.notna(row['normal_min']) and pd.notna(row['normal_max']):
        try:
            normal_min = float(row['normal_min'])
            normal_max = float(row['normal_max'])
            
            if value < normal_min:
                color = "blue"
                msg = f"{row['term'].title()} est <b>bas</b> ({value} {row['unit']}). Normal: {normal_min}–{normal_max} {row['unit']}."
            elif value > normal_max:
                color = "orange"
                msg = f"{row['term'].title()} est <b>élevé</b> ({value} {row['unit']}). Normal: {normal_min}–{normal_max} {row['unit']}."
            else:
                color = "green"
                msg = f"{row['term'].title()} est <b>normal</b> ({value} {row['unit']})."
            
            return f"<span style='color:{color}'>{msg}</span>"
        except (ValueError, TypeError):
            # Si les valeurs ne peuvent pas être converties en float
            return f"<span style='color:gray'>{row['explanation']}</span>"
    else:
        # Si pas de plage normale, afficher l'explication
        return f"<span style='color:gray'>{row['explanation']}</span>"

# -----------------------------
# Explain labs
# -----------------------------
def explain_labs(user_input):
    labs = extract_lab_values(user_input)
    responses = []
    
    for lab_name, value in labs:
        interpretation = interpret_lab(lab_name, value)
        responses.append(f"<div class='lab-item'>{interpretation}</div>")
    
    if not responses:
        return "Aucun résultat de laboratoire détecté dans votre texte."
    
    return "<br>".join(responses)

# -----------------------------
# Explain input (wrapper for chatbot)
# -----------------------------
def explain_input(text, generate_patient_text_func=None):
    """
    Explain lab results in patient-friendly HTML.
    This is the function your chatbot should call.
    """
    return explain_labs(text)

# -----------------------------
# Match exact term from dataset
# -----------------------------
def match_term(query):
    query = query.lower().strip()
    matches = df[df['term'].str.lower() == query]
    if not matches.empty:
        return matches.iloc[0]['explanation']
    else:
        return None

# -----------------------------
# Extract text from PDF
# -----------------------------
def extract_text_from_pdf(pdf_path):
    """Extract text from PDF lab report."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Erreur lors de l'extraction PDF: {e}")
        return ""
    return text

# -----------------------------
# Preprocess PDF text
# -----------------------------
def preprocess_pdf_text(text):
    """Clean and split PDF text into lines."""
    lines = text.split("\n")
    lines = [line.strip() for line in lines if line.strip()]
    return "\n".join(lines)

# -----------------------------
# Get normal range for a test
# -----------------------------
def get_normal_range(test_name):
    """Get normal range for a specific test"""
    test_name = test_name.lower().strip()
    
    # Chercher le terme exact
    row = df[(df['term'] == test_name) | (df['abbreviation'] == test_name)]
    
    if row.empty:
        matched_name = match_lab_name(test_name)
        if matched_name:
            row = df[(df['term'] == matched_name) | (df['abbreviation'] == matched_name)]
    
    if not row.empty:
        row = row.iloc[0]
        if pd.notna(row['normal_min']) and pd.notna(row['normal_max']):
            return {
                'min': row['normal_min'],
                'max': row['normal_max'],
                'unit': row['unit'] if pd.notna(row['unit']) else '',
                'term': row['term']
            }
    
    return None

# -----------------------------
# Main test
# -----------------------------
if __name__ == "__main__":
    # Test
    test_text = "Hémoglobine: 14.2 g/dL, Glucose: 5.6 mmol/L, HbA1c 6.2%"
    print("Test d'extraction:")
    print(extract_lab_values(test_text))
    print("\nExplication:")
    print(explain_input(test_text))