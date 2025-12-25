import pandas as pd
import subprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("datasets/comprehensive_medical_dataset.csv")
df['term'] = df['term'].str.lower()
df['abbreviation'] = df['abbreviation'].fillna("").str.lower()
df['all_names'] = df.apply(
    lambda row: f"{row['term']}; {row['abbreviation']}" if row['abbreviation'] else row['term'],
    axis=1
)

# -----------------------------
# Build RAG retrieval
# -----------------------------
corpus = df['all_names'] + ". " + df['explanation']
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

def retrieve_info(query, top_k=2):
    query = query.lower().strip()
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, X)
    top_indices = sims[0].argsort()[-top_k:][::-1]
    return "\n".join([corpus[i] for i in top_indices])

# -----------------------------
# Ollama integration
# -----------------------------
OLLAMA_PATH = r"C:\Users\Admin\AppData\Local\Programs\Ollama\ollama.exe"

def generate_patient_text(user_input):
    context = retrieve_info(user_input)
    prompt = f"""[ROLE] Assistant médical pour patients
    [CONTEXT] {context}
    [QUESTION] {user_input}
    [INSTRUCTION] Explique de manière simple et claire en français.
    [RÉPONSE]:"""
    try:
        result = subprocess.run(
            [OLLAMA_PATH, "run", "phi:2.7b", "--prompt", prompt],
            capture_output=True,
            text=True,
            timeout=10
        )
        output = result.stdout.strip()
        if output:
            return output
        else:
            row = df[(df['term'] == user_input.lower()) | (df['abbreviation'] == user_input.lower())]
            if not row.empty:
                return row.iloc[0]['explanation']
            return f"Pas de réponse générée pour: {user_input}"
    except Exception as e:
        return f"Erreur Ollama: {str(e)}"
