# test_app.py
import subprocess
import gradio as gr
import pandas as pd
import re

print("Test de l'application médicale...")

# Test du dataset
try:
    df = pd.read_csv("datasets/comprehensive_medical_dataset.csv")
    print(f"✓ Dataset chargé: {len(df)} entrées")
except:
    print("✗ Dataset non trouvé")
    # Créer un dataset de test
    test_data = {
        'term': ['hémoglobine', 'glucose'],
        'abbreviation': ['hb', 'glu'],
        'normal_min': [12, 4],
        'normal_max': [16, 6],
        'unit': ['g/dl', 'mmol/l'],
        'explanation': ['Transporte l\'oxygène', 'Taux de sucre']
    }
    df = pd.DataFrame(test_data)
    print("✓ Dataset de test créé")

# Test d'Ollama
print("\nTest d'Ollama avec Phi-2.7b...")
try:
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True
    )
    if "phi:2.7b" in result.stdout:
        print("✓ Phi-2.7b est installé")
    else:
        print("⚠️ Phi-2.7b non trouvé, installez-le avec: ollama pull phi:2.7b")
except:
    print("✗ Ollama n'est pas accessible")

# Interface de test
def test_interface(text):
    return f"<h3>Test réussi!</h3><p>Vous avez écrit: {text}</p>"

if __name__ == "__main__":
    print("\nLancement de l'interface de test...")
    gr.Interface(
        fn=test_interface,
        inputs="text",
        outputs="html",
        title="Test Chatbot Médical"
    ).launch()