import subprocess

# Full path to ollama.exe
OLLAMA_PATH = r"C:\Users\Admin\AppData\Local\Programs\Ollama\ollama.exe"

# Example prompt
prompt = "Bonjour, explique simplement l'hypertension."

try:
    result = subprocess.run(
        [OLLAMA_PATH, "run", "llama2", "--prompt", prompt],
        capture_output=True,
        text=True
    )
    print("Ollama output:\n")
    print(result.stdout.strip())
except Exception as e:
    print(f"Erreur lors de l'exécution d'Ollama: {str(e)}")
