@echo off
echo Installation des dependances...
pip install gradio pandas rapidfuzz scikit-learn wikipedia pdfplumber Pillow pytesseract opencv-python-headless
pip install pdf2image camelot-py[cv]

echo.
echo Verification d'Ollama...
where ollama >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ Ollama est installe
    ollama list
) else (
    echo ✗ Ollama n'est pas installe
    echo Telechargez-le depuis: https://ollama.com/
)

echo.
echo Installation terminee!
echo.
echo Pour lancer l'application: python app.py
pause