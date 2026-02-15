import pandas as pd
import re
from rapidfuzz import process, fuzz
from data_loader import data_loader
import pdfplumber
from datetime import datetime

data_loader.load_all_datasets()

# -----------------------------
# Get all known labs and terms
# -----------------------------
KNOWN_LABS = data_loader.get_all_terms()
KNOWN_ABBREVIATIONS = data_loader.get_all_abbreviations()

# -----------------------------
# Extract lab values from text 
# -----------------------------
def extract_lab_values(text):
    """
    Extract lab values from text with improved detection
    """
    text_lower = text.lower()
    results = []
    
    # Pattern 1: "Hb: 12.0" ou "Hémoglobine = 14"
    pattern1 = r'([a-zà-ÿ]+(?:\s+[a-zà-ÿ]+)*)\s*[:=]?\s*([0-9]+[.,]?[0-9]*)'
    matches1 = re.findall(pattern1, text_lower)
    
    # Pattern 2: "14.0 g/dL" (valeur avant unité)
    pattern2 = r'([0-9]+[.,]?[0-9]*)\s*(?:g/dl|mmol/l|mg/l|μmol/l|u/l|%|ui/l)\s+([a-zà-ÿ]+(?:\s+[a-zà-ÿ]+)*)'
    matches2 = re.findall(pattern2, text_lower)
    
    # Pattern 3: Abréviations "Hb 12"
    for abbrev in KNOWN_ABBREVIATIONS:
        if abbrev and len(abbrev) > 1:
            pattern = r'\b' + re.escape(abbrev) + r'\s*[:=]?\s*([0-9]+[.,]?[0-9]*)'
            matches = re.findall(pattern, text_lower)
            for match in matches:
                try:
                    value = float(match.replace(',', '.'))
                    # Trouver le terme complet pour l'abréviation
                    term_data = data_loader.search_term(abbrev)
                    if term_data:
                        results.append((term_data.get('row_data', {}).get('term', abbrev), value))
                except:
                    continue
    
    all_matches = matches1 + matches2
    
    for match in all_matches:
        if len(match) >= 2:
            lab_name = match[0].strip()
            value_str = match[1].replace(',', '.')
            
            # Vérifier si c'est une valeur numérique
            try:
                value = float(value_str)
                if len(lab_name) > 2:  # Éviter les mots courts
                    # Vérifier si c'est un labo connu
                    if lab_name in KNOWN_LABS or any(abbrev in lab_name for abbrev in KNOWN_ABBREVIATIONS if abbrev):
                        results.append((lab_name, value))
                    else:
                        # Chercher avec fuzzy matching
                        matched = match_lab_name(lab_name)
                        if matched:
                            results.append((matched, value))
            except ValueError:
                continue
    
    # Supprimer les doublons
    unique_results = []
    seen = set()
    for term, value in results:
        key = f"{term}_{value}"
        if key not in seen:
            unique_results.append((term, value))
            seen.add(key)
    
    return unique_results

# -----------------------------
# Fuzzy match lab name 
# -----------------------------
def match_lab_name(user_lab):
    """Fuzzy matching amélioré"""
    if not user_lab or len(user_lab) < 2:
        return None
    
    user_lab_lower = user_lab.lower().strip()
    
    # 1. Chercher dans les termes exacts
    if user_lab_lower in KNOWN_LABS:
        return user_lab_lower
    
    # 2. Chercher dans les abréviations
    if user_lab_lower in KNOWN_ABBREVIATIONS:
        term_data = data_loader.search_term(user_lab_lower)
        if term_data and 'row_data' in term_data:
            return term_data['row_data'].get('term', user_lab_lower)
    
    # 3. Fuzzy matching
    all_terms = KNOWN_LABS + KNOWN_ABBREVIATIONS
    
    # Chercher les meilleurs matches
    matches = process.extract(user_lab_lower, all_terms, scorer=fuzz.partial_ratio, limit=3)
    
    for match, score, _ in matches:
        if score > 75:  # Seuil plus bas pour mieux détecter
            # Si c'est une abréviation, trouver le terme complet
            if match in KNOWN_ABBREVIATIONS:
                term_data = data_loader.search_term(match)
                if term_data and 'row_data' in term_data:
                    return term_data['row_data'].get('term', match)
            return match
    
    return None

# -----------------------------
# Interpret lab value 
# -----------------------------
def interpret_lab(test_name, value):
    """
    Interprète une valeur de laboratoire avec toutes les données
    """
    # 1. Chercher le terme
    term_data = data_loader.search_term(test_name)
    
    if not term_data:
        matched_name = match_lab_name(test_name)
        if matched_name:
            term_data = data_loader.search_term(matched_name)
    
    if not term_data:
        return f"<b>{test_name.title()}</b>: Test non trouvé dans nos bases de données."
    
    # 2. Préparer les informations
    row_data = term_data.get('row_data', {})
    term = row_data.get('term', test_name).title()
    explanation = term_data.get('explanation', '')
    unit = row_data.get('unit', '')
    source = term_data.get('source', 'inconnu')
    
    # 3. Vérifier si c'est un résultat numérique avec plages
    normal_min = row_data.get('normal_min')
    normal_max = row_data.get('normal_max')
    
    try:
        value_float = float(value)
        
        if pd.notna(normal_min) and pd.notna(normal_max):
            min_val = float(normal_min)
            max_val = float(normal_max)
            
            # Déterminer le statut
            if value_float < min_val:
                status = "🔵 BAS"
                color = "#3498db"
                css_class = "low-value"
                interpretation = f"<b>{value} {unit}</b> - <b>Inférieur</b> à la normale ({min_val}-{max_val} {unit})"
            elif value_float > max_val:
                status = "🟠 ÉLEVÉ"
                color = "#e67e22"
                css_class = "high-value"
                interpretation = f"<b>{value} {unit}</b> - <b>Supérieur</b> à la normale ({min_val}-{max_val} {unit})"
            else:
                status = "🟢 NORMAL"
                color = "#27ae60"
                css_class = "normal-value"
                interpretation = f"<b>{value} {unit}</b> - Dans les <b>limites normales</b> ({min_val}-{max_val} {unit})"
            
            # Ajouter l'explication si disponible
            if explanation:
                interpretation += f"<br><small>{explanation}</small>"
            
            # Ajouter la source
            interpretation += f"<br><small style='color: #7f8c8d;'>Source: {source}</small>"
            
            return {
                'html': f"<div class='lab-result {css_class}' style='border-left-color: {color};'>"
                       f"<div style='display: flex; justify-content: space-between; align-items: center;'>"
                       f"<h4 style='margin: 0;'>{term} <span style='color: {color}; font-size: 0.9em;'>{status}</span></h4>"
                       f"<span style='font-weight: bold; font-size: 1.2em;'>{value} {unit}</span>"
                       f"</div>"
                       f"<div style='margin-top: 10px;'>{interpretation}</div>"
                       f"</div>",
                'status': status,
                'color': color,
                'css_class': css_class,
                'value': value,
                'unit': unit,
                'term': term
            }
    
    except (ValueError, TypeError):
        pass
    
    # 4. Si pas de plages ou erreur, retourner juste l'explication
    if explanation:
        return {
            'html': f"<div class='lab-result info-value'>"
                   f"<h4 style='margin: 0;'>{term}</h4>"
                   f"<div style='margin-top: 10px;'>{explanation}</div>"
                   f"<div style='margin-top: 5px;'><small style='color: #7f8c8d;'>Source: {source}</small></div>"
                   f"</div>",
            'status': 'INFO',
            'color': '#7f8c8d',
            'css_class': 'info-value',
            'value': value if isinstance(value, (int, float)) else '',
            'unit': unit,
            'term': term
        }
    
    return {
        'html': f"<div class='lab-result info-value'>"
               f"<h4 style='margin: 0;'>{term}</h4>"
               f"<div style='margin-top: 10px;'>Information disponible dans nos bases de données.</div>"
               f"</div>",
        'status': 'INFO',
        'color': '#7f8c8d',
        'css_class': 'info-value',
        'value': '',
        'unit': '',
        'term': term
    }

# -----------------------------
# Explain labs 
# -----------------------------
def explain_labs(user_input):
    """Analyse les résultats de labo dans le texte"""
    labs = extract_lab_values(user_input)
    
    if not labs:
        # Vérifier si c'est une question sur un terme
        for term in KNOWN_LABS:
            if term.lower() in user_input.lower():
                term_data = data_loader.search_term(term)
                if term_data:
                    explanation = term_data.get('explanation', '')
                    if explanation:
                        return f"<div class='info-box'><h4>{term.title()}</h4><p>{explanation}</p></div>"
        
        return "<div class='warning-box'>Aucun résultat de laboratoire détecté. Essayez: 'Hémoglobine 14' ou 'Qu'est-ce que le cholestérol?'</div>"
    
    responses = []
    for lab_name, value in labs:
        result = interpret_lab(lab_name, value)
        responses.append(result['html'])
    
    return "<br>".join(responses)

# -----------------------------
# Search for medical information 
# -----------------------------
def search_medical_info(query):
    """Recherche des informations médicales générales"""
    # 1. Chercher dans les paires Q/R
    qa_answer = data_loader.search_qa(query)
    if qa_answer:
        return {
            'type': 'qa',
            'content': qa_answer,
            'source': 'qa_dataset'
        }
    
    # 2. Chercher un terme spécifique
    term_data = data_loader.search_term(query)
    if term_data:
        explanation = term_data.get('explanation', '')
        if explanation:
            return {
                'type': 'term',
                'content': explanation,
                'source': term_data.get('source', 'unknown'),
                'term': term_data.get('row_data', {}).get('term', query).title()
            }
    
    return None

# -----------------------------
# Get normal range
# -----------------------------
def get_normal_range(test_name):
    """Obtient les plages normales pour un test"""
    term_data = data_loader.search_term(test_name)
    
    if term_data:
        row_data = term_data.get('row_data', {})
        normal_min = row_data.get('normal_min')
        normal_max = row_data.get('normal_max')
        unit = row_data.get('unit', '')
        
        if pd.notna(normal_min) and pd.notna(normal_max):
            return {
                'min': normal_min,
                'max': normal_max,
                'unit': unit,
                'term': row_data.get('term', test_name).title(),
                'source': term_data.get('source')
            }
    
    return None

# -----------------------------
# Wrapper function for chatbot
# -----------------------------
def explain_input(text):
    """
    Fonction wrapper principale pour le chatbot
    """
    return explain_labs(text)

# -----------------------------
# PDF functions 
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

def preprocess_pdf_text(text):
    """Clean and split PDF text into lines."""
    lines = text.split("\n")
    lines = [line.strip() for line in lines if line.strip()]
    return "\n".join(lines)

# -----------------------------
# PDF Analysis Functions
# -----------------------------

def extract_lab_results_from_pdf(pdf_path):
    """
    Extrait les résultats de laboratoire d'un PDF structuré - Version améliorée
    """
    try:
        print(f"📄 Analyse du PDF: {pdf_path}")
        text = extract_text_from_pdf(pdf_path)
        
        if not text:
            return {"error": "PDF vide ou non lisible"}
        
        print(f"   📝 Texte extrait: {len(text)} caractères")
        
        # 1. Essayer le parsing standard
        standard_results = parse_lab_results(text)
        print(f"   🔍 Résultats standards trouvés: {len(standard_results)}")
        
        # 2. Essayer le parsing de tableau
        table_results = parse_table_lab_results(text)
        print(f"   📊 Résultats tableaux trouvés: {len(table_results)}")
        
        # 3. Combiner les résultats
        all_results = standard_results + table_results
        
        if not all_results:
            # Afficher un extrait pour débogage
            print(f"   🔍 Aperçu du texte (500 premiers caractères):")
            print(text[:500])
            return {"error": "Aucun résultat détecté", "text_preview": text[:500]}
        
        # 4. Interpréter les résultats
        interpretations = []
        for result in all_results:
            interpretation = interpret_lab(result['test'], result['value'])
            if interpretation:
                # Extraire les informations de l'interprétation
                if isinstance(interpretation, dict) and 'html' in interpretation:
                    # C'est une interprétation structurée
                    interpretations.append({
                        'test': result['test'],
                        'value': result['value'],
                        'unit': result.get('unit', ''),
                        'interpretation': interpretation['html'],
                        'reference_range': result.get('reference', ''),
                        'status': interpretation.get('status', 'INFO'),
                        'color': interpretation.get('color', '#7f8c8d')
                    })
                else:
                    # C'est une chaîne simple
                    interpretations.append({
                        'test': result['test'],
                        'value': result['value'],
                        'unit': result.get('unit', ''),
                        'interpretation': str(interpretation),
                        'reference_range': result.get('reference', ''),
                        'status': 'INFO',
                        'color': '#7f8c8d'
                    })
        
        return {
            'success': True,
            'total_results': len(all_results),
            'interpreted_results': len(interpretations),
            'results': interpretations,
            'raw_text_preview': text[:1000] + "..." if len(text) > 1000 else text,
            'source': 'pdf_analysis'
        }
        
    except Exception as e:
        print(f"❌ Erreur analyse PDF: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Erreur d'analyse: {str(e)}"}
    
def preprocess_lab_text(text):
    """
    Nettoie et structure le texte du PDF
    """
    # Supprimer les sauts de ligne multiples
    text = re.sub(r'\n\s*\n+', '\n', text)
    
    # Standardiser les séparateurs
    text = text.replace(':', ' : ')
    text = text.replace('=', ' = ')
    
    # Supprimer les espaces multiples
    text = re.sub(r'\s+', ' ', text)
    
    return text

def parse_lab_results(text):
    """
    Parse le texte pour trouver VRAIMENT les résultats de labo
    """
    results = []
    
    # 1. Chercher spécifiquement les formats de résultats de labo
    patterns = [
        # Format: "Hémoglobine : 14,5 g/dL" ou "Valeur mesurée : 14,5 g/dL"
        r'(?:Hémoglobine|CHOLESTÉROL LDL|CRÉATININE|TSH|Valeur mesurée)\s*[:：]\s*([0-9]+[.,]?[0-9]*)\s*([a-zA-Z/%μ\.]+)',
        
        # Format: "14,5 g/dL" après un nom de test
        r'(?:Hémoglobine|Cholestérol LDL|Créatinine|TSH)[\s\S]{1,50}?([0-9]+[.,]?[0-9]*)\s*([a-zA-Z/%μ\.]+)',
        
        # Format avec unités spécifiques
        r'([0-9]+[.,]?[0-9]*)\s*(g/dL|mmol/L|μmol/L|mUI/L)\b'
    ]
    
    # D'abord, chercher les sections de résultats
    sections = re.split(r'\d+\.\s+[A-ZÀ-ÿ\s]+', text)  # "1. HÉMOGLOBINE", etc.
    
    for section in sections:
        if not section.strip():
            continue
            
        # Chercher dans chaque section
        for pattern in patterns:
            matches = re.finditer(pattern, section, re.IGNORECASE | re.DOTALL)
            for match in matches:
                try:
                    # Grouper différemment selon le pattern
                    if len(match.groups()) >= 2:
                        # Pattern avec nom et unité
                        value_str = match.group(1).replace(',', '.')
                        unit = match.group(2)
                    else:
                        # Pattern simple
                        value_str = match.group(0)
                        unit = ''
                    
                    value = float(value_str)
                    
                    # Filtrer les valeurs improbables pour des tests médicaux
                    if not is_likely_lab_value(value, unit):
                        continue
                    
                    # Trouver le nom du test correspondant
                    test_name = find_test_name(section)
                    if test_name:
                        results.append({
                            'test': test_name,
                            'value': value,
                            'unit': unit,
                            'section': section[:100]  # Pour débogage
                        })
                        
                except (ValueError, AttributeError):
                    continue
    
    # 2. Approche alternative : chercher les paires nom-valeur
    lines = text.split('\n')
    for i, line in enumerate(lines):
        line_lower = line.lower()
        
        # Chercher les noms de tests connus
        test_info = identify_test_from_line(line)
        if test_info:
            test_name = test_info['name']
            
            # Chercher la valeur dans les lignes suivantes
            for j in range(1, 5):  # Regarder jusqu'à 5 lignes après
                if i + j < len(lines):
                    next_line = lines[i + j]
                    value_match = re.search(r'([0-9]+[.,]?[0-9]*)\s*(g/dL|mmol/L|μmol/L|mUI/L)', next_line)
                    if value_match:
                        try:
                            value = float(value_match.group(1).replace(',', '.'))
                            unit = value_match.group(2)
                            
                            if is_likely_lab_value(value, unit):
                                results.append({
                                    'test': test_name,
                                    'value': value,
                                    'unit': unit,
                                    'source': 'line_analysis'
                                })
                            break
                        except ValueError:
                            continue
    
    return results

def is_likely_lab_value(value, unit):
    """
    Filtre pour ne garder que les valeurs plausibles de labo
    """
    # Âges (trop bas ou trop hauts)
    if 0 < value < 2 or (value > 120 and unit in ['ans', 'années']):
        return False
    
    # Dates (mars = 3, mais interprété comme 3.0)
    if value in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] and not unit:
        return False
    
    # Scores /100
    if 0 <= value <= 100 and unit in ['/', '%', 'sur']:
        return False
    
    # Unités valides pour les tests de labo
    valid_units = ['g/dl', 'mmol/l', 'μmol/l', 'mmol/L', 'mUI/L', 'UI/L', 'U/L', '%']
    if unit and unit.lower() not in [u.lower() for u in valid_units]:
        return False
    
    # Plages de valeurs plausibles par test
    if unit.lower() == 'g/dl' and not (5 <= value <= 20):  # Hémoglobine
        return False
    elif unit.lower() == 'mmol/l' and not (1 <= value <= 10):  # Cholestérol
        return False
    elif unit.lower() == 'μmol/l' and not (30 <= value <= 500):  # Créatinine
        return False
    elif unit.lower() == 'mui/l' and not (0.1 <= value <= 20):  # TSH
        return False
    
    return True

def identify_test_from_line(line):
    """
    Identifie le test médical depuis une ligne de texte
    """
    line_lower = line.lower().strip()
    
    test_patterns = [
        ('hémoglobine', ['hémoglobine', 'hb', 'hémog']),
        ('lipoprotéines de basse densité', ['cholestérol ldl', 'ldl', 'lipoprotéines de basse densité']),
        ('créatinine', ['créatinine', 'créat', 'cr']),
        ('hormone thyréostimulante', ['tsh', 'hormone thyréostimulante', 'thyrotropine'])
    ]
    
    for test_name, keywords in test_patterns:
        if any(keyword in line_lower for keyword in keywords):
            return {'name': test_name, 'line': line}
    
    return None

def find_test_name(section):
    """
    Trouve le nom du test dans une section de texte
    """
    section_lower = section.lower()
    
    if 'hémoglobine' in section_lower:
        return 'hémoglobine'
    elif 'cholestérol' in section_lower and 'ldl' in section_lower:
        return 'lipoprotéines de basse densité'
    elif 'créatinine' in section_lower:
        return 'créatinine'
    elif 'tsh' in section_lower or 'hormone thyréostimulante' in section_lower:
        return 'hormone thyréostimulante'
    
    return None

def parse_table_lab_results(text):
    """
    Parse spécifiquement les résultats en format tableau
    """
    results = []
    
    # Diviser en lignes
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Chercher des patterns de tableau
        # Format: "Hémoglobine | 14.5 | g/dL | Normal"
        if '|' in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 2:
                test_name = parts[0]
                value_part = parts[1]
                
                # Extraire la valeur numérique
                value_match = re.search(r'([0-9]+[.,]?[0-9]*)', value_part)
                if value_match:
                    try:
                        value = float(value_match.group(1).replace(',', '.'))
                        test_name = clean_test_name(test_name)
                        
                        # Chercher l'unité
                        unit = ''
                        if len(parts) > 2:
                            unit_part = parts[2]
                            unit_match = re.search(r'([a-zA-Z/%μ\.]+)', unit_part)
                            if unit_match:
                                unit = unit_match.group(1)
                        
                        results.append({
                            'test': test_name,
                            'value': value,
                            'unit': unit,
                            'source': 'tableau'
                        })
                    except ValueError:
                        continue
    
    return results

def clean_test_name(name):
    """
    Nettoie le nom du test pour correspondre à nos données
    """
    # Supprimer les caractères spéciaux
    name = re.sub(r'[^\w\sÀ-ÿ-]', ' ', name)
    
    # Standardiser les noms
    replacements = {
        'HEMOGLOBINE': 'hémoglobine',
        'HB': 'hémoglobine',
        'GLUCOSE': 'glycémie',
        'GAJ': 'glycémie à jeun',
        'CREATININE': 'créatinine',
        'CR': 'créatinine',
        'CHOLESTEROL TOTAL': 'cholestérol total',
        'CT': 'cholestérol total',
        'LDL': 'lipoprotéines de basse densité',
        'HDL': 'lipoprotéines de haute densité',
        'TG': 'triglycérides',
        'TSH': 'hormone thyréostimulante',
        'T4': 't4 libre',
        'FT4': 't4 libre',
        'CRP': 'protéine c-réactive',
        'VS': 'vitesse de sédimentation',
        'ALT': 'alanine aminotransférase',
        'AST': 'aspartate aminotransférase',
        'PAL': 'phosphatase alcaline',
        'NA': 'sodium',
        'K': 'potassium',
        'BUN': 'urée sanguine'
    }
    
    name_upper = name.upper()
    for key, value in replacements.items():
        if key in name_upper:
            return value
    
    return name.lower().strip()

def generate_pdf_report(pdf_analysis):
    """
    Génère un rapport HTML à partir de l'analyse PDF
    """
    if 'error' in pdf_analysis:
        return f"""
        <div class="warning-box">
            <h4>❌ Erreur d'analyse PDF</h4>
            <p>{pdf_analysis['error']}</p>
        </div>
        """
    
    if not pdf_analysis.get('success', False):
        return """
        <div class="warning-box">
            <h4>⚠️ Aucun résultat trouvé</h4>
            <p>Le PDF ne contient pas de résultats de laboratoire identifiables.</p>
        </div>
        """
    
    results = pdf_analysis.get('results', [])
    
    if not results:
        return f"""
        <div style="padding: 20px; background: #f8f9fa; border-radius: 10px;">
            <h4>📄 PDF analysé</h4>
            <p>Total de résultats détectés: {pdf_analysis.get('total_results', 0)}</p>
            <p>Aucun résultat interprétable trouvé.</p>
            <details>
                <summary>Aperçu du texte extrait</summary>
                <pre style="background: white; padding: 10px; border-radius: 5px; max-height: 200px; overflow: auto;">
                {pdf_analysis.get('raw_text_preview', '')}
                </pre>
            </details>
        </div>
        """
    
    # Générer le rapport HTML
    html_parts = []
    
    html_parts.append(f"""
    <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
        <div style="display: flex; align-items: center; margin-bottom: 20px; padding-bottom: 15px; border-bottom: 2px solid #3498db;">
            <div style="background: #3498db; color: white; width: 50px; height: 50px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 15px; font-size: 24px;">
                📊
            </div>
            <div>
                <h2 style="margin: 0; color: #2c3e50;">Rapport d'analyse de laboratoire</h2>
                <p style="margin: 5px 0 0 0; color: #7f8c8d;">
                    {len(results)} résultats analysés • {datetime.now().strftime('%d/%m/%Y %H:%M')}
                </p>
            </div>
        </div>
    """)
    
    # Résultats par catégorie
    categories = {}
    for result in results:
        test_name = result['test']
        category = get_test_category(test_name)
        if category not in categories:
            categories[category] = []
        categories[category].append(result)
    
    # Afficher par catégorie
    for category, category_results in categories.items():
        category_name = category.replace('_', ' ').title()
        category_icon = get_category_icon(category)
        
        html_parts.append(f"""
        <div style="margin: 25px 0;">
            <div style="display: flex; align-items: center; margin-bottom: 15px;">
                <div style="font-size: 20px; margin-right: 10px;">{category_icon}</div>
                <h3 style="margin: 0; color: #2c3e50;">{category_name}</h3>
                <span style="margin-left: 10px; background: #e8f4fc; color: #3498db; padding: 3px 10px; border-radius: 12px; font-size: 14px;">
                    {len(category_results)} résultats
                </span>
            </div>
        """)
        
        for result in category_results:
            test_name = result['test'].title()
            value = result['value']
            unit = result.get('unit', '')
            interpretation_html = result['interpretation']
            
            # Extraire la couleur et le statut de l'interprétation
            status = "normal"
            color = "#27ae60"
            if "élevé" in interpretation_html.lower():
                status = "élevé"
                color = "#e67e22"
            elif "bas" in interpretation_html.lower():
                status = "bas"
                color = "#3498db"
            
            html_parts.append(f"""
            <div style="margin: 12px 0; padding: 15px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid {color};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div style="font-weight: 600; color: #2c3e50; font-size: 16px;">{test_name}</div>
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <div style="font-weight: bold; font-size: 18px; color: {color};">
                            {value} {unit}
                        </div>
                        <span style="background: {color}; color: white; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 500;">
                            {status.upper()}
                        </span>
                    </div>
                </div>
                <div style="color: #34495e; font-size: 14px; line-height: 1.4;">
                    {interpretation_html}
                </div>
                {f"<div style='margin-top: 8px; font-size: 13px; color: #7f8c8d;'>Référence: {result.get('reference', '')}</div>" if result.get('reference') else ''}
            </div>
            """)
        
        html_parts.append("</div>")
    
    # Résumé
    abnormal_count = sum(1 for r in results if "normal" not in r['interpretation'].lower())
    normal_count = len(results) - abnormal_count
    
    html_parts.append(f"""
        <div style="margin-top: 30px; padding: 20px; background: linear-gradient(135deg, #f8f9fa, #e9ecef); border-radius: 10px;">
            <h4 style="margin-top: 0; color: #2c3e50;">📋 Résumé de l'analyse</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin: 15px 0;">
                <div style="text-align: center; padding: 15px; background: white; border-radius: 8px;">
                    <div style="font-size: 24px; font-weight: bold; color: #2c3e50;">{len(results)}</div>
                    <div style="color: #7f8c8d;">Total résultats</div>
                </div>
                <div style="text-align: center; padding: 15px; background: white; border-radius: 8px;">
                    <div style="font-size: 24px; font-weight: bold; color: #27ae60;">{normal_count}</div>
                    <div style="color: #7f8c8d;">Résultats normaux</div>
                </div>
                <div style="text-align: center; padding: 15px; background: white; border-radius: 8px;">
                    <div style="font-size: 24px; font-weight: bold; color: #e67e22;">{abnormal_count}</div>
                    <div style="color: #7f8c8d;">Résultats anormaux</div>
                </div>
            </div>
            <div style="margin-top: 20px; padding: 15px; background: #fff3cd; border-radius: 8px; border-left: 4px solid #ffc107;">
                <strong>⚠️ Avertissement important:</strong><br>
                Cette analyse est automatique et ne remplace pas l'interprétation d'un médecin.
                Consultez toujours un professionnel de santé pour l'interprétation complète de vos résultats.
            </div>
        </div>
    </div>
    """)
    
    return "".join(html_parts)

def get_test_category(test_name):
    """Détermine la catégorie d'un test"""
    categories = {
        'hématologie': ['hémoglobine', 'globule', 'plaquette', 'hématocrite'],
        'biochimie': ['créatinine', 'glycémie', 'cholestérol', 'triglycéride', 'urée', 'sodium', 'potassium'],
        'endocrinologie': ['tsh', 't4', 'ft4', 'hormone'],
        'inflammatoire': ['crp', 'vs', 'protéine c-réactive'],
        'hépatique': ['alt', 'ast', 'pal', 'transaminase', 'phosphatase'],
        'rénal': ['créatinine', 'urée', 'clairance']
    }
    
    test_lower = test_name.lower()
    for category, keywords in categories.items():
        if any(keyword in test_lower for keyword in keywords):
            return category
    
    return 'autres'

def get_category_icon(category):
    """Retourne l'icône pour une catégorie"""
    icons = {
        'hématologie': '🩸',
        'biochimie': '🧪',
        'endocrinologie': '🦋',
        'inflammatoire': '🔥',
        'hépatique': '🧬',
        'rénal': '💧',
        'autres': '📋'
    }
    return icons.get(category, '📋')

# -----------------------------
# Main test
# -----------------------------
if __name__ == "__main__":
    # Tests
    test_cases = [
        "Hémoglobine: 14.2 g/dL",
        "Hb 12.5",
        "Qu'est-ce que l'hémoglobine?",
        "Cholestérol 5.8",
        "LDL 2.5 mmol/L"
    ]
    
    for test in test_cases:
        print(f"\n{'='*50}")
        print(f"Test: {test}")
        print(f"{'='*50}")
        print(explain_input(test))
