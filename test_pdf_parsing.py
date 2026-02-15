import re

pdf_content = """Rapport d'Analyse de Résultats de Laboratoire
Rapport généré le : 12 mars 2024
Patient : M. Jean Dupont
Âge : 45 ans

📊 1. RÉSUMÉ EXÉCUTIF
Métrique
Valeur
Statut
Priorité
Hémoglobine
14.5 g/dL
✅ Normal
Priorité 1 - Labo
Cholestérol LDL
3.2 mmol/L
⚠️ Limite supérieure
Priorité 1 - Labo
Créatinine
85 μmol/L
✅ Normal
Priorité 1 - Labo
TSH
2.5 mUI/L
✅ Normal
Priorité 1 - Labo

🧪 2. RÉSULTATS DÉTAILLÉS PAR PRIORITÉ
PRIORITÉ 1 : ANALYSE LABORATOIRE 🧪
Paramètre
Valeur
Référence normale
Interprétation
Classification
Hémoglobine
14.5 g/dL
13.5-17.5 g/dL
Valeur normale
Normal
Cholestérol LDL
3.2 mmol/L
< 3.4 mmol/L
Limite supérieure normale
Surveillance
Créatinine sérique
85 μmol/L
60-110 μmol/L
Fonction rénale normale
Normal
TSH
2.5 mUI/L
0.4-4.0 mUI/L
Fonction thyroïdienne normale
Normal"""

def parse_pdf_with_line_analysis(text):
    """Parse spécialement pour les PDF avec valeurs sur lignes séparées"""
    print("🔍 Analyse ligne par ligne...")
    
    # Diviser en lignes
    lines = text.split('\n')
    
    results = []
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Chercher un nom de test médical
        if is_test_name(line):
            test_name = line
            
            # Regarder les 3 lignes suivantes pour trouver la valeur
            for j in range(1, 4):
                if i + j < len(lines):
                    next_line = lines[i + j].strip()
                    
                    # Chercher une valeur numérique avec unité
                    value_match = re.search(r'([0-9]+[.,]?[0-9]*)\s*([a-zA-Z/%μ\.]+)', next_line)
                    if value_match:
                        try:
                            value = float(value_match.group(1).replace(',', '.'))
                            unit = value_match.group(2)
                            
                            results.append({
                                'test': clean_test_name(test_name),
                                'value': value,
                                'unit': unit,
                                'line': i
                            })
                            
                            print(f"✓ Ligne {i}: {test_name} → {value} {unit}")
                            break
                            
                        except ValueError:
                            continue
            i += 1
        else:
            i += 1
    
    return results

def is_test_name(text):
    """Vérifie si le texte ressemble à un nom de test"""
    test_keywords = [
        'hémoglobine', 'cholestérol', 'créatinine', 'tsh', 'hdl', 'ldl',
        'glucose', 'glycémie', 'sodium', 'potassium', 'urée', 'crp'
    ]
    
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in test_keywords)

def clean_test_name(name):
    """Nettoie le nom du test"""
    replacements = {
        'HEMOGLOBINE': 'hémoglobine',
        'CHOLESTÉROL LDL': 'lipoprotéines de basse densité',
        'CRÉATININE': 'créatinine',
        'CRÉATININE SÉRIQUE': 'créatinine',
        'TSH': 'hormone thyréostimulante'
    }
    
    name_upper = name.upper()
    for key, value in replacements.items():
        if key in name_upper:
            return value
    
    return name.lower().strip()

# Exécuter le test
print("="*60)
print("TEST DE PARSING PDF - Version corrigée")
print("="*60)

results = parse_pdf_with_line_analysis(pdf_content)
print(f"\n✅ Total résultats trouvés: {len(results)}")

# Afficher les résultats
for i, result in enumerate(results, 1):
    print(f"{i}. {result['test']}: {result['value']} {result['unit']}")