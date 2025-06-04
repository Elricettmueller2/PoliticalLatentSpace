# Political Latent Space Visualizer - Projektplanung

## Projektübersicht
Dieses Uni-Projekt zielt darauf ab, eine interaktive Installation zu entwickeln, die politische Bewegungen in einem semantischen Raum visualisiert. Wir nutzen den vorhandenen Prototyp `enhanced_political_analyzer.py` als Grundlage und entwickeln daraus eine webbasierte Visualisierung für eine einmalige Präsentation.

## Teamstruktur (3 Personen)

### Person 1: Datenaufbereitung & NLP
- Verantwortlich für die Sammlung und Aufbereitung politischer Texte
- Anpassung des `HybridPoliticalSpace`-Modells für die Kontextanalyse
- Generierung der finalen Daten für die Visualisierung

### Person 2: Visualisierung & Frontend
- Entwicklung der interaktiven Visualisierung mit D3.js oder p5.js
- Gestaltung der Benutzeroberfläche
- Integration der vorverarbeiteten Daten

### Person 3: Projektkoordination & Dokumentation
- Koordination des Teams und Zeitplanung
- Erstellung der Präsentationsmaterialien
- Unterstützung bei Datensammlung und Visualisierung

## Projektphasen

### Phase 1: Vorbereitung (Woche 1)
1. **Projektsetup**
   - Repository einrichten
   - Vorhandenen Code verstehen und dokumentieren
   - Arbeitsumgebung für alle Teammitglieder einrichten

2. **Datensammlung**
   - Identifizierung von 8-12 relevanten politischen Bewegungen/Parteien
   - Sammlung repräsentativer Texte (Manifeste, Positionspapiere, etc.)
   - Strukturierung der Daten in einem einheitlichen Format

### Phase 2: Datenverarbeitung (Woche 2)
1. **Erweiterung des NLP-Modells**
   - Integration der Kontextfenster-Analyse für Schlüsselwörter
   - Optimierung der Embedding-Generierung
   - Test mit verschiedenen politischen Texten

2. **Datenaufbereitung für Visualisierung**
   - Berechnung der Positionen im semantischen Raum
   - Extraktion relevanter Schlüsselwörter und deren Kontexte
   - Generierung eines JSON-Datensatzes für die Visualisierung

### Phase 3: Visualisierungsentwicklung (Woche 3-4)
1. **Basisvisualisierung**
   - Entwicklung einer statischen HTML/JS-Seite
   - Implementation der D3.js/p5.js Visualisierung
   - Darstellung der politischen Bewegungen im semantischen Raum

2. **Interaktive Elemente**
   - Zoom- und Pan-Funktionalität
   - Hover-Effekte mit Detailinformationen
   - Filteroptionen für verschiedene Dimensionen

3. **Kontextanalyse-Visualisierung**
   - Visuelle Darstellung der unterschiedlichen Verwendung gleicher Begriffe
   - Interaktive Exploration von Schlüsselwörtern im Kontext
   - Farbkodierung für semantische Unterschiede

### Phase 4: Finalisierung (Woche 5)
1. **Integration und Testing**
   - Zusammenführung aller Komponenten
   - Optimierung der Performance
   - Umfassendes Testing auf verschiedenen Geräten

2. **Dokumentation und Präsentation**
   - Erstellung einer Projektdokumentation
   - Vorbereitung der Präsentationsmaterialien
   - Planung der Installation

## Technischer Ansatz

### Datenverarbeitung
Wir nutzen den vorhandenen `HybridPoliticalSpace`-Prototyp und erweitern ihn um die Kontextfenster-Analyse:

```python
# Erweiterung für enhanced_political_analyzer.py
def analyze_term_in_context(self, text, term, window_size=30):
    """Analyze how a specific term is used in context"""
    import re
    
    # Finde alle Vorkommen des Begriffs (case-insensitive)
    pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
    matches = list(pattern.finditer(text))
    
    contexts = []
    for match in matches:
        start = max(0, match.start() - window_size)
        end = min(len(text), match.end() + window_size)
        context = text[start:end]
        contexts.append({
            "text": context,
            "position": match.start()
        })
    
    # Analysiere jeden Kontext
    results = []
    if contexts:
        for context_data in contexts:
            context = context_data["text"]
            embedding = self.model.encode(context)
            
            similarities = {dim: cosine_similarity([embedding], [anchor_emb])[0][0] 
                          for dim, anchor_emb in self.anchor_embeddings.items()}
            
            # Berechne Positionierung für diesen Kontext
            position = {}
            position["economic_axis"] = similarities["economic_left"] - similarities["economic_right"]
            position["social_axis"] = similarities["progressive"] - similarities["conservative"]
            position["ecological_axis"] = similarities["ecological"] - similarities["growth"]
            position["governance_axis"] = similarities["internationalist"] - similarities["nationalist"]
            
            results.append({
                "context": context,
                "position": position,
                "similarities": similarities
            })
    
    return results
```

### Datenexport für Visualisierung
Wir erstellen ein Script, das die Analyseergebnisse in ein JSON-Format für die Visualisierung exportiert:

```python
# export_visualization_data.py
import json
from enhanced_political_analyzer import HybridPoliticalSpace

def export_data():
    analyzer = HybridPoliticalSpace()
    
    # Politische Bewegungen und ihre Texte
    movements_data = {
        "Partei Alpha (Grün-Progressiv)": "...",
        "Partei Beta (Wirtschaftsliberal-Konservativ)": "...",
        # weitere Bewegungen...
    }
    
    # Schlüsselwörter für Kontextanalyse
    keywords = ["frieden", "sicherheit", "freiheit", "gerechtigkeit"]
    
    results = []
    for name, text in movements_data.items():
        # Grundpositionierung
        position_data = analyzer.position_movement(name, text)
        
        # Kontextanalyse für Schlüsselwörter
        keyword_contexts = {}
        for keyword in keywords:
            contexts = analyzer.analyze_term_in_context(text, keyword)
            if contexts:
                keyword_contexts[keyword] = contexts
        
        # Ergebnisse zusammenführen
        position_data["keyword_contexts"] = keyword_contexts
        results.append(position_data)
    
    # Als JSON speichern
    with open("web_visualization/data/political_data.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("Daten erfolgreich exportiert!")

if __name__ == "__main__":
    export_data()
```

### Webvisualisierung
Wir erstellen eine einfache Webseite mit D3.js für die Visualisierung:

```
web_visualization/
├── index.html           # Hauptseite
├── css/
│   └── style.css        # Styling
├── js/
│   ├── main.js          # Hauptlogik
│   └── visualization.js # D3.js Visualisierung
└── data/
    └── political_data.json # Exportierte Daten
```

## Projektstruktur

```
political-latent-space/
├── prototypes/
│   └── enhanced_political_analyzer.py  # Vorhandener Prototyp
├── data_processing/
│   ├── collect_data.py                 # Datensammlung
│   ├── process_texts.py                # Textverarbeitung
│   └── export_visualization_data.py    # Datenexport
├── web_visualization/                  # Webbasierte Visualisierung
│   ├── index.html                      # Hauptseite
│   ├── css/                            # Styling
│   ├── js/                             # JavaScript
│   └── data/                           # Exportierte Daten
├── data/
│   ├── raw/                            # Rohtexte
│   └── processed/                      # Verarbeitete Daten
└── docs/                               # Dokumentation
```

## Zeitplan

| Woche | Hauptaufgaben | Verantwortlich |
|-------|--------------|----------------|
| 1     | Projektsetup, Datensammlung | Alle |
| 2     | NLP-Modell erweitern, Datenverarbeitung | Person 1 |
| 3     | Basisvisualisierung entwickeln | Person 2 |
| 4     | Interaktive Elemente, Kontextanalyse-Visualisierung | Person 2, Person 1 |
| 5     | Integration, Testing, Dokumentation | Alle |

## Benötigte Ressourcen

### Software
- Python 3.8+ mit folgenden Bibliotheken:
  - sentence-transformers
  - numpy
  - pandas
  - scikit-learn
  - plotly (für Prototyping)
- Web-Technologien:
  - HTML5/CSS3
  - JavaScript
  - D3.js oder p5.js

### Hardware
- Für die Präsentation: Computer mit Webbrowser
- Für die Entwicklung: Ausreichend RAM für NLP-Verarbeitung (min. 8GB)

## Nächste Schritte

1. **Sofort**
   - Repository einrichten
   - Vorhandenen Code verstehen und dokumentieren
   - Liste politischer Bewegungen erstellen

2. **Diese Woche**
   - Datensammlung beginnen
   - Kontextanalyse-Funktion implementieren
   - Erste Skizzen der Visualisierung erstellen

3. **Nächste Woche**
   - Erste Datenverarbeitung durchführen
   - Basis-HTML-Struktur erstellen
   - D3.js/p5.js einbinden
