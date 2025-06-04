#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Political Space Implementation
------------------------------------
A test implementation of the hybrid positioning system for political movements
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import umap

class HybridPoliticalSpace:
    def __init__(self):
        print("Initializing Hybrid Political Space...")
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
        
        # Expert-defined anchor points for key political dimensions
        self.anchors = {
            "economic_left": "Wir setzen uns für staatliche Regulierung, Umverteilung und soziale Absicherung ein. Der Staat muss aktiv in die Wirtschaft eingreifen, um Ungleichheit zu bekämpfen und öffentliche Dienstleistungen zu garantieren.",
            "economic_right": "Wir fördern freie Märkte, niedrige Steuern und unternehmerische Freiheit. Die Wirtschaft funktioniert am besten mit minimaler staatlicher Einmischung und maximaler individueller wirtschaftlicher Freiheit.",
            "progressive": "Wir stehen für gesellschaftlichen Wandel, Diversität und neue Lebensformen. Gesellschaftliche Normen müssen sich weiterentwickeln, um Vielfalt und Inklusion zu fördern.",
            "conservative": "Wir bewahren Traditionen, Familie und kulturelle Identität. Bewährte Werte und Institutionen müssen geschützt werden, um gesellschaftliche Stabilität zu gewährleisten.",
            "ecological": "Klimaschutz und Umweltschutz haben für uns höchste Priorität. Wir müssen unsere Wirtschaft und Lebensweise grundlegend umgestalten, um die ökologische Krise zu bewältigen.",
            "growth": "Wirtschaftswachstum und Wohlstand sind entscheidend. Umweltschutz darf nicht zu Lasten von Arbeitsplätzen und wirtschaftlicher Entwicklung gehen.",
            "nationalist": "Nationale Souveränität und Identität müssen geschützt werden. Die Interessen unseres Landes haben Vorrang vor internationalen Verpflichtungen.",
            "internationalist": "Internationale Zusammenarbeit und globale Lösungen sind entscheidend. Nur durch gemeinsames Handeln können wir die großen Herausforderungen unserer Zeit bewältigen."
        }
        
        print("Generating anchor embeddings...")
        # Generate embeddings for anchors
        self.anchor_embeddings = {k: self.model.encode(v) for k, v in self.anchors.items()}
        
    def position_movement(self, name, text):
        """Position a political movement in the hybrid space"""
        # Get embedding for movement text
        embedding = self.model.encode(text)
        
        # Calculate similarity to each anchor
        similarities = {dim: cosine_similarity([embedding], [anchor_emb])[0][0] 
                       for dim, anchor_emb in self.anchor_embeddings.items()}
        
        # Calculate position on key axes
        economic_axis = similarities["economic_right"] - similarities["economic_left"]
        social_axis = similarities["progressive"] - similarities["conservative"]
        ecological_axis = similarities["ecological"] - similarities["growth"]
        governance_axis = similarities["nationalist"] - similarities["internationalist"]
        
        return {
            "name": name,
            "economic_axis": economic_axis,
            "social_axis": social_axis,
            "ecological_axis": ecological_axis,
            "governance_axis": governance_axis,
            "raw_embedding": embedding,
            "anchor_similarities": similarities
        }

    def extract_key_terms(self, text, top_n=10):
        """Extract key terms from text using TF-IDF"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Simple German stopwords
        german_stopwords = [
            "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer", "eines", 
            "einem", "einen", "und", "oder", "aber", "wenn", "dann", "als", "wie", "wo", 
            "wer", "was", "warum", "wieso", "weshalb", "welche", "welcher", "welches", 
            "für", "von", "mit", "zu", "zur", "zum", "auf", "in", "im", "bei", "an", 
            "am", "um", "durch", "über", "unter", "gegen", "nach", "vor", "ist", "sind", 
            "war", "waren", "wird", "werden", "wurde", "wurden", "hat", "haben", "hatte", 
            "hatten", "kann", "können", "darf", "dürfen", "muss", "müssen", "soll", 
            "sollen", "will", "wollen", "mag", "mögen", "dass", "daß", "weil", "obwohl", 
            "damit", "dafür", "dabei", "dazu", "daran", "darauf", "darunter", "darüber"
        ]
        
        # Create TF-IDF vectorizer
        tfidf = TfidfVectorizer(max_features=100, stop_words=german_stopwords)
        tfidf_matrix = tfidf.fit_transform([text])
        feature_names = np.array(tfidf.get_feature_names_out())
        
        # Get top terms
        tfidf_scores = tfidf_matrix.toarray()[0]
        sorted_indices = np.argsort(tfidf_scores)[::-1]
        top_terms = [(feature_names[idx], tfidf_scores[idx]) 
                    for idx in sorted_indices[:top_n] if tfidf_scores[idx] > 0]
        
        return top_terms

# Sample political movement texts (simplified for testing)
sample_movements = {
    "CDU": """
    Die CDU steht für eine soziale Marktwirtschaft, die auf Eigenverantwortung und Wettbewerb setzt. 
    Wir fördern Familien als Kern unserer Gesellschaft und setzen uns für innere Sicherheit ein. 
    Die CDU steht für eine starke europäische Integration bei gleichzeitiger Bewahrung nationaler Identität. 
    Wir setzen auf wirtschaftliches Wachstum und Klimaschutz durch Innovation und neue Technologien.
    """,
    
    "SPD": """
    Die SPD kämpft für soziale Gerechtigkeit und einen starken Sozialstaat. 
    Wir setzen uns für faire Löhne, bezahlbaren Wohnraum und gute Arbeitsbedingungen ein. 
    Die SPD steht für eine solidarische Gesellschaft, in der niemand zurückgelassen wird. 
    Wir fördern die europäische Integration und internationale Zusammenarbeit.
    """,
    
    "Die Grünen": """
    Die Grünen setzen sich für konsequenten Klimaschutz und ökologische Transformation ein. 
    Wir kämpfen für eine nachhaltige Wirtschaft, die Ressourcen schont und die Umwelt schützt. 
    Die Grünen stehen für eine offene, vielfältige Gesellschaft und die Stärkung von Bürgerrechten. 
    Wir fördern erneuerbare Energien und den Ausstieg aus fossilen Brennstoffen.
    """,
    
    "Die Linke": """
    Die Linke kämpft gegen soziale Ungleichheit und für eine gerechte Verteilung des Reichtums. 
    Wir setzen uns für einen starken Sozialstaat, höhere Löhne und die Überwindung des Kapitalismus ein. 
    Die Linke steht für Frieden, Abrüstung und internationale Solidarität. 
    Wir fordern bezahlbaren Wohnraum und den Ausbau öffentlicher Dienstleistungen.
    """,
    
    "AfD": """
    Die AfD setzt sich für die Bewahrung deutscher Kultur und Identität ein. 
    Wir fordern eine Begrenzung der Zuwanderung und eine Rückkehr zu nationaler Souveränität. 
    Die AfD steht für traditionelle Familienwerte und lehnt Gender-Ideologie ab. 
    Wir sind kritisch gegenüber dem Euro und fordern mehr direkte Demokratie.
    """,
    
    "FDP": """
    Die FDP steht für wirtschaftliche Freiheit, Eigenverantwortung und einen schlanken Staat. 
    Wir setzen uns für Steuersenkungen, Bürokratieabbau und die Förderung von Unternehmertum ein. 
    Die FDP steht für Bürgerrechte, Digitalisierung und technologischen Fortschritt. 
    Wir fördern Bildung als Schlüssel zu individueller Freiheit und Wohlstand.
    """,
    
    "Fridays for Future": """
    Fridays for Future fordert sofortiges Handeln gegen die Klimakrise und die Einhaltung des 1,5-Grad-Ziels. 
    Wir setzen uns für einen schnellen Kohleausstieg und 100% erneuerbare Energien ein. 
    Fridays for Future steht für Klimagerechtigkeit und eine lebenswerte Zukunft für alle Generationen. 
    Wir fordern von der Politik, auf die Wissenschaft zu hören und entsprechend zu handeln.
    """,
    
    "Extinction Rebellion": """
    Extinction Rebellion kämpft gegen das Massenaussterben und den ökologischen Kollaps. 
    Wir fordern die Ausrufung des Klimanotstands und Netto-Null-Emissionen bis 2025. 
    Extinction Rebellion setzt auf zivilen Ungehorsam als Mittel des politischen Protests. 
    Wir stehen für Bürgerversammlungen und radikale Demokratisierung der Klimapolitik.
    """
}

def main():
    # Initialize the hybrid political space
    political_space = HybridPoliticalSpace()
    
    # Position each movement
    positions = []
    for name, text in sample_movements.items():
        print(f"Positioning {name}...")
        position = political_space.position_movement(name, text)
        
        # Extract key terms
        key_terms = political_space.extract_key_terms(text)
        position["key_terms"] = key_terms
        
        positions.append(position)
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(positions)
    
    # Create a 2D visualization of economic vs. social axes
    create_2d_visualization(df, "economic_axis", "social_axis", 
                           "Wirtschaftliche Position", "Gesellschaftliche Position",
                           "Politischer Raum: Wirtschaft vs. Gesellschaft")
    
    # Create a 2D visualization of ecological vs. governance axes
    create_2d_visualization(df, "ecological_axis", "governance_axis", 
                           "Ökologische Position", "Governance Position",
                           "Politischer Raum: Ökologie vs. Governance")
    
    # Create an interactive visualization with Plotly
    create_interactive_visualization(df)
    
    print("Done! You can now explore the visualizations.")

def create_2d_visualization(df, x_axis, y_axis, x_label, y_label, title):
    """Create a 2D visualization of the political space"""
    plt.figure(figsize=(10, 8))
    
    # Plot each movement
    for i, row in df.iterrows():
        plt.scatter(row[x_axis], row[y_axis], s=100)
        plt.text(row[x_axis], row[y_axis], row["name"], fontsize=12)
    
    # Add axes
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    
    # Add labels
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    
    # Save figure
    filename = f"political_space_{x_axis}_vs_{y_axis}.png"
    plt.savefig(filename)
    print(f"Saved visualization to {filename}")

def create_interactive_visualization(df):
    """Create an interactive visualization with Plotly"""
    fig = go.Figure()
    
    # Add scatter points for movements
    for i, row in df.iterrows():
        # Format key terms for hover text
        terms_html = "<br>".join([f"<b>{term}</b>: {score:.3f}" 
                                for term, score in row["key_terms"]])
        
        # Format anchor similarities
        similarities = row["anchor_similarities"]
        similarities_html = "<br>".join([f"<b>{dim}</b>: {sim:.3f}" 
                                      for dim, sim in similarities.items()])
        
        # Create hover text
        hover_text = f"""
        <b>{row['name']}</b><br>
        <br>
        <b>Positionen:</b><br>
        Wirtschaft: {row['economic_axis']:.3f}<br>
        Gesellschaft: {row['social_axis']:.3f}<br>
        Ökologie: {row['ecological_axis']:.3f}<br>
        Governance: {row['governance_axis']:.3f}<br>
        <br>
        <b>Charakteristische Begriffe:</b><br>
        {terms_html}<br>
        <br>
        <b>Anker-Ähnlichkeiten:</b><br>
        {similarities_html}
        """
        
        fig.add_trace(go.Scatter(
            x=[row["economic_axis"]],
            y=[row["social_axis"]],
            mode="markers+text",
            marker=dict(size=15),
            text=row["name"],
            textposition="top center",
            name=row["name"],
            hovertext=hover_text,
            hoverinfo="text"
        ))
    
    # Add axes
    fig.add_shape(
        type="line",
        x0=-0.5, y0=0,
        x1=0.5, y1=0,
        line=dict(color="gray", width=1, dash="dash")
    )
    fig.add_shape(
        type="line",
        x0=0, y0=-0.5,
        x1=0, y1=0.5,
        line=dict(color="gray", width=1, dash="dash")
    )
    
    # Add quadrant labels
    fig.add_annotation(x=0.4, y=0.4, text="Progressiv-Wirtschaftsliberal",
                      showarrow=False, bgcolor="rgba(255,255,255,0.5)")
    fig.add_annotation(x=-0.4, y=0.4, text="Progressiv-Wirtschaftslinks",
                      showarrow=False, bgcolor="rgba(255,255,255,0.5)")
    fig.add_annotation(x=-0.4, y=-0.4, text="Konservativ-Wirtschaftslinks",
                      showarrow=False, bgcolor="rgba(255,255,255,0.5)")
    fig.add_annotation(x=0.4, y=-0.4, text="Konservativ-Wirtschaftsliberal",
                      showarrow=False, bgcolor="rgba(255,255,255,0.5)")
    
    # Update layout
    fig.update_layout(
        title="Interaktiver Politischer Raum",
        xaxis=dict(
            title="Wirtschaftliche Position (Links ← → Rechts)",
            range=[-0.5, 0.5]
        ),
        yaxis=dict(
            title="Gesellschaftliche Position (Konservativ ← → Progressiv)",
            range=[-0.5, 0.5]
        ),
        hovermode="closest"
    )
    
    # Save interactive visualization
    fig.write_html("interactive_political_space.html")
    print("Saved interactive visualization to interactive_political_space.html")

if __name__ == "__main__":
    main()
