#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Political Movement Space Prototype
---------------------------------
A prototype for visualizing political movements in a shared embedding space
based on their textual content.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import umap
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px

# Create data directory if it doesn't exist
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# Enhanced sample data with more detailed content for each movement
sample_movements = {
    "Fridays for Future": """Wir sind eine globale Bewegung von jungen Menschen, die sich für Klimagerechtigkeit einsetzen. Wir fordern die Einhaltung des 1,5-Grad-Ziels und Klimagerechtigkeit für alle. Unsere Bewegung basiert auf der wissenschaftlichen Erkenntnis, dass der menschengemachte Klimawandel eine existenzielle Bedrohung darstellt. Wir fordern einen sofortigen Kohleausstieg, den massiven Ausbau erneuerbarer Energien und eine CO2-Steuer, die die wahren Kosten des Klimawandels widerspiegelt. Wir stehen für Generationengerechtigkeit und das Recht auf eine lebenswerte Zukunft. Klimaschutz ist ein Menschenrecht und muss sozial gerecht gestaltet werden, damit nicht die Ärmsten die Hauptlast tragen müssen.""",
    
    "CDU": """Als Volkspartei der Mitte setzen wir auf wirtschaftliche Vernunft und soziale Gerechtigkeit. Wir stehen für eine nachhaltige Entwicklung, die ökonomische, ökologische und soziale Ziele in Einklang bringt. Die Soziale Marktwirtschaft ist unser Leitbild für eine zukunftsfähige Wirtschaftsordnung. Wir setzen auf einen starken Mittelstand, Innovationskraft und Wettbewerbsfähigkeit. Deutschland muss Industriestandort bleiben und gleichzeitig die Digitalisierung vorantreiben. In der Klimapolitik setzen wir auf marktwirtschaftliche Instrumente und technologische Innovation statt auf Verbote. Die Familie ist das Fundament unserer Gesellschaft und verdient besondere Unterstützung. In der Außenpolitik stehen wir für eine starke transatlantische Partnerschaft und die europäische Integration.""",
    
    "Die Grünen": """Wir kämpfen für Klimaschutz, soziale Gerechtigkeit und eine vielfältige Gesellschaft. Unser Ziel ist eine ökologische Transformation der Wirtschaft und eine gerechte Gesellschaft. Der Klimaschutz ist die Überlebensfrage unserer Zeit. Wir wollen bis 2035 klimaneutral werden und setzen auf 100 prozent erneuerbare Energien. Die sozial-ökologische Transformation muss gerecht gestaltet werden und neue Arbeitsplätze schaffen. Wir stehen für eine offene, vielfältige Gesellschaft, in der alle Menschen gleichberechtigt und selbstbestimmt leben können, unabhängig von Herkunft, Religion, Geschlecht oder sexueller Identität. Wir setzen uns für eine humane Flüchtlingspolitik ein und wollen das Asylrecht stärken. In der Wirtschaftspolitik fordern wir klare ökologische Leitplanken und soziale Standards.""",
    
    "Die Linke": """Wir stehen für soziale Gerechtigkeit, Frieden und Demokratie. Wir kämpfen für eine Gesellschaft, in der die Würde jedes Menschen im Mittelpunkt steht und nicht der Profit. Der Kapitalismus führt zu wachsender Ungleichheit und Ausbeutung. Wir fordern eine Umverteilung des gesellschaftlichen Reichtums durch höhere Steuern auf große Vermögen und Erbschaften. Der Mindestlohn muss deutlich steigen, prekäre Beschäftigung beendet werden. Das Gesundheitssystem und die Pflege müssen dem Profitstreben entzogen werden. Wir stehen für Abrüstung und lehnen Auslandseinsätze der Bundeswehr ab. Die NATO muss durch ein kollektives Sicherheitssystem unter Einbeziehung Russlands ersetzt werden. Klimaschutz kann nur sozial gerecht umgesetzt werden.""",
    
    "AfD": """Wir stehen für die Freiheit und Souveränität Deutschlands. Wir fordern eine Rückkehr zu nationaler Selbstbestimmung und traditionellen Werten. Die EU hat sich zu einem undemokratischen Konstrukt entwickelt, das die Souveränität der Nationalstaaten untergräbt. Wir fordern die Rückkehr zu einer Europäischen Wirtschaftsgemeinschaft souveräner Staaten. Die unkontrollierte Masseneinwanderung gefährdet unsere Kultur, Sicherheit und unseren Sozialstaat. Wir setzen uns für eine restriktive Einwanderungspolitik nach dem Vorbild Australiens oder Kanadas ein. Die Familie aus Mann, Frau und Kindern ist das Fundament unserer Gesellschaft und muss besonders gefördert werden. Wir lehnen die Gender-Ideologie ab und setzen uns für den Erhalt traditioneller Werte ein.""",
    
    "Letzte Generation": """Wir fordern die Regierung auf, den Ausstieg aus fossilen Brennstoffen bis 2030 zu beschließen. Unser ziviler Widerstand ist notwendig, um die Klimakatastrophe abzuwenden. Wir befinden uns in einer existenziellen Krise, die das Überleben der menschlichen Zivilisation bedroht. Die bisherigen politischen Maßnahmen sind völlig unzureichend. Wir fordern ein Gesellschaftsrat, in dem Bürger:innen gemeinsam mit Expert:innen verbindliche Maßnahmen für Klimaneutralität erarbeiten. Der fossile Kapitalismus muss überwunden werden. Unser gewaltfreier ziviler Widerstand ist legitim und notwendig, wenn die Regierung ihrer Verantwortung nicht nachkommt. Wir stehen in der Tradition von Mahatma Gandhi und der Bürgerrechtsbewegung und nutzen Störungen des Alltags, um die Dringlichkeit der Klimakrise ins Bewusstsein zu rufen.""",
    
    "BSW": """Wir setzen uns für Frieden, soziale Gerechtigkeit und Souveränität ein. Wir wollen eine Politik, die den Interessen der Mehrheit der Bevölkerung dient und nicht den Profitinteressen einiger weniger. Die wachsende soziale Ungleichheit muss bekämpft werden durch höhere Löhne, sichere Renten und bezahlbaren Wohnraum. Die Privatisierung öffentlicher Daseinsvorsorge muss rückgängig gemacht werden. In der Außenpolitik stehen wir für Diplomatie statt Konfrontation. Deutschland darf nicht zur Kriegspartei werden. Wir fordern ein Ende der Waffenlieferungen in Kriegsgebiete und Verhandlungen zur Beendigung des Ukraine-Krieges. Die Souveränität Deutschlands muss wiederhergestellt werden, sowohl gegenüber den USA als auch gegenüber supranationalen Organisationen wie der EU.""",
    
    "Extinction Rebellion": """Wir sind eine dezentrale, internationale Bewegung, die mit gewaltfreiem zivilem Ungehorsam gegen das Massenaussterben von Tieren und Pflanzen und den drohenden ökologischen Kollaps vorgeht. Wir befinden uns in der sechsten großen Aussterbewelle der Erdgeschichte, diesmal verursacht durch menschliches Handeln. Die Regierungen müssen die Wahrheit über die ökologische Krise aussprechen und den Klimanotstand ausrufen. Wir fordern Netto-Null-Treibhausgasemissionen bis 2025 durch eine radikale Transformation von Wirtschaft und Gesellschaft. Bürgerversammlungen sollen über die notwendigen Maßnahmen entscheiden, da das parlamentarische System versagt hat. Unser Wirtschaftssystem basiert auf endlosem Wachstum auf einem endlichen Planeten, was zwangsläufig zum Kollaps führen muss. Wir brauchen eine regenerative Kultur, die im Einklang mit den planetaren Grenzen steht.""",
    
    "SPD": """Die Sozialdemokratie steht für eine gerechte Gesellschaft, in der alle Menschen die gleichen Chancen haben, unabhängig von ihrer Herkunft. Wir kämpfen für gute Arbeit, faire Löhne und soziale Sicherheit. Der Sozialstaat muss gestärkt werden, um allen Menschen ein würdevolles Leben zu ermöglichen. Wir setzen uns für bezahlbaren Wohnraum, ein gerechtes Bildungssystem und eine solidarische Gesundheitsversorgung ein. In der Wirtschaftspolitik verbinden wir ökonomische Vernunft mit sozialer Gerechtigkeit und ökologischer Verantwortung. Deutschland muss bis 2045 klimaneutral werden, wobei die Kosten der Transformation gerecht verteilt werden müssen. In der Außenpolitik stehen wir für multilaterale Zusammenarbeit, eine starke EU und die transatlantische Partnerschaft.""",
    
    "FDP": """Wir stehen für die Freiheit des Einzelnen, Eigenverantwortung und die Kraft des Marktes. Der Staat sollte sich auf seine Kernaufgaben konzentrieren und den Bürgern mehr Freiräume lassen. Wir setzen uns für Bürokratieabbau, Steuersenkungen und die Förderung von Unternehmertum ein. Die Digitalisierung bietet enorme Chancen, die wir durch moderne Infrastruktur und digitale Bildung nutzen müssen. In der Klimapolitik setzen wir auf Innovation, neue Technologien und einen funktionierenden Emissionshandel statt auf Verbote und staatliche Bevormundung. Jeder Mensch soll unabhängig von seiner Herkunft die Chance haben, durch eigene Leistung aufzusteigen. Die EU muss als Wettbewerbs- und Rechtsgemeinschaft gestärkt werden, ohne zu einem zentralistischen Superstaat zu werden.""",
    
    "Volt Europa": """Wir sind die erste paneuropäische Partei, die für ein geeintes, demokratisches und föderales Europa kämpft. Die großen Herausforderungen unserer Zeit – Klimawandel, Migration, Digitalisierung, soziale Ungleichheit – können nur auf europäischer Ebene gelöst werden. Wir fordern eine echte europäische Demokratie mit einem starken Europäischen Parlament und transnationalen Listen. Die EU muss bis 2040 klimaneutral werden durch massive Investitionen in erneuerbare Energien und nachhaltige Infrastruktur. Wir stehen für eine progressive, evidenzbasierte Politik jenseits ideologischer Grabenkämpfe. Digitalisierung muss menschenzentriert gestaltet werden und allen zugutekommen. In der Migrationspolitik setzen wir uns für legale Einwanderungswege, faire Asylverfahren und eine gerechte Verteilung von Geflüchteten ein.""",
    
    "DiEM25": """Die Demokratie in Europa Bewegung 25 kämpft für eine radikale Demokratisierung der EU und eine Überwindung des neoliberalen Kapitalismus. Die Austeritätspolitik hat zu wachsender Ungleichheit und der Schwächung demokratischer Institutionen geführt. Wir fordern einen Green New Deal für Europa mit massiven öffentlichen Investitionen in den ökologischen Umbau, finanziert durch die Europäische Investitionsbank. Die Macht der Finanzmärkte und Konzerne muss gebrochen werden. Wir stehen für ein offenes Europa ohne Grenzen und eine humane Migrationspolitik. Technologischer Fortschritt muss allen Menschen zugutekommen, nicht nur den Eigentümern der Plattformen. Wir wollen eine demokratische Wirtschaft mit mehr Mitbestimmung und genossenschaftlichen Strukturen."""
}

# Create a DataFrame
df = pd.DataFrame(list(sample_movements.items()), columns=['movement', 'text'])
print(f"Loaded {len(df)} political movements")

# Load model and generate embeddings
print("Loading sentence transformer model...")
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')  # Good for German text

print("Generating embeddings...")
embeddings = model.encode(df['text'].tolist())
df['embedding'] = list(embeddings)

# Dimensionality reduction with UMAP
print("Performing dimensionality reduction with UMAP...")
reducer = umap.UMAP(n_components=2, n_neighbors=4, min_dist=0.3, random_state=42)
reduced_embeddings = reducer.fit_transform(np.array([e for e in embeddings]))
df['x'] = reduced_embeddings[:, 0]
df['y'] = reduced_embeddings[:, 1]

# Calculate similarity matrix
similarity_matrix = cosine_similarity(embeddings)
np.fill_diagonal(similarity_matrix, 0)  # Remove self-similarity

# Create network graph
print("Creating network visualization...")
G = nx.Graph()

# Add nodes
for i, row in df.iterrows():
    G.add_node(row['movement'], pos=(row['x'], row['y']))

# Add edges based on similarity threshold
threshold = 0.5  # Adjust this threshold as needed
for i in range(len(df)):
    for j in range(i+1, len(df)):
        sim = similarity_matrix[i, j]
        if sim > threshold:
            G.add_edge(df.iloc[i]['movement'], df.iloc[j]['movement'], 
                      weight=sim, width=sim*5)

# Get positions from the graph
pos = nx.get_node_attributes(G, 'pos')

# Create color mapping with more nuanced categories
movement_types = {
    "Fridays for Future": "climate_activist",
    "CDU": "conservative",
    "Die Grünen": "green_party",
    "Die Linke": "left",
    "AfD": "right",
    "Letzte Generation": "climate_activist",
    "BSW": "left_populist",
    "Extinction Rebellion": "climate_activist",
    "SPD": "social_democratic",
    "FDP": "liberal",
    "Volt Europa": "pro_european",
    "DiEM25": "left_european"
}

color_map = {
    "climate_activist": "#1a9850",  # Dark green
    "green_party": "#66c2a5",      # Light green
    "conservative": "#000000",     # Black
    "left": "#d73027",            # Red
    "right": "#4575b4",           # Blue
    "left_populist": "#fc8d59",   # Orange
    "social_democratic": "#e41a1c", # Bright red
    "liberal": "#ffff33",         # Yellow
    "pro_european": "#984ea3",    # Purple
    "left_european": "#f781bf"    # Pink
}

df['movement_type'] = df['movement'].map(movement_types)
df['color'] = df['movement_type'].map(color_map)

# Create matplotlib visualization
plt.figure(figsize=(12, 8))
nx.draw_networkx(
    G, pos=pos,
    node_color=[color_map[movement_types[node]] for node in G.nodes()],
    node_size=500,
    font_size=10,
    width=[G[u][v].get('width', 1) for u, v in G.edges()],
    with_labels=True,
    alpha=0.8
)
plt.title("Political Movement Space - Network Visualization")
plt.axis('off')
plt.tight_layout()
plt.savefig('political_movement_network.png', dpi=300)
print("Saved network visualization to political_movement_network.png")

# Create interactive Plotly visualization
fig = go.Figure()

# Add nodes
for i, row in df.iterrows():
    fig.add_trace(go.Scatter(
        x=[row['x']],
        y=[row['y']],
        mode='markers+text',
        marker=dict(size=20, color=row['color']),
        text=row['movement'],
        name=row['movement'],
        textposition="top center",
        hoverinfo="text",
        hovertext=f"{row['movement']}<br>{row['text'][:100]}..."
    ))

# Add edges
for u, v, data in G.edges(data=True):
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    weight = data.get('weight', 0.1)
    
    fig.add_trace(go.Scatter(
        x=[x0, x1], 
        y=[y0, y1],
        mode='lines',
        line=dict(width=weight*5, color='rgba(120, 120, 120, 0.5)'),
        hoverinfo='text',
        hovertext=f"Similarity: {weight:.2f}",
        showlegend=False
    ))

fig.update_layout(
    title="Interactive Political Movement Space",
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    hovermode='closest',
    plot_bgcolor='rgba(240, 240, 240, 0.8)'
)

fig.write_html("political_movement_interactive.html")
print("Saved interactive visualization to political_movement_interactive.html")
print("\nDone! You can now explore the visualizations.")
