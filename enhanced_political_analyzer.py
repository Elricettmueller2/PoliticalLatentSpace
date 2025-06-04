import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist

class HybridPoliticalSpace:
    def __init__(self):
        print("Initializing Enhanced Hybrid Political Space...")
        # Using a multilingual model suitable for German and other languages
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
        
        # Expert-defined anchor points for key political dimensions
        self.anchors = {
            # Original Dimensions
            "economic_left": "Wir setzen uns für staatliche Regulierung, Umverteilung und soziale Absicherung ein. Der Staat muss aktiv in die Wirtschaft eingreifen, um Ungleichheit zu bekämpfen und öffentliche Dienstleistungen zu garantieren.",
            "economic_right": "Wir fördern freie Märkte, niedrige Steuern und unternehmerische Freiheit. Die Wirtschaft funktioniert am besten mit minimaler staatlicher Einmischung und maximaler individueller wirtschaftlicher Freiheit.",
            "progressive": "Wir stehen für gesellschaftlichen Wandel, Diversität und neue Lebensformen. Gesellschaftliche Normen müssen sich weiterentwickeln, um Vielfalt und Inklusion zu fördern.",
            "conservative": "Wir bewahren Traditionen, Familie und kulturelle Identität. Bewährte Werte und Institutionen müssen geschützt werden, um gesellschaftliche Stabilität zu gewährleisten.",
            "ecological": "Klimaschutz und Umweltschutz haben für uns höchste Priorität. Wir müssen unsere Wirtschaft und Lebensweise grundlegend umgestalten, um die ökologische Krise zu bewältigen.",
            "growth": "Wirtschaftswachstum und Wohlstand sind entscheidend. Umweltschutz darf nicht zu Lasten von Arbeitsplätzen und wirtschaftlicher Entwicklung gehen.",
            "nationalist": "Nationale Souveränität und Identität müssen geschützt werden. Die Interessen unseres Landes haben Vorrang vor internationalen Verpflichtungen.",
            "internationalist": "Internationale Zusammenarbeit und globale Lösungen sind entscheidend. Nur durch gemeinsames Handeln können wir die großen Herausforderungen unserer Zeit bewältigen.",

            # Newly Added Dimensions
            "authoritarian": "Die Führung muss stark sein und Ordnung durchsetzen. Gesellschaftliche Stabilität erfordert klare Hierarchien und entschlossenes Handeln gegen Störer.",
            "libertarian": "Individuelle Freiheit und persönliche Autonomie sind nicht verhandelbar. Der Staat sollte minimalen Einfluss auf das Leben der Bürger haben.",
            "secular": "Politik und Religion müssen strikt getrennt sein. Staatliche Entscheidungen dürfen nicht auf religiösen Überzeugungen basieren.",
            "religious": "Religiöse Werte und Traditionen sollten eine zentrale Rolle in der Gesellschaft und Politik spielen. Moralische Grundsätze sind wichtige Leitlinien.",
            "centralized": "Zentrale Steuerung und einheitliche Standards sind wichtig für Effizienz und Gleichheit. Wesentliche Entscheidungen sollten auf nationaler Ebene getroffen werden.",
            "federalist": "Regionale Autonomie und lokale Selbstverwaltung sind entscheidend. Entscheidungen sollten so nah wie möglich an den betroffenen Bürgern getroffen werden.",
            "tech_progressive": "Technologischer Fortschritt ist ein wichtiger Treiber für gesellschaftliche Entwicklung und Wohlstand. Innovationen müssen gefördert werden.",
            "tech_skeptical": "Technologische Entwicklungen müssen kritisch hinterfragt und streng reguliert werden, um negative Auswirkungen auf Gesellschaft und Umwelt zu vermeiden."
        }
        
        print("Generating anchor embeddings...")
        self.anchor_embeddings = {k: self.model.encode(v) for k, v in self.anchors.items()}
        
    def position_movement(self, name, text):
        """Position a political movement in the hybrid space including new dimensions"""
        embedding = self.model.encode(text)
        
        similarities = {dim: cosine_similarity([embedding], [anchor_emb])[0][0] 
                       for dim, anchor_emb in self.anchor_embeddings.items()}
        
        position = {"name": name}
        
        # Original Axes
        position["economic_axis"] = similarities["economic_left"] - similarities["economic_right"]
        position["social_axis"] = similarities["progressive"] - similarities["conservative"]
        position["ecological_axis"] = similarities["ecological"] - similarities["growth"]
        position["governance_axis"] = similarities["internationalist"] - similarities["nationalist"]
        
        # New Axes (Libertarian/Secular/Federalist/Tech-Progressive are positive poles by convention here)
        position["auth_lib_axis"] = similarities["libertarian"] - similarities["authoritarian"]
        position["sec_rel_axis"] = similarities["secular"] - similarities["religious"]
        position["cent_fed_axis"] = similarities["federalist"] - similarities["centralized"]
        position["tech_axis"] = similarities["tech_progressive"] - similarities["tech_skeptical"]
        
        # Store raw similarities for potential hover text or detailed views
        position["similarities"] = {k: round(v, 3) for k,v in similarities.items()}

        return position

    def extract_key_terms(self, text, top_n=10):
        """Extract key terms from text using TF-IDF"""
        # Simple German stopwords (can be expanded or replaced with a standard library)
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
        
        try:
            tfidf = TfidfVectorizer(max_features=100, stop_words=german_stopwords)
            tfidf_matrix = tfidf.fit_transform([text])
            feature_names = np.array(tfidf.get_feature_names_out())
            
            tfidf_scores = tfidf_matrix.toarray()[0]
            sorted_indices = np.argsort(tfidf_scores)[::-1]
            top_terms = [(feature_names[idx], round(tfidf_scores[idx], 3)) 
                        for idx in sorted_indices[:top_n] if tfidf_scores[idx] > 0]
        except ValueError: # Handles empty vocabulary, e.g. if text is only stopwords
            top_terms = [("N/A", 0)]
        
        return top_terms

# --- Visualization Functions (Stubs) ---

def create_dimension_pair_matrix(df):
    print("\nGenerating Dimension-Pair Matrix Visualization...")
    # TODO: Implement using Plotly Express scatter_matrix or similar
    # Example: fig = px.scatter_matrix(df, dimensions=[col for col in df.columns if col.endswith('_axis')], color="name")
    # fig.write_html("dimension_pair_matrix.html")
    pass

def create_interactive_dimension_explorer(df):
    print("\nGenerating Interactive Dimension Explorer...")
    # TODO: Implement using Plotly with dropdowns for X and Y axes
    pass

def create_hierarchical_dimension_tree(df):
    print("\nGenerating Hierarchical Dimension Tree...")
    # TODO: Implement using Scipy for clustering and Plotly/Matplotlib for dendrogram
    pass

def create_parallel_coordinates_plot(df):
    print("\nGenerating Parallel Coordinates Plot...")
    # TODO: Implement using Plotly Express parallel_coordinates or Parcoords
    pass

def create_dimension_importance_profiles(df):
    print("\nGenerating Dimension Importance Profiles...")
    # TODO: Implement using Plotly bar charts for each party across dimensions
    pass 

def create_main_interactive_political_diagram(df):
    """
    Create a comprehensive, integrated interactive political visualization that combines
    multiple visualization approaches into a single HTML output.
    
    Args:
        df (pandas.DataFrame): DataFrame containing political positions and key terms
                              for various political movements/parties.
    """
    print("\nGenerating Main Interactive Political Diagram...")
    
    # List of all political dimensions in the data
    dimensions = [
        'economic_axis', 'social_axis', 'ecological_axis', 'governance_axis',
        'auth_lib_axis', 'sec_rel_axis', 'cent_fed_axis', 'tech_axis'
    ]
    
    dimension_labels = {
        'economic_axis': 'Economic (Left-Right)',
        'social_axis': 'Social (Progressive-Conservative)',
        'ecological_axis': 'Ecological (Green-Growth)',
        'governance_axis': 'Governance (National-International)',
        'auth_lib_axis': 'Authority (Authoritarian-Libertarian)',
        'sec_rel_axis': 'Religion (Secular-Religious)',
        'cent_fed_axis': 'Structure (Centralized-Federalist)',
        'tech_axis': 'Technology (Progressive-Skeptical)'
    }
    
    # Create a Plotly figure with subplots (one main plot for scatter, one for profiles)
    fig = make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3],
                       specs=[[{"type": "scatter"}, {"type": "bar"}]],
                       subplot_titles=["Political Positions", "Dimension Profile"],
                       horizontal_spacing=0.05)
    
    # Define marker colors for each party for consistency across views
    party_colors = px.colors.qualitative.Plotly[:len(df)]
    color_dict = {party: color for party, color in zip(df['name'].tolist(), party_colors)}
    
    # Initialize the scatter plot with default dimensions (economic and social axes)
    scatter_data = []
    for i, row in df.iterrows():
        # Format key terms for hover text
        if 'key_terms' in row and row['key_terms']:
            terms_html = "<br>".join([f"<b>{term}</b>: {score:.3f}" 
                                    for term, score in row["key_terms"]])
        else:
            terms_html = "No key terms available"
        
        # Format all dimension scores for hover text
        dimensions_html = "<br>".join([f"<b>{dimension_labels[dim]}</b>: {row[dim]:.3f}" 
                                     for dim in dimensions])
        
        # Format anchor similarities if available
        similarities_html = ""
        if 'anchor_similarities' in row and row['anchor_similarities']:
            similarities_html = "<br><br><b>Anchor Similarities:</b><br>" + "<br>".join(
                [f"<b>{dim}</b>: {sim:.3f}" for dim, sim in row["anchor_similarities"].items()])
        
        # Create hover text
        hover_text = f"""
        <b>{row['name']}</b><br>
        <br>
        <b>Positions:</b><br>
        {dimensions_html}
        <br>
        <b>Characteristic Terms:</b><br>
        {terms_html}
        {similarities_html}
        """
        
        scatter_data.append(go.Scatter(
            x=[row["economic_axis"]],
            y=[row["social_axis"]],
            mode="markers+text",
            marker=dict(size=15, color=color_dict[row['name']]),
            text=row["name"],
            textposition="bottom center",
            hovertemplate=hover_text,
            name=row["name"],
            customdata=[row['name']],  # Store name for click events
        ))
    
    # Add all scatter traces to the main subplot and debug output
    print(f"Adding {len(scatter_data)} data points to visualization:")
    for i, trace in enumerate(scatter_data):
        print(f"  Point {i+1}: Name={trace['name']}, X={trace['x'][0]:.3f}, Y={trace['y'][0]:.3f}")
        fig.add_trace(trace, row=1, col=1)
    
    # Add quadrant lines
    fig.add_shape(type="line", x0=0, y0=-1, x1=0, y1=1, line=dict(color="gray", width=1, dash="dash"), row=1, col=1)
    fig.add_shape(type="line", x0=-1, y0=0, x1=1, y1=0, line=dict(color="gray", width=1, dash="dash"), row=1, col=1)
    
    # Add quadrant labels
    quadrant_labels = {
        (-0.8, 0.8): "Left<br>Progressive",
        (0.8, 0.8): "Right<br>Progressive",
        (-0.8, -0.8): "Left<br>Conservative",
        (0.8, -0.8): "Right<br>Conservative"
    }
    
    for (x, y), text in quadrant_labels.items():
        fig.add_annotation(
            x=x, y=y, text=text,
            showarrow=False,
            font=dict(size=10, color="gray"),
            row=1, col=1
        )
    
    # Empty placeholder for the profile plot (will be populated on click)
    fig.add_trace(go.Bar(
        x=dimensions,
        y=[0] * len(dimensions),
        name="Select a party",
        visible=True,
        marker_color="lightgray",
        hoverinfo="none"
    ), row=1, col=2)
    
    # Update layout
    fig.update_layout(
        title="Interactive Multi-Dimensional Political Space",
        height=800,
        width=1200,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        updatemenus=[
            # Dropdown for X-axis
            dict(
                buttons=[
                    dict(label=dimension_labels[dim],
                         method="update",
                         args=[
                             {"x": [[row[dim] for _ in range(1)] for i, row in df.iterrows()]},
                             {"xaxis.title": dimension_labels[dim]}
                         ])
                    for dim in dimensions
                ],
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top"
            ),
            # Dropdown for Y-axis
            dict(
                buttons=[
                    dict(label=dimension_labels[dim],
                         method="update",
                         args=[
                             {"y": [[row[dim] for _ in range(1)] for i, row in df.iterrows()]},
                             {"yaxis.title": dimension_labels[dim]}
                         ])
                    for dim in dimensions
                ],
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.37,
                xanchor="left",
                y=1.15,
                yanchor="top"
            ),
            # Menu for additional views (removed toggle as parallel coordinates is now a separate file)
            dict(
                buttons=[
                    dict(label="View Parallel Coordinates",
                         method="restyle",
                         args=[
                            {"type": "scatter"},
                            {"title": "View opened in separate window: parallel_coordinates_view.html"}
                         ])
                ],
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=False,
                x=0.64,
                xanchor="left",
                y=1.15,
                yanchor="top"
            )
        ],
        annotations=[
            dict(text="X-Axis:", x=0.02, y=1.15, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=14)),
            dict(text="Y-Axis:", x=0.3, y=1.15, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=14)),
            dict(text="Additional Views:", x=0.54, y=1.15, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=14))
        ]
    )
    
    # Calculate better axis ranges based on data (with padding)
    x_values = [trace['x'][0] for trace in scatter_data]
    y_values = [trace['y'][0] for trace in scatter_data]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    
    # Add padding (50% on each side)
    x_padding = (x_max - x_min) * 0.5
    y_padding = (y_max - y_min) * 0.5
    x_range = [x_min - x_padding, x_max + x_padding]
    y_range = [y_min - y_padding, y_max + y_padding]
    
    print(f"Using optimized axis ranges: X={x_range}, Y={y_range}")
    
    # Update axes
    fig.update_xaxes(title=dimension_labels["economic_axis"], range=x_range, zeroline=True, row=1, col=1)
    fig.update_yaxes(title=dimension_labels["social_axis"], range=y_range, zeroline=True, row=1, col=1)
    
    # Update profile subplot
    fig.update_xaxes(title="Political Dimensions", tickangle=45, row=1, col=2)
    fig.update_yaxes(title="Position Score", row=1, col=2)
    
    # Create the parallel coordinates plot as a separate figure for alternative view
    # Using a numeric column for color to avoid errors
    # First, create a numeric color index for each party
    df_with_color = df.copy()
    df_with_color['color_index'] = np.arange(len(df))
    
    parcoords_fig = px.parallel_coordinates(
        df_with_color,
        dimensions=dimensions,
        labels=dimension_labels,
        color='color_index',
        color_continuous_scale=px.colors.qualitative.Plotly
    )
    
    # Update layout for better readability
    parcoords_fig.update_layout(
        title="Parallel Coordinates View of Political Dimensions",
        font=dict(size=12),
    )
    
    # Save the parallel coordinates as a separate HTML file
    parcoords_fig.write_html("parallel_coordinates_view.html", include_plotlyjs=True, full_html=True)
    
    # Add JavaScript callbacks for interactivity
    fig.update_layout(
        # This is a simplified version of what would be needed for full interactivity
        # In a real implementation, this would be more complex with proper JavaScript callbacks
        clickmode='event+select'
    )
    
    # Write to HTML file with full HTML wrapper that includes JS for plotly
    # This would include additional JavaScript for the click handling in a real implementation
    fig.write_html("integrated_political_diagram.html", include_plotlyjs=True, full_html=True)
    
    print(f"Interactive visualization saved as 'integrated_political_diagram.html'")
    return fig

# --- Main Execution Block ---
if __name__ == "__main__":
    print("Starting Enhanced Political Analyzer Prototype...")
    
    # Initialize the political space model
    political_analyzer = HybridPoliticalSpace()
    
    # Sample data for political movements (replace with actual manifestos or texts)
    movements_data = {
        "Partei Alpha (Grün-Progressiv)": 
            "Wir kämpfen für konsequenten Klimaschutz, erneuerbare Energien und eine ökologische Landwirtschaft. Gleichzeitig setzen wir uns für soziale Gerechtigkeit, Diversität und eine offene Gesellschaft ein. Internationale Zusammenarbeit ist für uns unerlässlich. Wir fordern mehr Bürgerbeteiligung und Transparenz.",
        "Partei Beta (Wirtschaftsliberal-Konservativ)": 
            "Unser Fokus liegt auf einer starken Wirtschaft, niedrigen Steuern und Bürokratieabbau. Private Initiative und Wettbewerb sind die Motoren des Fortschritts. Wir stehen für traditionelle Werte, Familie und nationale Interessen. Sicherheit und Ordnung müssen gewährleistet sein.",
        "Partei Gamma (Sozial-National)": 
            "Wir fordern einen starken Sozialstaat, der die heimische Bevölkerung schützt. Nationale Interessen und kulturelle Identität stehen im Vordergrund. Die Wirtschaft muss dem Volk dienen, nicht umgekehrt. Wir sind skeptisch gegenüber globalen Abkommen und unkontrollierter Zuwanderung.",
        "Partei Delta (Libertär-Technokratisch)":
            "Maximale individuelle Freiheit und minimale staatliche Einmischung sind unsere Kernprinzipien. Freie Märkte und technologischer Fortschritt werden alle Probleme lösen. Wir sind für eine strikte Trennung von Staat und Kirche und für eine dezentrale Machtverteilung."
    }
    
    all_positions = []
    
    print("\nProcessing political movements...")
    for name, text_content in movements_data.items():
        print(f"  Positioning: {name}")
        position_data = political_analyzer.position_movement(name, text_content)
        key_terms = political_analyzer.extract_key_terms(text_content)
        position_data["key_terms"] = key_terms
        all_positions.append(position_data)
        
    # Convert to DataFrame for easier handling by visualization functions
    df_positions = pd.DataFrame(all_positions)
    
    print("\n--- Political Positions DataFrame ---")
    print(df_positions[['name', 'economic_axis', 'social_axis', 'ecological_axis', 'governance_axis', 'auth_lib_axis', 'sec_rel_axis', 'cent_fed_axis', 'tech_axis']].head())
    print("-------------------------------------")
    
    # Call visualization functions (currently stubs)
    # Individual visualizations (can be commented out if only using the main diagram)
    # create_dimension_pair_matrix(df_positions)
    # create_interactive_dimension_explorer(df_positions)
    # create_hierarchical_dimension_tree(df_positions)
    # create_parallel_coordinates_plot(df_positions)
    # create_dimension_importance_profiles(df_positions)
    
    # Create the main integrated interactive visualization
    print("\nCreating the integrated interactive political diagram...")
    main_fig = create_main_interactive_political_diagram(df_positions)
    
    print("\nPrototype execution complete. Open 'integrated_political_diagram.html' to see the interactive visualization.")
