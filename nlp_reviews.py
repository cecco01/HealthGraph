"""
Analisi NLP delle Recensioni Ospedali
====================================

Questo modulo gestisce l'analisi del linguaggio naturale delle recensioni degli ospedali
utilizzando Google Gemini 1.5 Flash.

Funzionalità:
- Estrazione dei temi principali dalle recensioni
- Analisi avanzata con AI
- Visualizzazione del sottografo temi
"""

import os
import pandas as pd
import streamlit as st
import time
import json
from typing import Dict, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import re
from dotenv import load_dotenv

# Google Gemini
import google.generativeai as genai

# Carica le variabili d'ambiente
load_dotenv('api_keys.env')

# =============================================================================
# Configurazione Gemini
# =============================================================================

class GeminiNLPReviewAnalyzer:
    """
    Classe per l'analisi NLP delle recensioni degli ospedali usando Google Gemini
    """
    
    def __init__(self):
        """
        Inizializza l'analizzatore NLP con Gemini
        """
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None
    
    def load_reviews_data(self, file_path: str = "data/hospital-reviews.csv") -> pd.DataFrame:
        """
        Carica i dati delle recensioni dal file CSV
        
        Args:
            file_path: Percorso del file CSV delle recensioni
            
        Returns:
            DataFrame con i dati delle recensioni
        """
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            df = df.dropna(subset=['Feedback'])
            df = df[df['Feedback'].str.strip() != '']
            
            return df
        except Exception as e:
            return pd.DataFrame()
    
    def extract_themes_gemini(self, text: str) -> Dict:
        """
        Estrae i temi principali usando Gemini
        
        Args:
            text: Testo da analizzare
            
        Returns:
            Dizionario con i temi estratti
        """
        if not self.model:
            return {"themes": [], "error": "Gemini non configurato"}
        
        try:
            prompt = f"""
            Estrai i temi principali dalla seguente recensione di un ospedale.
            Identifica temi specifici come: personale, pulizia, tempi di attesa, costi, qualità medica, accoglienza, strutture, emergenza, comfort, comunicazione, etc.
            Rispondi solo con un JSON nel formato:
            {{"themes": ["tema1", "tema2", "tema3"], "main_theme": "tema_principale"}}
            
            Recensione: "{text}"
            """
            
            response = self.model.generate_content(prompt)
            result = json.loads(response.text)
            
            # Assicurati che ci siano temi
            if not result.get('themes') or len(result.get('themes', [])) == 0:
                # Fallback con analisi basata su parole chiave
                text_lower = text.lower()
                theme_keywords = {
                    'personale': ['doctor', 'nurse', 'staff', 'dottore', 'infermiere', 'medico', 'personale'],
                    'pulizia': ['clean', 'hygiene', 'pulito', 'igiene', 'sanitario', 'sterile'],
                    'tempi_attesa': ['wait', 'time', 'attesa', 'tempo', 'lungo', 'veloce', 'rapido'],
                    'costi': ['cost', 'price', 'costo', 'prezzo', 'caro', 'economico', 'spesa'],
                    'qualita_medica': ['treatment', 'care', 'cura', 'trattamento', 'terapia', 'diagnosi'],
                    'accoglienza': ['hospitality', 'welcome', 'accoglienza', 'cortese', 'gentile'],
                    'strutture': ['facility', 'room', 'struttura', 'stanza', 'moderno', 'vecchio'],
                    'emergenza': ['emergency', 'urgente', 'emergenza', 'critico', 'grave'],
                    'comfort': ['comfort', 'comfortable', 'comodo', 'scomodo', 'letto', 'camera']
                }
                
                found_themes = []
                for theme, keywords in theme_keywords.items():
                    if any(keyword in text_lower for keyword in keywords):
                        found_themes.append(theme)
                
                if found_themes:
                    result['themes'] = found_themes[:3]  # Massimo 3 temi
                    result['main_theme'] = found_themes[0]
                else:
                    result['themes'] = ['qualita_medica']
                    result['main_theme'] = 'qualita_medica'
            
            return result
            
        except Exception as e:
            # Fallback con analisi basata su parole chiave
            text_lower = text.lower()
            theme_keywords = {
                'personale': ['doctor', 'nurse', 'staff', 'dottore', 'infermiere', 'medico', 'personale'],
                'pulizia': ['clean', 'hygiene', 'pulito', 'igiene', 'sanitario', 'sterile'],
                'tempi_attesa': ['wait', 'time', 'attesa', 'tempo', 'lungo', 'veloce', 'rapido'],
                'costi': ['cost', 'price', 'costo', 'prezzo', 'caro', 'economico', 'spesa'],
                'qualita_medica': ['treatment', 'care', 'cura', 'trattamento', 'terapia', 'diagnosi'],
                'accoglienza': ['hospitality', 'welcome', 'accoglienza', 'cortese', 'gentile'],
                'strutture': ['facility', 'room', 'struttura', 'stanza', 'moderno', 'vecchio'],
                'emergenza': ['emergency', 'urgente', 'emergenza', 'critico', 'grave'],
                'comfort': ['comfort', 'comfortable', 'comodo', 'scomodo', 'letto', 'camera']
            }
            
            found_themes = []
            for theme, keywords in theme_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    found_themes.append(theme)
            
            if found_themes:
                return {"themes": found_themes[:3], "main_theme": found_themes[0]}
            else:
                return {"themes": ['qualita_medica'], "main_theme": 'qualita_medica'}
    
    def analyze_reviews_batch(self, reviews_df: pd.DataFrame, max_reviews: int = 50) -> pd.DataFrame:
        """
        Analizza un batch di recensioni usando Gemini
        
        Args:
            reviews_df: DataFrame con le recensioni
            max_reviews: Numero massimo di recensioni da analizzare
            
        Returns:
            DataFrame con i risultati dell'analisi
        """
        if not self.model:
            st.error("Gemini non configurato. Aggiungi la chiave API GEMINI_API_KEY.")
            return pd.DataFrame()
        
        reviews_df = reviews_df.head(max_reviews).copy()
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, row in reviews_df.iterrows():
            status_text.text(f"Analizzando recensione {idx+1}/{len(reviews_df)}...")
            
            text = row['Feedback']
            
            # Estrazione temi
            themes_result = self.extract_themes_gemini(text)
            
            # Salva risultati
            result = {
                'original_text': text,
                'themes': themes_result.get('themes', []),
                'main_theme': themes_result.get('main_theme', 'generale'),
                'ratings': row.get('Ratings', 0)
            }
            
            results.append(result)
            
            # Aggiorna progresso
            progress_bar.progress((idx + 1) / len(reviews_df))
            time.sleep(0.5)  # Pausa per evitare rate limiting
        
        progress_bar.empty()
        status_text.empty()
        
        return pd.DataFrame(results)
    
    def build_hospital_theme_graph_gemini(self, reviews_df: pd.DataFrame, selected_hospital: str, max_reviews: int = 20) -> nx.Graph:
        """
        Costruisce un sottografo che collega un ospedale ai temi estratti dalle sue recensioni
        
        Args:
            reviews_df: DataFrame con le recensioni
            selected_hospital: Nome dell'ospedale selezionato
            max_reviews: Numero massimo di recensioni da analizzare
            
        Returns:
            NetworkX Graph con ospedale e temi collegati
        """
        if not self.model:
            return nx.Graph()
        
        # Usa tutte le recensioni disponibili (simuliamo che siano per l'ospedale selezionato)
        hosp_reviews = reviews_df.copy()
        
        if len(hosp_reviews) > max_reviews:
            hosp_reviews = hosp_reviews.head(max_reviews)
        
        # Analizza temi
        theme_counts = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, row in hosp_reviews.iterrows():
            status_text.text(f"Analizzando recensione {idx+1}/{len(hosp_reviews)}...")
            
            text = row['Feedback']
            
            # Estrai temi
            themes_result = self.extract_themes_gemini(text)
            themes = themes_result.get('themes', [])
            
            # Aggiorna statistiche temi
            for theme in themes:
                if theme not in theme_counts:
                    theme_counts[theme] = 0
                theme_counts[theme] += 1
            
            # Se non trova temi, usa "qualita_medica"
            if not themes:
                if 'qualita_medica' not in theme_counts:
                    theme_counts['qualita_medica'] = 0
                theme_counts['qualita_medica'] += 1
            
            # Aggiorna progresso
            progress_bar.progress((idx + 1) / len(hosp_reviews))
            time.sleep(0.5)
        
        progress_bar.empty()
        
        # Costruisci il grafo
        G = nx.Graph()
        G.add_node(selected_hospital, type='Hospital')
        
        # Aggiungi temi principali
        for theme, count in theme_counts.items():
            if count > 0:
                G.add_node(theme, type='Theme')
                G.add_edge(selected_hospital, theme, count=count)
        
        # Aggiungi nodi tematici generali se il grafo è troppo piccolo
        if len(G.nodes()) < 5:
            general_themes = ['personale', 'pulizia', 'tempi_attesa', 'costi', 'qualita_medica', 'accoglienza']
            for theme in general_themes:
                if theme not in G.nodes():
                    G.add_node(theme, type='GeneralTheme')
                    G.add_edge(selected_hospital, theme, count=1)
        
        return G
    
    def plot_hospital_theme_graph(self, G: nx.Graph, selected_hospital: str):
        """
        Visualizza il sottografo Ospedale ↔ Temi con Plotly.
        Spessore arco: proporzionale al count (numero di recensioni che menzionano il tema).
        """
        pos = nx.spring_layout(G, k=1, iterations=50)
        edge_x = []
        edge_y = []
        edge_widths = []
        
        for u, v, data in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            
            count = data.get('count', 1)
            edge_widths.append(2 + count * 0.5)  # Spessore basato sul count
        
        fig = go.Figure()
        for i in range(len(edge_x)//3):
            fig.add_trace(go.Scatter(
                x=edge_x[i*3:i*3+2],
                y=edge_y[i*3:i*3+2],
                mode='lines',
                line=dict(width=edge_widths[i], color='blue'),
                hoverinfo='none',
                showlegend=False
            ))
        
        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]
        node_text = []
        node_color = []
        
        for n, attr in G.nodes(data=True):
            if attr['type'] == 'Hospital':
                node_text.append(f"{n} (Ospedale)")
                node_color.append('#1f77b4')
            elif attr['type'] == 'SimilarHospital':
                node_text.append(f"{n} (Ospedale Simile)")
                node_color.append('#ff7f0e')
            elif attr['type'] == 'GeneralTheme':
                node_text.append(f"{n} (Tema Generale)")
                node_color.append('#d62728')
            else:
                node_text.append(f"{n} (Tema)")
                node_color.append('#2ca02c')
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=[n for n in G.nodes()],
            textposition="bottom center",
            marker=dict(size=38, color=node_color, line=dict(width=2, color='white')),
            hoverinfo='text',
            hovertext=node_text
        ))
        
        fig.update_layout(
            title=f'Sottografo Temi Recensioni per {selected_hospital}',
            showlegend=False,
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            dragmode='pan',
            height=600
        )
        
        return fig

# =============================================================================
# Funzioni di utilità per l'interfaccia
# =============================================================================

@st.cache_resource
def get_analyzer():
    """
    Ottiene l'istanza dell'analizzatore NLP con cache
    """
    return GeminiNLPReviewAnalyzer()

def display_hospital_theme_graph_gemini():
    """
    Mostra la sezione del sottografo temi recensioni
    """
    st.subheader("Analisi Temi delle Recensioni")
    
    analyzer = get_analyzer()
    
    if not analyzer.model:
        st.error("Gemini non configurato. Aggiungi la chiave API GEMINI_API_KEY nel file api_keys.env")
        return
    
    # Carica le recensioni
    reviews_df = analyzer.load_reviews_data()
    if reviews_df.empty:
        st.warning("Non ci sono dati di recensioni disponibili.")
        return
    
    # Mostra statistiche base
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Recensioni Totali", len(reviews_df))
    with col2:
        st.metric("Recensioni con Rating", len(reviews_df[reviews_df['Ratings'] > 0]))
    with col3:
        st.metric("Rating Medio", f"{reviews_df['Ratings'].mean():.1f}")
    
    # Lista degli ospedali disponibili
    hospital_names = [
        "ST ROSE HOSPITAL",
        "ST MARYS MEDICAL CENTER", 
        "YUMA DISTRICT HOSPITAL",
        "GOOD SAMARITAN REGIONAL HLTH CENTER"
    ]
    
    selected_hospital = st.selectbox("Seleziona un ospedale per vedere i temi delle recensioni:", hospital_names, key="nlp_hosp_theme")
    
    # Slider per controllare il numero di recensioni da analizzare
    max_reviews = st.slider(
        "Numero massimo di recensioni da analizzare:",
        min_value=5,
        max_value=30,
        value=15,
        help="Limita il numero per ottimizzare le prestazioni"
    )
    
    if st.button("Crea sottografo temi per questo ospedale"):
        with st.spinner(f"Analizzando {max_reviews} recensioni..."):
            G = analyzer.build_hospital_theme_graph_gemini(reviews_df, selected_hospital, max_reviews)
            fig = analyzer.plot_hospital_theme_graph(G, selected_hospital)
            st.plotly_chart(fig, use_container_width=True)
            st.info("Spessore arco: proporzionale al numero di recensioni che menzionano il tema.")
 