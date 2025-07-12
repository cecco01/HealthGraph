"""
Analisi NLP delle Recensioni Ospedali
====================================

Questo modulo gestisce l'analisi del linguaggio naturale delle recensioni degli ospedali
utilizzando Google Gemini 1.5 Flash.

Funzionalità:
- Analisi del sentiment delle recensioni
- Estrazione dei temi principali
- Riassunto delle recensioni
- Analisi avanzata con AI
"""

import os
import pandas as pd
import streamlit as st
import time
import json
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import re
from collections import Counter
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
    
    def analyze_sentiment_gemini(self, text: str) -> Dict:
        """
        Analizza il sentiment usando Gemini
        
        Args:
            text: Testo da analizzare
            
        Returns:
            Dizionario con sentiment e punteggio
        """
        if not self.model:
            return {"sentiment": "neutral", "score": 0, "error": "Gemini non configurato"}
        
        try:
            prompt = f"""
            Analizza il sentiment della seguente recensione di un ospedale.
            Rispondi SOLO con un JSON valido nel formato esatto:
            {{"sentiment": "positive/negative/neutral", "score": 0.0-1.0, "confidence": 0.0-1.0}}
            
            Regole:
            - "positive" se la recensione è positiva/soddisfatta
            - "negative" se la recensione è negativa/insoddisfatta  
            - "neutral" se la recensione è neutrale/mista
            - score: 0.0-1.0 (0=molto negativo, 1=molto positivo)
            - confidence: 0.0-1.0 (quanto sei sicuro dell'analisi)
            
            Recensione: "{text}"
            """
            
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Pulisci il JSON se necessario
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            result = json.loads(response_text.strip())
            
            # Valida il risultato
            valid_sentiments = ['positive', 'negative', 'neutral']
            if result.get('sentiment') not in valid_sentiments:
                result['sentiment'] = 'neutral'
            
            if not isinstance(result.get('score'), (int, float)):
                result['score'] = 0.0
            if not isinstance(result.get('confidence'), (int, float)):
                result['confidence'] = 0.5
                
            return result
            
        except Exception as e:
            # Fallback con analisi basata su parole chiave
            return self._fallback_sentiment_analysis(text)
    
    def _fallback_sentiment_analysis(self, text: str) -> Dict:
        """
        Analisi sentiment di fallback basata su parole chiave
        """
        text_lower = text.lower()
        
        # Parole positive
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
                         'satisfied', 'happy', 'pleased', 'thank', 'thanks', 'love', 'like',
                         'helpful', 'caring', 'professional', 'clean', 'organized', 'efficient']
        
        # Parole negative  
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'disappointed',
                         'unhappy', 'angry', 'frustrated', 'hate', 'dislike', 'poor',
                         'rude', 'unprofessional', 'dirty', 'disorganized', 'slow', 'expensive']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = 'positive'
            score = min(0.8, 0.3 + (positive_count * 0.1))
        elif negative_count > positive_count:
            sentiment = 'negative'
            score = max(-0.8, -0.3 - (negative_count * 0.1))
        else:
            sentiment = 'neutral'
            score = 0.0
            
        return {
            'sentiment': sentiment,
            'score': score,
            'confidence': 0.6
        }
    
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
            
            # Analisi sentiment
            sentiment_result = self.analyze_sentiment_gemini(text)
            
            # Estrazione temi
            themes_result = self.extract_themes_gemini(text)
            
            # Salva risultati
            result = {
                'original_text': text,
                'ai_sentiment': sentiment_result.get('sentiment', 'neutral'),
                'ai_score': sentiment_result.get('score', 0),
                'confidence': sentiment_result.get('confidence', 0),
                'themes': themes_result.get('themes', []),
                'main_theme': themes_result.get('main_theme', 'generale'),
                'original_sentiment': 'positive' if row.get('Sentiment Label') == 1 else 'negative' if row.get('Sentiment Label') == 0 else 'neutral',
                'ratings': row.get('Ratings', 0)
            }
            
            results.append(result)
            
            # Aggiorna progresso
            progress_bar.progress((idx + 1) / len(reviews_df))
            time.sleep(0.5)  # Pausa per evitare rate limiting
        
        progress_bar.empty()
        status_text.empty()
        
        return pd.DataFrame(results)
    
    def create_sentiment_comparison_chart(self, results_df: pd.DataFrame) -> go.Figure:
        """
        Crea un grafico di confronto tra sentiment originale e AI
        """
        comparison_data = []
        for _, row in results_df.iterrows():
            original = row['original_sentiment']
            ai = row['ai_sentiment']
            comparison_data.append({
                'Original': original,
                'AI': ai,
                'Match': original == ai
            })
        
        comp_df = pd.DataFrame(comparison_data)
        
        fig = go.Figure()
        
        # Sentiment Originale
        original_counts = comp_df['Original'].value_counts()
        if not original_counts.empty:
            fig.add_trace(go.Bar(
                x=original_counts.index,
                y=original_counts.values,
                name='Sentiment Originale',
                marker_color='lightblue'
            ))
        
        # Sentiment AI
        ai_counts = comp_df['AI'].value_counts()
        if not ai_counts.empty:
            fig.add_trace(go.Bar(
                x=ai_counts.index,
                y=ai_counts.values,
                name='Sentiment AI',
                marker_color='orange'
            ))
        
        fig.update_layout(
            title='Confronto Sentiment: Originale vs AI',
            xaxis_title='Sentiment',
            yaxis_title='Numero di Recensioni',
            barmode='group',
            height=400
        )
        
        return fig
    
    def create_rating_vs_sentiment_chart(self, results_df: pd.DataFrame) -> go.Figure:
        """
        Crea un grafico rating vs sentiment
        """
        fig = go.Figure()
        
        for sentiment in ['positive', 'negative', 'neutral']:
            subset = results_df[results_df['ai_sentiment'] == sentiment]
            if not subset.empty:
                fig.add_trace(go.Scatter(
                    x=subset['ratings'],
                    y=subset['ai_score'],
                    mode='markers',
                    name=f'Sentiment {sentiment.capitalize()}',
                    marker=dict(size=8)
                ))
        
        fig.update_layout(
            title='Rating vs Sentiment Score',
            xaxis_title='Rating Originale',
            yaxis_title='Sentiment Score AI',
            height=400
        )
        
        return fig
    
    def build_hospital_theme_graph_gemini(self, reviews_df: pd.DataFrame, max_reviews: int = 20) -> nx.Graph:
        """
        Costruisce un sottografo che collega l'ospedale ai temi estratti dalle sue recensioni
        
        Args:
            reviews_df: DataFrame con le recensioni
            max_reviews: Numero massimo di recensioni da analizzare
            
        Returns:
            NetworkX Graph con ospedale e temi collegati
        """
        if not self.model:
            return nx.Graph()
        
        # Usa tutte le recensioni disponibili
        hosp_reviews = reviews_df.copy()
        
        if len(hosp_reviews) > max_reviews:
            hosp_reviews = hosp_reviews.head(max_reviews)
        
        # Analizza sentiment e temi
        theme_sentiments = {}
        theme_counts = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, row in hosp_reviews.iterrows():
            status_text.text(f"Analizzando recensione {idx+1}/{len(hosp_reviews)}...")
            
            text = row['Feedback']
            
            # Analizza sentiment
            sentiment_result = self.analyze_sentiment_gemini(text)
            sentiment = sentiment_result.get('sentiment', 'neutral')
            
            # Mappa sentiment
            if sentiment == 'positive':
                sentiment = 'positivo'
            elif sentiment == 'negative':
                sentiment = 'negativo'
            else:
                sentiment = 'neutro'
            
            # Estrai temi
            themes_result = self.extract_themes_gemini(text)
            themes = themes_result.get('themes', [])
            
            # Aggiorna statistiche temi
            for theme in themes:
                if theme not in theme_sentiments:
                    theme_sentiments[theme] = []
                    theme_counts[theme] = 0
                theme_sentiments[theme].append(sentiment)
                theme_counts[theme] += 1
            
            # Se non trova temi, usa "qualita_medica"
            if not themes:
                if 'qualita_medica' not in theme_sentiments:
                    theme_sentiments['qualita_medica'] = []
                    theme_counts['qualita_medica'] = 0
                theme_sentiments['qualita_medica'].append(sentiment)
                theme_counts['qualita_medica'] += 1
            
            # Aggiorna progresso
            progress_bar.progress((idx + 1) / len(hosp_reviews))
            time.sleep(0.5)
        
        progress_bar.empty()
        
        # Costruisci il grafo
        G = nx.Graph()
        G.add_node('Ospedale', type='Hospital')
        
        # Aggiungi temi principali
        for theme, sentiments in theme_sentiments.items():
            if theme_counts[theme] > 0:
                G.add_node(theme, type='Theme')
                
                # Calcola sentiment medio
                sentiment_map = {'positivo': 1, 'neutro': 0, 'negativo': -1}
                sentiment_vals = [sentiment_map.get(s, 0) for s in sentiments]
                avg_sentiment = sum(sentiment_vals) / len(sentiment_vals) if sentiment_vals else 0
                
                G.add_edge('Ospedale', theme, avg_sentiment=avg_sentiment, count=theme_counts[theme])
        
        # Aggiungi nodi tematici generali se il grafo è troppo piccolo
        if len(G.nodes()) < 5:
            general_themes = ['personale', 'pulizia', 'tempi_attesa', 'costi', 'qualita_medica', 'accoglienza']
            for theme in general_themes:
                if theme not in G.nodes():
                    G.add_node(theme, type='GeneralTheme')
                    G.add_edge('Ospedale', theme, avg_sentiment=0, count=1)
        
        return G
    
    def plot_hospital_theme_graph(self, G: nx.Graph):
        """
        Visualizza il sottografo Ospedale ↔ Temi con Plotly.
        Colore arco: verde (positivo), rosso (negativo), grigio (neutro).
        Spessore arco: proporzionale a count.
        """
        pos = nx.spring_layout(G, k=1, iterations=50)
        edge_x = []
        edge_y = []
        edge_colors = []
        edge_widths = []
        
        for u, v, data in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            
            s = data.get('avg_sentiment', 0)
            if s > 0.2:
                color = 'green'
            elif s < -0.2:
                color = 'red'
            else:
                color = 'gray'
            
            edge_colors.append(color)
            edge_widths.append(2 + 3 * abs(s))
        
        fig = go.Figure()
        for i in range(len(edge_x)//3):
            fig.add_trace(go.Scatter(
                x=edge_x[i*3:i*3+2],
                y=edge_y[i*3:i*3+2],
                mode='lines',
                line=dict(width=edge_widths[i], color=edge_colors[i]),
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
            title='Sottografo Temi Recensioni',
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

def display_nlp_analysis_page():
    """
    Mostra la pagina di analisi NLP nell'interfaccia Streamlit
    """
    st.header("Analisi NLP delle Recensioni")
    
    # Inizializza l'analizzatore
    analyzer = get_analyzer()
    
    # Carica i dati
    reviews_df = analyzer.load_reviews_data()
    
    if reviews_df.empty:
        st.error("Impossibile caricare i dati delle recensioni")
        return
    
    # Mostra statistiche base
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Recensioni Totali", len(reviews_df))
    with col2:
        st.metric("Sentiment Positivi", len(reviews_df[reviews_df['Sentiment Label'] == 1]))
    with col3:
        st.metric("Rating Medio", f"{reviews_df['Ratings'].mean():.1f}")
    
    # Configurazione Analisi
    st.subheader("Configurazione Analisi")
    
    # Numero di recensioni da analizzare
    max_reviews = st.slider(
        "Numero di recensioni da analizzare:",
        min_value=5,
        max_value=min(50, len(reviews_df)),
        value=20,
        help="Limita il numero per ottimizzare le prestazioni"
    )
    
    # Pulsante per avviare l'analisi
    if st.button("Avvia Analisi NLP", type="primary"):
        if not analyzer.model:
            st.error("Gemini non configurato. Aggiungi la chiave API GEMINI_API_KEY nel file api_keys.env")
            return
        
        # Avvia l'analisi
        with st.spinner("Analizzando le recensioni..."):
            results_df = analyzer.analyze_reviews_batch(reviews_df, max_reviews)
        
        # Salva i risultati nella session state
        st.session_state.nlp_results = results_df
        
        st.success("Analisi completata!")
    
    # Mostra i risultati se disponibili
    if 'nlp_results' in st.session_state:
        st.subheader("Risultati dell'Analisi")
        
        results_df = st.session_state.nlp_results
        
        # Statistiche generali
        st.markdown("**Statistiche dell'Analisi:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Recensioni Analizzate", len(results_df))
        with col2:
            st.metric("Modello Utilizzato", "Gemini 1.5 Flash")
        with col3:
            accuracy = len(results_df[results_df['original_sentiment'] == results_df['ai_sentiment']]) / len(results_df) * 100
            st.metric("Accuratezza", f"{accuracy:.1f}%")
        
        # Grafici
        tab1, tab2, tab3 = st.tabs(["Confronto Sentiment", "Rating vs Sentiment", "Dettagli"])
        
        with tab1:
            fig_sentiment = analyzer.create_sentiment_comparison_chart(results_df)
            st.plotly_chart(fig_sentiment, use_container_width=True)
        
        with tab2:
            fig_rating = analyzer.create_rating_vs_sentiment_chart(results_df)
            st.plotly_chart(fig_rating, use_container_width=True)
        
        with tab3:
            st.dataframe(results_df[['original_text', 'ai_sentiment', 'ai_score']])
        
        # Download dei risultati
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Scarica Risultati (CSV)",
            data=csv,
            file_name="nlp_analysis_results_gemini.csv",
            mime="text/csv"
        )

def display_hospital_theme_graph_gemini():
    """
    Mostra la sezione del sottografo temi recensioni
    """
    st.markdown("---")
    st.subheader("Sottografo Temi Recensioni")
    
    analyzer = get_analyzer()
    
    if not analyzer.model:
        st.error("Gemini non configurato. Aggiungi la chiave API GEMINI_API_KEY nel file api_keys.env")
        return
    
    # Carica le recensioni
    reviews_df = analyzer.load_reviews_data()
    if reviews_df.empty:
        st.warning("Non ci sono dati di recensioni disponibili.")
        return
    
    # Slider per controllare il numero di recensioni da analizzare
    max_reviews = st.slider(
        "Numero massimo di recensioni da analizzare:",
        min_value=5,
        max_value=30,
        value=15,
        help="Limita il numero per ottimizzare le prestazioni"
    )
    
    if st.button("Crea sottografo temi"):
        with st.spinner(f"Analizzando {max_reviews} recensioni..."):
            G = analyzer.build_hospital_theme_graph_gemini(reviews_df, max_reviews)
            fig = analyzer.plot_hospital_theme_graph(G)
            st.plotly_chart(fig, use_container_width=True)
            st.info("Colore arco: verde=positivo, rosso=negativo, grigio=neutro. Spessore=importanza del tema.")
 