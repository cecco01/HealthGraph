"""
Healthcare Knowledge Graph - Interfaccia Utente
=============================================

Questo file implementa l'interfaccia utente dell'applicazione Healthcare Knowledge Graph
utilizzando Streamlit. L'applicazione permette di visualizzare e analizzare i dati degli
ospedali attraverso un grafo di conoscenza interattivo.

Funzionalità principali:
- Visualizzazione del grafo di conoscenza degli ospedali
- Filtrazione per tipo di ospedale e stato
- Analisi dei dati degli ospedali
- Visualizzazione dei servizi offerti
- Clustering e layout personalizzabili
"""

import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from kg_builder import KnowledgeGraph
import time
import numpy as np
from sklearn.cluster import KMeans
import random

# =============================================================================
# Configurazione Iniziale
# =============================================================================

# Configurazione della pagina Streamlit
st.set_page_config(
    page_title="Healthcare Knowledge Graph",
    page_icon="🏥",
    layout="wide"
)

# Titolo principale dell'applicazione
st.title("Healthcare Knowledge Graph")

# =============================================================================
# Funzioni di Caricamento Dati
# =============================================================================

@st.cache_data
def load_data():
    """
    Carica i dati degli ospedali dal file CSV.
    Utilizza la cache di Streamlit per evitare ricaricamenti non necessari.
    
    Returns:
        DataFrame: Dati degli ospedali
    """
    return pd.read_csv("data/Hospital General Information.csv", encoding='latin1')

@st.cache_resource
def init_knowledge_graph():
    """
    Inizializza il grafo di conoscenza.
    Utilizza la cache di Streamlit per mantenere il grafo in memoria.
    
    Returns:
        KnowledgeGraph: Istanza del grafo di conoscenza
    """
    with st.spinner("Creazione del grafo di conoscenza in corso..."):
        time.sleep(0.5)  # Ritardo per mostrare lo spinner
        return KnowledgeGraph("data/Hospital General Information.csv")

# =============================================================================
# Mappa inversa servizio -> ospedali (per ottimizzazione sottografo)
# =============================================================================
@st.cache_data
def build_service_to_hospitals(_kg):
    service_to_hospitals = {}
    for hospital in _kg.get_hospitals():
        for service in _kg.get_services(hospital):
            if service not in service_to_hospitals:
                service_to_hospitals[service] = set()
            service_to_hospitals[service].add(hospital)
    return service_to_hospitals

# =============================================================================
# Inizializzazione Dati e Stato
# =============================================================================

# Carica i dati e inizializza il grafo
data = load_data()
kg = init_knowledge_graph()
service_to_hospitals = build_service_to_hospitals(kg)

# Inizializza le variabili di stato per i filtri e le opzioni di visualizzazione
if 'selected_type' not in st.session_state:
    st.session_state.selected_type = "Tutti"
if 'selected_state' not in st.session_state:
    st.session_state.selected_state = "Tutti"
if 'layout_type' not in st.session_state:
    st.session_state.layout_type = "Circolare"
if 'show_labels' not in st.session_state:
    st.session_state.show_labels = True
if 'clustering_method' not in st.session_state:
    st.session_state.clustering_method = "Nessuno"
if 'max_edges_per_node' not in st.session_state:
    st.session_state.max_edges_per_node = 5

# =============================================================================
# Interfaccia Utente - Sidebar
# =============================================================================

# Selezione della pagina
page = st.sidebar.selectbox(
    "Seleziona una pagina",
    ["Panoramica e Analisi", "Visualizzazione Grafo"]
)

# Filtri nella sidebar
st.sidebar.subheader("Filtri")

# Ottieni le liste dei tipi e stati unici
hospital_types = ["Tutti"] + list(data['Hospital Type'].unique())
states = ["Tutti"] + list(data['State'].unique())

# Trova gli indici corretti per i filtri selezionati
type_index = 0
if st.session_state.selected_type in hospital_types:
    type_index = hospital_types.index(st.session_state.selected_type)

state_index = 0
if st.session_state.selected_state in states:
    state_index = states.index(st.session_state.selected_state)

# Aggiorna i filtri nella sidebar
st.session_state.selected_type = st.sidebar.selectbox(
    "Tipo di Ospedale", 
    hospital_types,
    index=type_index
)
st.session_state.selected_state = st.sidebar.selectbox(
    "Stato", 
    states,
    index=state_index
)

# =============================================================================
# Filtrazione Dati
# =============================================================================

# Applica i filtri ai dati
filtered_data = data
if st.session_state.selected_type != "Tutti":
    filtered_data = filtered_data[filtered_data['Hospital Type'] == st.session_state.selected_type]
if st.session_state.selected_state != "Tutti":
    filtered_data = filtered_data[filtered_data['State'] == st.session_state.selected_state]

# =============================================================================
# Pagina Panoramica e Analisi
# =============================================================================

if page == "Panoramica e Analisi":
    st.header("Panoramica e Analisi Ospedali")
    
    # Metriche principali in colonne
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Numero di Ospedali", len(filtered_data))
    with col2:
        st.metric("Numero di Servizi", len(kg.get_services()))
    with col3:
        st.metric("Tipi di Ospedale", len(filtered_data['Hospital Type'].unique()))

    # =====================
    # DASHBOARD AVANZATA
    # =====================
    st.subheader("Dashboard Statistiche Avanzate")
    dash1, dash2 = st.columns(2)
    
    # Ospedali per Stato (bar chart)
    with dash1:
        st.markdown("**Ospedali per Stato**")
        state_counts = filtered_data['State'].value_counts().sort_values(ascending=False)
        fig_state = go.Figure(data=[go.Bar(x=state_counts.index, y=state_counts.values)])
        fig_state.update_layout(xaxis_title="Stato", yaxis_title="Numero di Ospedali", height=350)
        st.plotly_chart(fig_state, use_container_width=True)
    
    # Distribuzione Tipi di Ospedale (pie chart)
    with dash2:
        st.markdown("**Distribuzione Tipi di Ospedale**")
        type_counts = filtered_data['Hospital Type'].value_counts()
        fig_type = go.Figure(data=[go.Pie(labels=type_counts.index, values=type_counts.values, hole=0.4)])
        fig_type.update_layout(height=350)
        st.plotly_chart(fig_type, use_container_width=True)

    # Ospedale con più servizi
    hospital_service_counts = {h: len(kg.get_services(h)) for h in kg.get_hospitals()}
    if hospital_service_counts:
        top_hospital = max(hospital_service_counts, key=hospital_service_counts.get)
        st.info(f"🏥 **Ospedale con più servizi:** {top_hospital} ({hospital_service_counts[top_hospital]} servizi)")

    # Stato con più ospedali
    if not state_counts.empty:
        top_state = state_counts.idxmax()
        st.info(f"📍 **Stato con più ospedali:** {top_state} ({state_counts.max()} ospedali)")

    # Distribuzione dei rating (se disponibile)
    if 'Hospital overall rating' in filtered_data.columns:
        st.markdown("**Distribuzione dei Rating**")
        rating_counts = filtered_data['Hospital overall rating'].value_counts().sort_index()
        fig_rating = go.Figure(data=[go.Bar(x=rating_counts.index.astype(str), y=rating_counts.values)])
        fig_rating.update_layout(xaxis_title="Rating", yaxis_title="Numero di Ospedali", height=300)
        st.plotly_chart(fig_rating, use_container_width=True)

    # Visualizzazione dei dati in formato tabella
    st.subheader("Dati Ospedali")
    columns_to_show = [
    'Hospital Name',
    'Address',
    'City',
    'State',
    'ZIP Code',
    'County Name',
    'Phone Number',
    'Hospital Type',
    'Hospital Ownership',
    'Emergency Services',
    'Hospital overall rating'
        ]
    st.dataframe(filtered_data[columns_to_show])

    # =====================
    # CONFRONTO DETTAGLIATO TRA OSPEDALI
    # =====================
    st.markdown("---")
    st.subheader("Confronto Dettagliato tra Ospedali")
    hospital_names = filtered_data['Hospital Name'].unique().tolist()
    if len(hospital_names) >= 2:
        colA, colB = st.columns(2)
        with colA:
            hospital1 = st.selectbox("Seleziona il primo ospedale", hospital_names, key="confronto_hosp1")
        # Filtra ospedali dello stesso stato e tipo
        info1 = kg.get_hospital_info(hospital1)
        same_state_type = filtered_data[(filtered_data['State'] == info1.get('State')) & (filtered_data['Hospital Type'] == info1.get('Hospital Type'))]['Hospital Name'].tolist()
        same_state_type = [h for h in same_state_type if h != hospital1]
        # Calcola l'ospedale più simile per servizi
        services1 = set(kg.get_services(hospital1))
        similarity = []
        for h in same_state_type:
            services2 = set(kg.get_services(h))
            n_common = len(services1 & services2)
            similarity.append((h, n_common))
        similarity.sort(key=lambda x: x[1], reverse=True)
        most_similar = [h for h, n in similarity[:1] if n > 0]
        with colB:
            hospital2 = st.selectbox("Ospedale più simile per servizi", most_similar, key="confronto_hosp2")
        if hospital1 and hospital2:
            info2 = kg.get_hospital_info(hospital2)
            # Visualizzazione affiancata delle info principali
            st.markdown("**Confronto Caratteristiche Principali**")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"### {hospital1}")
                st.write(f"**Stato:** {info1.get('State','-')}")
                st.write(f"**Tipo:** {info1.get('Hospital Type','-')}")
                st.write(f"**Rating:** {info1.get('Hospital overall rating','-')}")
                st.write(f"**Servizi:** {', '.join(kg.get_services(hospital1))}")
            with c2:
                st.markdown(f"### {hospital2}")
                st.write(f"**Stato:** {info2.get('State','-')}")
                st.write(f"**Tipo:** {info2.get('Hospital Type','-')}")
                st.write(f"**Rating:** {info2.get('Hospital overall rating','-')}")
                st.write(f"**Servizi:** {', '.join(kg.get_services(hospital2))}")
            # Grafico radar per confronto servizi
            st.markdown("**Confronto Servizi Offerti (Radar Chart)**")
            all_services = sorted(list(set(kg.get_services(hospital1)) | set(kg.get_services(hospital2))))
            values1 = [1 if s in kg.get_services(hospital1) else 0 for s in all_services]
            values2 = [1 if s in kg.get_services(hospital2) else 0 for s in all_services]
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=values1, theta=all_services, fill='toself', name=hospital1))
            fig_radar.add_trace(go.Scatterpolar(r=values2, theta=all_services, fill='toself', name=hospital2))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=True)
            st.plotly_chart(fig_radar, use_container_width=True)
        elif not most_similar:
            st.info("Non ci sono altri ospedali dello stesso stato e tipo con servizi in comune per il confronto.")
    else:
        st.info("Sono necessari almeno due ospedali per il confronto.")

# =============================================================================
# Pagina Visualizzazione Grafo
# =============================================================================

else:  # Visualizzazione Grafo
    st.header("Visualizzazione Grafo")
    
    # Crea un sottografo con solo gli ospedali filtrati
    G = kg.graph
    filtered_hospitals = set(filtered_data['Hospital Name'].values)
    
    # Ottieni tutti i nodi e gli archi relativi agli ospedali filtrati
    nodes_to_keep = set()
    edges_to_keep = set()
    
    for hospital in filtered_hospitals:
        if hospital in G:
            nodes_to_keep.add(hospital)
            # Aggiungi tutti i nodi dei servizi collegati
            for service in kg.get_services(hospital):
                nodes_to_keep.add(service)
                edges_to_keep.add((hospital, service))
    
    # Crea il sottografo
    subgraph = G.subgraph(nodes_to_keep)
    
    # Gestione del caso in cui il sottografo è vuoto
    if len(subgraph) == 0:
        st.warning("Nessun ospedale trovato nel grafo per i filtri selezionati. Prova a selezionare filtri diversi.")
    else:
        # =============================================================================
        # Opzioni di Visualizzazione
        # =============================================================================
        
        st.sidebar.subheader("Opzioni di Visualizzazione")
        
        # Layout e etichette
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.session_state.layout_type = st.selectbox(
                "Tipo di Layout",
                ["Circolare", "Kamada-Kawai", "Fruchterman-Reingold", "Spring", "Spectral", "Shell"],
                index=["Circolare", "Kamada-Kawai", "Fruchterman-Reingold", "Spring", "Spectral", "Shell"].index(st.session_state.layout_type)
            )
        with col2:
            st.session_state.show_labels = st.checkbox("Mostra Etichette", value=st.session_state.show_labels)
        
        # Clustering e densità
        st.sidebar.subheader("Gestione Densitá")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.session_state.clustering_method = st.selectbox(
                "Metodo di Clustering",
                ["Nessuno", "Per Stato", "Per Tipo", "K-Means"],
                index=["Nessuno", "Per Stato", "Per Tipo", "K-Means"].index(st.session_state.clustering_method)
            )
        with col2:
            st.session_state.max_edges_per_node = st.slider(
                "Max Archi per Nodo", 
                min_value=1, 
                max_value=10, 
                value=st.session_state.max_edges_per_node,
                help="Limita il numero di archi per nodo per ridurre la densità del grafo"
            )
        
        # =============================================================================
        # Applicazione Clustering
        # =============================================================================
        
        if st.session_state.clustering_method != "Nessuno":
            # Crea un grafo di clustering
            cluster_graph = nx.Graph()
            
            # Aggiungi i nodi per gli ospedali
            for hospital in filtered_hospitals:
                if hospital in G:
                    # Ottieni lo stato e il tipo dell'ospedale
                    state = G.nodes[hospital].get('State', 'Unknown')
                    hospital_type = G.nodes[hospital].get('Hospital Type', 'Unknown')
                    
                    # Aggiungi il nodo dell'ospedale
                    cluster_graph.add_node(hospital, type='Hospital', state=state, hospital_type=hospital_type)
                    
                    # Aggiungi il nodo del cluster in base al metodo selezionato
                    if st.session_state.clustering_method == "Per Stato":
                        cluster_name = f"Stato: {state}"
                        cluster_graph.add_node(cluster_name, type='Cluster')
                        cluster_graph.add_edge(hospital, cluster_name, relation='belongs_to')
                    elif st.session_state.clustering_method == "Per Tipo":
                        cluster_name = f"Tipo: {hospital_type}"
                        cluster_graph.add_node(cluster_name, type='Cluster')
                        cluster_graph.add_edge(hospital, cluster_name, relation='belongs_to')
                    elif st.session_state.clustering_method == "K-Means":
                        # Usa K-Means per raggruppare gli ospedali
                        pos = nx.spring_layout(subgraph)
                        X = np.array([pos[n] for n in subgraph.nodes() if subgraph.nodes[n]['type'] == 'Hospital'])
                        if len(X) > 0:
                            k = min(5, len(X))
                            kmeans = KMeans(n_clusters=k, random_state=42)
                            labels = kmeans.fit_predict(X)
                            
                            # Aggiungi i nodi dei cluster
                            hospital_nodes = [n for n in subgraph.nodes() if subgraph.nodes[n]['type'] == 'Hospital']
                            for i, hospital in enumerate(hospital_nodes):
                                if i < len(labels):
                                    cluster_name = f"Cluster {labels[i]}"
                                    cluster_graph.add_node(cluster_name, type='Cluster')
                                    cluster_graph.add_edge(hospital, cluster_name, relation='belongs_to')
            
            # Aggiungi i nodi per i servizi
            for service in kg.get_services():
                cluster_graph.add_node(service, type='Service')
                
                # Collega i servizi agli ospedali
                for hospital in filtered_hospitals:
                    if hospital in G and service in kg.get_services(hospital):
                        cluster_graph.add_edge(hospital, service, relation='offers')
            
            # Usa il grafo di clustering
            subgraph = cluster_graph
        
        # =============================================================================
        # Limitazione Archi
        # =============================================================================
        
        if st.session_state.max_edges_per_node < 10:
            # Crea un nuovo grafo con archi limitati
            limited_graph = nx.Graph()
            
            # Aggiungi tutti i nodi
            for node in subgraph.nodes():
                limited_graph.add_node(node, **subgraph.nodes[node])
            
            # Aggiungi solo un numero limitato di archi per nodo
            for node in subgraph.nodes():
                edges = list(subgraph.edges(node))
                if len(edges) > st.session_state.max_edges_per_node:
                    # Seleziona un sottoinsieme casuale di archi
                    edge_indices = list(range(len(edges)))
                    selected_indices = random.sample(edge_indices, st.session_state.max_edges_per_node)
                    selected_edges = [edges[i] for i in selected_indices]
                    
                    for edge in selected_edges:
                        limited_graph.add_edge(edge[0], edge[1], **subgraph.edges[edge])
                else:
                    # Aggiungi tutti gli archi
                    for edge in edges:
                        limited_graph.add_edge(edge[0], edge[1], **subgraph.edges[edge])
            
            # Usa il grafo limitato
            subgraph = limited_graph
        
        # =============================================================================
        # Generazione Layout
        # =============================================================================
        
        with st.spinner("Generazione del layout del grafo..."):
            if st.session_state.layout_type == "Circolare":
                pos = nx.circular_layout(subgraph)
            elif st.session_state.layout_type == "Kamada-Kawai":
                pos = nx.kamada_kawai_layout(subgraph)
            elif st.session_state.layout_type == "Fruchterman-Reingold":
                pos = nx.fruchterman_reingold_layout(subgraph, k=2, iterations=50)
            elif st.session_state.layout_type == "Spring":
                pos = nx.spring_layout(subgraph, k=2, iterations=50)
            elif st.session_state.layout_type == "Spectral":
                pos = nx.spectral_layout(subgraph)
            else:  # Shell
                pos = nx.shell_layout(subgraph)
        
        # =============================================================================
        # Visualizzazione Grafo
        # =============================================================================
        
        # Crea la figura
        fig = go.Figure()
        
        # Definisci i colori per i diversi tipi di nodi
        node_colors = {
            'Hospital': '#1f77b4',  # blu
            'Service': '#2ca02c',   # verde
            'Cluster': '#ff7f0e',   # arancione
            'Unknown': '#7f7f7f'    # grigio
        }
        
        # Aggiungi i nodi al grafo
        for node_type in node_colors:
            node_list = [n for n, attr in subgraph.nodes(data=True) if attr.get('type') == node_type]
            if node_list:
                fig.add_trace(go.Scatter(
                    x=[pos[n][0] for n in node_list],
                    y=[pos[n][1] for n in node_list],
                    mode='markers+text' if st.session_state.show_labels else 'markers',
                    name=node_type,
                    text=[n for n in node_list],
                    textposition="bottom center",
                    textfont=dict(size=10),
                    marker=dict(
                        size=30 if node_type == 'Hospital' else 40 if node_type == 'Cluster' else 20,
                        color=node_colors[node_type],
                        line=dict(width=2, color='white')
                    ),
                    hoverinfo='text',
                    hovertext=[f"{n}<br>Tipo: {node_type}" for n in node_list]
                ))
        
        # Aggiungi gli archi al grafo
        edge_x = []
        edge_y = []
        for edge in subgraph.edges():
            edge_x.extend([pos[edge[0]][0], pos[edge[1]][0], None])
            edge_y.extend([pos[edge[0]][1], pos[edge[1]][1], None])
        
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            name='Connessioni'
        ))
        
        # Configura il layout del grafo
        fig.update_layout(
            title='Knowledge Graph Visualization',
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            dragmode='pan',
            modebar=dict(
                orientation='v',
                bgcolor='rgba(255, 255, 255, 0.7)',
                color='#1f77b4',
                activecolor='#ff7f0e'
            )
        )
        
        # Configura i pulsanti della barra degli strumenti
        config = {
            'scrollZoom': True,
            'displayModeBar': True,
            'modeBarButtonsToAdd': ['zoom2d', 'pan2d', 'resetScale2d'],
            'modeBarButtonsToRemove': ['lasso2d', 'select2d']
        }
        
        # Visualizza il grafo
        st.plotly_chart(fig, use_container_width=True, config=config)
        
        # =============================
        # CATENA DI RELAZIONI - SOTTOGRAFO INTERATTIVO
        # =============================
        st.markdown("---")
        st.subheader("Esplora le Relazioni di un Ospedale (Sottografo Interattivo)")
        hospital_options = [n for n, attr in subgraph.nodes(data=True) if attr.get('type') == 'Hospital']
        if hospital_options:
            selected_hospital = st.selectbox("Seleziona un ospedale per esplorare il suo network", hospital_options, key="sottografo_hospital")
            # Slider per max archi per nodo nel sottografo
            max_edges_sottografo = st.slider(
                "Numero massimo di archi per nodo (solo sottografo)",
                min_value=1,
                max_value=10,
                value=5,
                help="Limita la complessità del sottografo visualizzato"
            )
            if selected_hospital:
                # Trova servizi collegati
                services = set(kg.get_services(selected_hospital))
                # Trova i 5 ospedali più simili per servizi (invece di tutti)
                all_hospitals = [h for h in kg.get_hospitals() if h != selected_hospital]
                similarity = []
                for h in all_hospitals:
                    services2 = set(kg.get_services(h))
                    n_common = len(services & services2)
                    similarity.append((h, n_common))
                similarity.sort(key=lambda x: x[1], reverse=True)
                top_similar_hospitals = [h for h, n in similarity[:5] if n > 0]
                # Costruisci nodi
                nodes = set([selected_hospital]) | services | set(top_similar_hospitals)
                # Costruisci archi senza duplicati
                edges = set()
                for service in services:
                    edges.add((selected_hospital, service))
                    for h in top_similar_hospitals:
                        if service in kg.get_services(h):
                            edges.add((h, service))
                # Costruisci il sottografo
                sottografo = nx.Graph()
                for n in nodes:
                    if n == selected_hospital:
                        sottografo.add_node(n, type='SelectedHospital')
                    elif n in services:
                        sottografo.add_node(n, type='Service')
                    else:
                        sottografo.add_node(n, type='OtherHospital')
                for u, v in edges:
                    sottografo.add_edge(u, v)
                # Limita il numero di archi per nodo nel sottografo
                if max_edges_sottografo < 10:
                    limited_graph = nx.Graph()
                    for node in sottografo.nodes():
                        limited_graph.add_node(node, **sottografo.nodes[node])
                    for node in sottografo.nodes():
                        node_edges = list(sottografo.edges(node))
                        if len(node_edges) > max_edges_sottografo:
                            selected_edges = random.sample(node_edges, max_edges_sottografo)
                            for edge in selected_edges:
                                limited_graph.add_edge(edge[0], edge[1])
                        else:
                            for edge in node_edges:
                                limited_graph.add_edge(edge[0], edge[1])
                    sottografo = limited_graph
                # Layout
                pos = nx.spring_layout(sottografo, k=1, iterations=50)
                # Colori
                node_colors = {
                    'SelectedHospital': '#d62728',  # rosso
                    'OtherHospital': '#1f77b4',    # blu
                    'Service': '#2ca02c'           # verde
                }
                fig_sub = go.Figure()
                for node_type, color in node_colors.items():
                    node_list = [n for n, attr in sottografo.nodes(data=True) if attr.get('type') == node_type]
                    if node_list:
                        fig_sub.add_trace(go.Scatter(
                            x=[pos[n][0] for n in node_list],
                            y=[pos[n][1] for n in node_list],
                            mode='markers+text',
                            name=node_type,
                            text=[n for n in node_list],
                            textposition="bottom center",
                            marker=dict(
                                size=38 if node_type=='SelectedHospital' else 28 if node_type=='OtherHospital' else 22,
                                color=color,
                                line=dict(width=2, color='white')
                            ),
                            hoverinfo='text',
                            hovertext=[f"{n} ({node_type})" for n in node_list]
                        ))
                # Edges
                edge_x = []
                edge_y = []
                for edge in sottografo.edges():
                    edge_x.extend([pos[edge[0]][0], pos[edge[1]][0], None])
                    edge_y.extend([pos[edge[0]][1], pos[edge[1]][1], None])
                fig_sub.add_trace(go.Scatter(
                    x=edge_x, y=edge_y,
                    mode='lines',
                    line=dict(width=1.5, color='#888'),
                    hoverinfo='none',
                    name='Connessioni'
                ))
                fig_sub.update_layout(
                    title=f'Sottografo di {selected_hospital} (Top 5 ospedali simili)',
                    showlegend=True,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    ),
                    dragmode='pan',
                )
                st.plotly_chart(fig_sub, use_container_width=True)
                st.markdown("""
                **Legenda:**
                - <span style='color:#d62728'><b>Ospedale selezionato</b></span>
                - <span style='color:#1f77b4'><b>Top 5 ospedali più simili per servizi</b></span>
                - <span style='color:#2ca02c'><b>Servizi</b></span>
                """, unsafe_allow_html=True)
        else:
            st.info("Nessun ospedale disponibile nel sottografo attuale.")
        
        # =============================
        # METRICHE DI CENTRALITÀ
        # =============================
        st.markdown("---")
        st.subheader("Metriche di Centralità del Grafo")
        degree_centrality = nx.degree_centrality(subgraph)
        betweenness_centrality = nx.betweenness_centrality(subgraph)
        # Mostra i top 5 nodi per ciascuna metrica
        top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        st.markdown("**Top 5 nodi per Degree Centrality:**")
        for n, v in top_degree:
            st.write(f"{n}: {v:.3f}")
        st.markdown("**Top 5 nodi per Betweenness Centrality:**")
        for n, v in top_betweenness:
            st.write(f"{n}: {v:.3f}")
        st.info("Le metriche di centralità aiutano a identificare i nodi più influenti o "
                "strategici nel grafo, utili per analisi di network e decisioni di business.")
        
        # =============================
        # Miglioramento UI: Help contestuale
        # =============================
        st.markdown("---")
        with st.expander("ℹ️ Come leggere il grafo e le statistiche?"):
            st.write("""
            - **Nodi blu**: Ospedali
            - **Nodi verdi**: Servizi
            - **Nodi arancioni**: Cluster (se attivo)
            - **Connessioni**: Relazioni tra ospedali e servizi
            - **Catena di relazioni**: Esplora i servizi di un ospedale e trova altri ospedali simili
            - **Metriche di centralità**: Identifica i nodi più importanti per la connettività e il flusso di informazioni
            """)
        
        # Aggiungi statistiche del grafo nella sidebar
        st.sidebar.subheader("Statistiche del Grafo")
        st.sidebar.write(f"Nodi: {len(subgraph.nodes())}")
        st.sidebar.write(f"Archi: {len(subgraph.edges())}")
        st.sidebar.write(f"Densità: {nx.density(subgraph):.2f}")
        
        # Aggiungi un pulsante per esportare il grafo
        if st.sidebar.button("Esporta Grafo (JSON)"):
            kg.export_graph("graph_export.json")
            st.sidebar.success("Grafo esportato con successo!")
