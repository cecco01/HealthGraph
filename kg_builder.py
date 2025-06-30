"""
Healthcare Knowledge Graph - Costruttore del Grafo
===============================================

Questo file implementa la classe KnowledgeGraph che costruisce e gestisce il grafo di conoscenza
degli ospedali. Il grafo rappresenta le relazioni tra ospedali, servizi, stati e tipi di strutture.

Funzionalità principali:
- Costruzione del grafo dai dati degli ospedali
- Gestione dei nodi e degli archi
- Assegnazione casuale dei servizi
- Esportazione del grafo in formato JSON
"""

import networkx as nx
import pandas as pd
import json
import random

class KnowledgeGraph:
    """
    Classe per la costruzione e gestione del grafo di conoscenza degli ospedali.
    
    Il grafo rappresenta le relazioni tra:
    - Ospedali
    - Servizi medici
    - Stati
    - Tipi di strutture
    - Valutazioni
    """
    
    def __init__(self, hospital_data_path):
        """
        Inizializza il grafo di conoscenza.
        
        Args:
            hospital_data_path (str): Percorso al file CSV dei dati degli ospedali
        """
        self.graph = nx.Graph()
        self.hospital_data = pd.read_csv(hospital_data_path, encoding='latin1')  # o 'cp1252'
        self._build_graph()

    def _build_graph(self):
        """
        Costruisce il grafo di conoscenza dagli ospedali
        """
        # Crea un set di servizi una sola volta
        services = ['Cardiologia', 'Neurologia', 'Pediatria', 'Ortopedia', 'Oncologia', 
                   'Chirurgia', 'Dermatologia', 'Oftalmologia', 'Psichiatria', 'Urologia']
        
        # Aggiungi i nodi per i servizi
        for service in services:
            self.graph.add_node(service, type='Service')
        
        # Aggiungi i nodi per gli ospedali
        for _, row in self.hospital_data.iterrows():
            hospital_name = row['Hospital Name']
            self.graph.add_node(hospital_name, type='Hospital')
            
            # Aggiungi gli attributi dell'ospedale
            for attr in row.index:
                if pd.notna(row[attr]):
                    self.graph.nodes[hospital_name][attr] = row[attr]
            
            # Assegna servizi casuali a ogni ospedale (3-7 servizi per ospedale)
            num_services = random.randint(3, 7)
            selected_services = random.sample(services, num_services)
            for service in selected_services:
                self.graph.add_edge(hospital_name, service, relation='offers')

    def get_hospitals(self):
        """
        Restituisce tutti gli ospedali nel grafo
        """
        return [n for n, attr in self.graph.nodes(data=True) if attr.get('type') == 'Hospital']
    
    def get_services(self, hospital=None):
        """
        Restituisce i servizi di un ospedale o tutti i servizi
        """
        if hospital:
            return [n for n in self.graph.neighbors(hospital) if self.graph.nodes[n]['type'] == 'Service']
        return [n for n, attr in self.graph.nodes(data=True) if attr.get('type') == 'Service']

    def get_hospital_info(self, hospital):
        """
        Restituisce le informazioni di un ospedale
        """
        if hospital in self.graph.nodes:
            return dict(self.graph.nodes[hospital])
        return None

    def export_graph(self, file_path):
        """
        Esporta il grafo in formato JSON
        """
        data = {
            'nodes': [
                {'id': n, **attr} for n, attr in self.graph.nodes(data=True)
            ],
            'edges': [
                {'source': u, 'target': v, 'relation': d.get('relation', 'connected')}
                for u, v, d in self.graph.edges(data=True)
            ]
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
