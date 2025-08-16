from flask import Flask, jsonify
from kg_builder import KnowledgeGraph

app = Flask(__name__)
kg = KnowledgeGraph("data/Hospital General Information.csv")

@app.route('/get_hospitals', methods=['GET'])
def get_hospitals():
    hospitals = [n for n, attr in kg.get_graph().nodes(data=True) if attr["type"] == "Hospital"]
    return jsonify(hospitals)

@app.route('/get_services/<hospital>', methods=['GET'])
def get_services(hospital):
    services = [n for n in kg.get_graph().neighbors(hospital) if kg.get_graph().nodes[n]["type"] == "Service"]
    return jsonify(services)

@app.route('/get_reviews/<service>', methods=['GET'])
def get_reviews(service):
    reviews = [n for n in kg.get_graph().neighbors(service) if kg.get_graph().nodes[n]["type"] == "Review"]
    return jsonify(reviews)

@app.route('/get_doctors/<hospital>', methods=['GET'])
def get_doctors(hospital):
    doctors = [n for n in kg.get_graph().neighbors(hospital) if kg.get_graph().nodes[n]["type"] == "Doctor"]
    return jsonify(doctors)

@app.route('/get_graph_data', methods=['GET'])
def get_graph_data():
    """Restituisce i dati del grafo in formato JSON per la visualizzazione"""
    graph = kg.get_graph()
    
    nodes = []
    edges = []
    
    for node, attr in graph.nodes(data=True):
        nodes.append({
            "id": node,
            "type": attr.get("type", "Unknown"),
            "label": node
        })
    
    for source, target, attr in graph.edges(data=True):
        edges.append({
            "source": source,
            "target": target,
            "relation": attr.get("relation", "connected")
        })
    
    return jsonify({
        "nodes": nodes,
        "edges": edges
    })

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

