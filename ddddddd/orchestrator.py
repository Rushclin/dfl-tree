import os
import time
import socket
import threading
from node import Node, elect_leader  # Assurez-vous que Node et elect_leader sont définis dans node.py

def orchestrator():
    # Déterminer les nœuds via des ports définis (ou spécifiés dans le docker-compose)
    node_ports = [5501, 5502, 5503]
    nodes = [Node(port=port) for port in node_ports]

    # Simuler un délai pour s'assurer que tous les nœuds démarrent
    time.sleep(2)

    # Élection du leader
    leader = elect_leader(nodes)
    leader.start_server()

    # Les autres nœuds se connectent au leader et envoient des messages
    threads = []
    for node in nodes:
        if node != leader:
            thread = threading.Thread(target=simulate_client_behavior, args=(node, leader))
            threads.append(thread)
            thread.start()

    # Attendre la fin de tous les threads clients
    for thread in threads:
        thread.join()

    # Arrêter le serveur une fois les tests terminés
    leader.stop_server()

def simulate_client_behavior(node, leader):
    """Comportement simulé d'un nœud client."""
    client_socket = node.connect_to_server(leader.ip, leader.port)
    if client_socket:
        messages = [f"Message from node {node.port}", Node.DISCONNECT_MESSAGE]
        for msg in messages:
            node.send(client_socket, msg)

if __name__ == "__main__":
    orchestrator()
