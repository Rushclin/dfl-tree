from core.tree import Tree
from core.node import Node
from core.mqtt_handler import MQTTHandler

import logging 
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info(f"Init")
    # Initialisation
    mqtt_handler = MQTTHandler(client_id="tree-handler", broker="localhost", port=1883)
    nodes = [Node(node_id=i, mqtt_handler=mqtt_handler) for i in range(1, 6)]  # Crée 5 nœuds

    tree = Tree(mqtt_handler=mqtt_handler, nodes=nodes)

    # Démarrer le système
    mqtt_handler.connect()
    tree.build_tree()

    # Évaluer l'arbre
    tree.evaluate_tree()

    # Élire un nouveau nœud central
    tree.elect_new_root()

    # Arrêter le système
    mqtt_handler.disconnect()
