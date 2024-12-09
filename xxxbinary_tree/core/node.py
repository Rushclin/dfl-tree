import json
from typing import Optional, List
from .mqtt_handler import MQTTHandler

import logging
logger = logging.getLogger(__name__)

class Node:
    """
    Classe représentant un nœud dans le système décentralisé.
    Chaque nœud peut envoyer et recevoir des messages pour construire un arbre binaire.
    """

    def __init__(self, node_id: int, mqtt_handler: MQTTHandler, data_size: int = 0):
        """
        Initialise un nœud avec un ID, des informations sur le broker MQTT et une taille de données.
        """
        self.node_id = node_id
        self.parent: Optional['Node'] = None
        self.left: Optional['Node'] = None
        self.right: Optional['Node'] = None
        self.data_size = data_size
        self.mqtt_handler = mqtt_handler
        self.peers: List[int] = []  # IDs des nœuds découverts

        # logger.info("Init node")
        # Démarrer la communication MQTT
        self.mqtt_handler.connect()

    def discover_peers(self):
        """
        Diffuse un message de découverte pour identifier les autres nœuds connectés.
        """
        discovery_message = json.dumps({"action": "discover", "node_id": self.node_id})
        self.mqtt_handler.publish_message("tree/discovery", discovery_message)

        # Abonnement pour écouter les réponses des autres nœuds
        self.mqtt_handler.subscribe_to_topic(
            topic=f"tree/discovery/response/{self.node_id}",
            callback=self._handle_discovery_response,
        )

    def respond_to_discovery(self, requester_id: int):
        """
        Répond à une demande de découverte d'un autre nœud.
        """
        response_message = json.dumps({"action": "response", "node_id": self.node_id})
        self.mqtt_handler.publish_message(f"tree/discovery/response/{requester_id}", response_message)

    def connect_to_parent(self, parent_node: 'Node'):
        """
        Connecte ce nœud à un parent et notifie le parent de la relation.
        """
        self.parent = parent_node
        connect_message = json.dumps({"action": "connect", "node_id": self.node_id, "parent_id": parent_node.node_id})
        self.mqtt_handler.publish_message(f"tree/{parent_node.node_id}", connect_message)

    def _handle_discovery_response(self, client, userdata, message):
        """
        Gère les réponses de découverte et ajoute les nœuds découverts à la liste des pairs.
        """
        payload = json.loads(message.payload.decode())
        if "node_id" in payload:
            self.peers.append(payload["node_id"])

    def handle_message(self, client, userdata, message):
        """
        Gère les messages reçus pour la gestion des relations parent-enfant.
        """
        payload = json.loads(message.payload.decode())
        action = payload.get("action")

        if action == "set_root":
            # Si ce nœud devient la racine
            root_id = payload.get("node_id")
            if root_id == self.node_id:
                print(f"Node {self.node_id} is now the root!")
                self.parent = None

        elif action == "connect":
            # Établir une connexion parent-enfant
            child_id = payload.get("node_id")
            if child_id and not self.left:
                self.left = child_id
            elif child_id and not self.right:
                self.right = child_id

    def broadcast_message(self, message: dict):
        """
        Diffuse un message à tous les nœuds via MQTT.
        """
        topic = "tree/broadcast"
        self.mqtt_handler.publish_message(topic, json.dumps(message))

    def elect_new_root(self):
        """
        Propose ce nœud comme racine et diffuse l'information.
        """
        election_message = json.dumps({"action": "set_root", "node_id": self.node_id})
        self.broadcast_message(election_message)

    def __str__(self):
        """
        Représentation textuelle du nœud.
        """
        return (
            f"Node {self.node_id}: "
            f"Parent={self.parent.node_id if self.parent else None}, "
            f"Left={self.left.node_id if self.left else None}, "
            f"Right={self.right.node_id if self.right else None}, "
            f"DataSize={self.data_size}"
        )
