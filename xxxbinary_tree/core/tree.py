import random
import json
import logging
from typing import List
from .mqtt_handler import MQTTHandler
from .node import Node

logger = logging.getLogger(__name__)

class Tree:
    """
    Classe pour gérer la structure d'un arbre binaire avec communication via MQTT.
    """

    def __init__(self, mqtt_handler: MQTTHandler, nodes: List[Node]):
        """
        Initialise l'arbre avec un gestionnaire MQTT et une liste de nœuds.
        """
        self.mqtt_handler = mqtt_handler
        self.nodes = nodes
        self.root = None

        # Associer les callbacks pour les messages MQTT
        self.mqtt_handler.subscribe_to_topic("tree/construction", self._on_tree_construction)
        self.mqtt_handler.subscribe_to_topic("tree/election", self._on_election)
        self.mqtt_handler.subscribe_to_topic("tree/evaluate", self._on_evaluation)

    def build_tree(self):
        """
        Construit l'arbre en envoyant des messages MQTT aux nœuds pour assigner leurs relations.
        """
        logger.info("Building the binary tree...")
        if not self.nodes:
            logger.warning("No nodes available to build the tree.")
            return

        self.root = self.nodes[0]
        queue = [self.root]
        idx = 1

        while queue and idx < len(self.nodes):
            current = queue.pop(0)

            if idx < len(self.nodes):
                current.left = self.nodes[idx]
                self._assign_relationship(current.node_id, current.left.node_id, "left")
                queue.append(current.left)
                idx += 1

            if idx < len(self.nodes):
                current.right = self.nodes[idx]
                self._assign_relationship(current.node_id, current.right.node_id, "right")
                queue.append(current.right)
                idx += 1

        logger.info("Tree construction completed.")

    def reorganize_tree(self, new_root_id: int):
        """
        Réorganise l'arbre en envoyant un message pour définir un nouveau nœud racine.
        """
        logger.info(f"Reorganizing tree: setting Node {new_root_id} as the new root.")
        self.mqtt_handler.publish_message("tree/election", json.dumps({
            "action": "reorganize",
            "new_root_id": new_root_id
        }))
        self.root = next((node for node in self.nodes if node.node_id == new_root_id), self.root)

    def evaluate_tree(self):
        """
        Évalue les performances de l'arbre via le nœud racine.
        """
        if not self.root:
            logger.error("Root node is not defined. Cannot evaluate tree.")
            return

        logger.info(f"Evaluating the tree from Node {self.root.node_id}.")
        self.mqtt_handler.publish_message("tree/evaluate", json.dumps({
            "action": "evaluate",
            "root_id": self.root.node_id
        }))

    def _assign_relationship(self, parent_id: int, child_id: int, relationship: str):
        """
        Assigne une relation parent-enfant dans l'arbre.
        """
        logger.info(f"Assigning {relationship} relationship: Parent {parent_id} -> Child {child_id}")
        self.mqtt_handler.publish_message("tree/construction", json.dumps({
            "action": "assign",
            "parent_id": parent_id,
            "child_id": child_id,
            "relationship": relationship
        }))

    def elect_new_root(self):
        """
        Élit un nouveau nœud central (racine) de manière aléatoire.
        """
        new_root = random.choice(self.nodes)
        logger.info(f"Electing a new root: Node {new_root.node_id}")
        self.reorganize_tree(new_root.node_id)

    def _on_tree_construction(self, client, userdata, message):
        """
        Callback pour gérer les événements de construction d'arbre.
        """
        payload = json.loads(message.payload.decode())
        logger.info(f"Tree construction event received: {payload}")

    def _on_election(self, client, userdata, message):
        """
        Callback pour gérer les événements d'élection.
        """
        payload = json.loads(message.payload.decode())
        logger.info(f"Election event received: {payload}")
        if payload.get("action") == "reorganize":
            new_root_id = payload.get("new_root_id")
            self.root = next((node for node in self.nodes if node.node_id == new_root_id), self.root)

    def _on_evaluation(self, client, userdata, message):
        """
        Callback pour gérer les événements d'évaluation.
        """
        payload = json.loads(message.payload.decode())
        logger.info(f"Evaluation event received: {payload}")
