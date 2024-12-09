import paho.mqtt.client as mqtt
import json
import time
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class BinaryTree:
    """
    Class to manage a binary tree structure using MQTT (EMQX).
    """
    def __init__(self, nodes: List[dict], broker: str, port: int):
        """
        :param nodes: List of nodes with their IDs.
        :param broker: Address of the MQTT broker (EMQX).
        :param port: Port of the MQTT broker.
        """
        self.nodes = nodes
        self.root = None
        self.mqtt_client = mqtt.Client()
        self.broker = broker
        self.port = port

        # Map to store node responses
        self.node_responses: Dict[str, dict] = {}

        # Setup MQTT client
        self.mqtt_client.on_message = self.on_message
        self.mqtt_client.connect(self.broker, self.port)
        self.mqtt_client.loop_start()

        # Subscribe to response topics for all nodes
        for node in self.nodes:
            self.mqtt_client.subscribe(f"node/{node['id']}/response")

    def on_message(self, client, userdata, msg):
        """
        Callback for handling incoming MQTT messages.
        """
        try:
            payload = json.loads(msg.payload.decode())
            node_id = payload.get("id")
            if node_id:
                self.node_responses[node_id] = payload
                logger.info(f"Received response from Node {node_id}: {payload}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode message: {e}")

    def send_request(self, node_id, action, data):
        """
        Send a request to a node via MQTT.
        """
        topic = f"node/{node_id}/request"
        payload = {
            "action": action,
            "data": data
        }
        self.mqtt_client.publish(topic, json.dumps(payload))
        logger.info(f"Sent request to Node {node_id}: {payload}")

    def wait_for_response(self, node_id, timeout=5):
        """
        Wait for a response from a node.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if node_id in self.node_responses:
                response = self.node_responses.pop(node_id)
                return response
            time.sleep(0.1)
        logger.warning(f"Timeout waiting for response from Node {node_id}")
        return None

    def build_tree(self):
        """
        Build the binary tree structure using MQTT.
        """
        logger.info("Building the binary tree via MQTT...")
        if not self.nodes:
            logger.warning("No nodes available to build the tree.")
            return

        # Set the root node
        self.root = self.nodes[0]
        queue = [self.root]
        idx = 1

        while queue and idx < len(self.nodes):
            current = queue.pop(0)
            left = self.nodes[idx] if idx < len(self.nodes) else None
            right = self.nodes[idx + 1] if idx + 1 < len(self.nodes) else None

            # Send initialization request to the current node
            self.send_request(
                node_id=current["id"],
                action="init",
                data={
                    "parent": current.get("parent"),
                    "left": left["id"] if left else None,
                    "right": right["id"] if right else None,
                }
            )

            # Wait for acknowledgment
            response = self.wait_for_response(current["id"])
            if response and response.get("status") == "ok":
                logger.info(f"Node {current['id']} initialized successfully.")
            else:
                logger.error(f"Failed to initialize Node {current['id']}.")

            # Add children to the queue
            if left:
                left["parent"] = current["id"]
                queue.append(left)
                idx += 1
            if right:
                right["parent"] = current["id"]
                queue.append(right)
                idx += 1
