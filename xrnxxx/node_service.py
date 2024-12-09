import os
import json
import time
from paho.mqtt import client as mqtt_client

# Configuration MQTT Broker
MQTT_BROKER = "localhost"
MQTT_PORT = 1883

node_id = int(os.getenv('NODE_ID'))

class NodeService:
    def __init__(self, node_id):
        self.node_id = node_id
        self.parent_id = None
        self.left_id = None
        self.right_id = None
        self.client = mqtt_client.Client(f"Node-{self.node_id}")

    def on_message(self, client, userdata, msg):
        """
        Callback pour gérer les messages reçus.
        """
        payload = json.loads(msg.payload.decode())
        action = payload.get("action")
        if action == "init":
            self.init_node(payload)
        elif action == "rotate_left":
            self.rotate_left()
        elif action == "rotate_right":
            self.rotate_right()
        elif action == "get_info":
            self.publish_info()

    def init_node(self, payload):
        """
        Initialiser le nœud avec les données de parent/gauche/droite.
        """
        self.parent_id = payload.get("parent")
        self.left_id = payload.get("left")
        self.right_id = payload.get("right")
        print(f"Node {self.node_id} initialized: Parent={self.parent_id}, Left={self.left_id}, Right={self.right_id}")

    def rotate_left(self):
        """
        Effectuer une rotation gauche.
        """
        if self.right_id:
            self.parent_id, self.right_id, self.left_id = self.right_id, self.left_id, self.parent_id
            print(f"Node {self.node_id} rotated left")
            self.publish_info()

    def rotate_right(self):
        """
        Effectuer une rotation droite.
        """
        if self.left_id:
            self.parent_id, self.left_id, self.right_id = self.left_id, self.right_id, self.parent_id
            print(f"Node {self.node_id} rotated right")
            self.publish_info()

    def publish_info(self):
        """
        Publier les informations du nœud sur MQTT.
        """
        info = {
            "id": self.node_id,
            "parent": self.parent_id,
            "left": self.left_id,
            "right": self.right_id,
        }
        self.client.publish(f"node/{self.node_id}", json.dumps({"action": "info", "data": info}))

    def start(self):
        """
        Démarrer le client MQTT et s'abonner au topic du nœud.
        """
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print("Connected to MQTT Broker!")
            else:
                print("Failed to connect, return code %d\n", rc)
        self.client.on_message = self.on_message
        self.client.on_connect = on_connect
        self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
        self.client.subscribe(f"node/{self.node_id}")
        self.client.loop_forever()

        print(f"Node {self.node_id} service started and subscribed to topic node/{self.node_id}")


if __name__ == "__main__":
    # node_id = int(input("Enter node ID: "))
    print(f"Node ID {node_id}")
    service = NodeService(node_id)
    service.start()

    # Boucle infinie pour garder le service actif
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        service.client.loop_stop()
