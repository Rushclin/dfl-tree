import random
import json
from paho.mqtt import client as mqtt_client


broker = 'broker.emqx.io'
port = 1883
topic_prefix = "node/"  # Base topic
client_id = f'subscribe-{random.randint(0, 1000)}'


class Node:
    def __init__(self, node_id):
        self.node_id = node_id
        self.parent = None
        self.left = None
        self.right = None

    def handle_message(self, message):
        message = json.loads(message)
        print(f"Message {message}")
        
        action = message.get("action")
        data = {}
        if message.get("data"):
            data = json.loads(message.get("data"))

        if action == "init":
            self.init_node(data)
        elif action == "rotate_left":
            self.rotate_left()
        elif action == "rotate_right":
            self.rotate_right()
        elif action == "get_info":
            self.publish_info()

    def init_node(self, data):
        self.parent = data.get("parent")
        self.left = data.get("left")
        self.right = data.get("right")
        print(f"Node {self.node_id} initialized: Parent={self.parent}, Left={self.left}, Right={self.right}")

    def rotate_left(self):
        if self.right:
            self.parent, self.right, self.left = self.right, self.left, self.parent
            print(f"Node {self.node_id} rotated left")

    def rotate_right(self):
        if self.left:
            self.parent, self.left, self.right = self.left, self.right, self.parent
            print(f"Node {self.node_id} rotated right")

    def publish_info(self):
        print(f"Node {self.node_id} Info: Parent={self.parent}, Left={self.left}, Right={self.right}")


def connect_mqtt(node: Node):
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print(f"Node {node.node_id} connected to MQTT Broker!")
            client.subscribe(f"{topic_prefix}{node.node_id}")
        else:
            print(f"Failed to connect, return code {rc}")

    def on_message(client: mqtt_client, userdata, msg):
        print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")

        payload = msg.payload.decode()
        node.handle_message(payload)

    client = mqtt_client.Client(client_id)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(broker, port)
    return client


def run(node_id):
    node = Node(node_id)
    client = connect_mqtt(node)
    client.loop_forever()


if __name__ == "__main__":
    node_id = int(input("Enter Node ID: "))
    run(node_id)
