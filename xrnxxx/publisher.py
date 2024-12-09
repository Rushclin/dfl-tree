import json

import time
import random
from paho.mqtt import client as mqtt_client


broker = 'broker.emqx.io'
port = 1883
topic_prefix = "node/"  # Topic base for each node
client_id = f'publish-{random.randint(0, 1000)}'


def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print(f"Failed to connect, return code {rc}")

    client = mqtt_client.Client(client_id)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client


def send_message(client, node_id, action, data):
    topic = f"{topic_prefix}{node_id}"
    message = json.dumps({"action": action, "data": data})
    result = client.publish(topic, str(message))
    status = result[0]
    if status == 0:
        print(f"Sent `{message}` to topic `{topic}`")
    else:
        print(f"Failed to send message to topic `{topic}`")


def run():
    client = connect_mqtt()
    client.loop_start()

    # Initializing nodes
    send_message(client, 1, "init", json.dumps({"parent": None, "left": 2, "right": 3}))
    time.sleep(1)
    send_message(client, 2, "init", json.dumps({"parent": 1, "left": None, "right": None}))
    time.sleep(1)
    send_message(client, 3, "init", json.dumps({"parent": 1, "left": None, "right": None}))
    time.sleep(1)

    # Example rotation
    
    send_message(client, 1, "rotate_left", {})
    time.sleep(1)

    send_message(client, 1, "get_info", {})
    time.sleep(1)

    client.loop_stop()


if __name__ == "__main__":
    run()
