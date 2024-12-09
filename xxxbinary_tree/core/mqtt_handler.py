import paho.mqtt.client as mqtt
import logging

logger = logging.getLogger(__name__)

class MQTTHandler:
    """
    Classe pour gérer la communication MQTT.
    Fournit des méthodes pour se connecter, publier, et s'abonner à des topics.
    """

    def __init__(self, client_id: str, broker: str, port: int):
        """
        Initialise le gestionnaire MQTT avec les paramètres de connexion.
        """
        self.client_id = client_id
        self.broker = broker
        self.port = port
        self.client = mqtt.Client(client_id=str(client_id))
        self.callbacks = {}

        # Définir les callbacks MQTT de base
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        
        logger.info("Init handler")

    def connect(self):
        """
        Se connecte au broker MQTT et démarre la boucle réseau.
        """
        try:
            logger.info(f"Connecting to MQTT broker {self.broker}:{self.port} with client ID {self.client_id}...")
            self.client.connect(self.broker, self.port)
            self.client.loop_start()  # Utilisation d'une boucle réseau non bloquante
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")

    def disconnect(self):
        """
        Déconnecte proprement le client MQTT.
        """
        logger.info(f"Disconnecting MQTT client {self.client_id}...")
        self.client.loop_stop()
        self.client.disconnect()

    def publish_message(self, topic: str, message: str):
        """
        Publie un message sur un topic donné.
        """
        try:
            logger.info(f"Publishing message to topic '{topic}': {message}")
            self.client.publish(topic, message)
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")

    def subscribe_to_topic(self, topic: str, callback):
        """
        S'abonne à un topic donné et enregistre un callback personnalisé.
        """
        try:
            logger.info(f"Subscribing to topic '{topic}'...")
            self.client.subscribe(topic)
            self.callbacks[topic] = callback
        except Exception as e:
            logger.error(f"Failed to subscribe to topic: {e}")

    def _on_connect(self, client, userdata, flags, rc):
        """
        Callback appelé lors de la connexion au broker.
        """
        if rc == 0:
            logger.info(f"MQTT client {self.client_id} connected successfully.")
        else:
            logger.error(f"MQTT client {self.client_id} failed to connect with return code {rc}.")

    def _on_disconnect(self, client, userdata, rc):
        """
        Callback appelé lors de la déconnexion du broker.
        """
        if rc == 0:
            logger.info(f"MQTT client {self.client_id} disconnected cleanly.")
        else:
            logger.warning(f"MQTT client {self.client_id} disconnected unexpectedly with return code {rc}.")

    def _on_message(self, client, userdata, message):
        """
        Callback appelé lorsqu'un message est reçu sur un topic abonné.
        """
        topic = message.topic
        payload = message.payload.decode("utf-8")
        logger.info(f"Message received on topic '{topic}': {payload}")

        # Appeler le callback personnalisé si défini pour ce topic
        if topic in self.callbacks:
            self.callbacks[topic](client, userdata, message)
        else:
            logger.warning(f"No callback registered for topic '{topic}'.")

