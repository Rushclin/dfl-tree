import socket
import threading
from typing import List
import os


class Node:
    HEADER = 64
    FORMAT = 'utf-8'
    DISCONNECT_MESSAGE = "!DISCONNECT"

    def __init__(self, port):
        self.port = port
        self.ip = socket.gethostbyname(socket.gethostname())
        self.addr = (self.ip, self.port)
        self.server_socket = None
        self.is_server = False
        self.clients = []
        self.server_thread = None

    def start_server(self):
        """Start the node as a server."""
        self.is_server = True
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(self.addr)
        self.server_socket.listen()
        print(f"[SERVER STARTED] {self.ip}:{self.port} is listening...")
        self.server_thread = threading.Thread(target=self._accept_connections)
        self.server_thread.start()

    def _accept_connections(self):
        """Accept incoming client connections."""
        while self.is_server:
            conn, addr = self.server_socket.accept()
            print(f"[NEW CONNECTION] {addr} connected")
            self.clients.append((conn, addr))
            thread = threading.Thread(target=self._handle_client, args=(conn, addr))
            thread.start()

    def _handle_client(self, conn, addr):
        """Handle communication with a connected client."""
        connected = True
        while connected:
            try:
                msg_length = conn.recv(self.HEADER).decode(self.FORMAT)
                if msg_length:
                    msg_length = int(msg_length)
                    msg = conn.recv(msg_length).decode(self.FORMAT)
                    if msg == self.DISCONNECT_MESSAGE:
                        connected = False
                        print(f"[DISCONNECT] {addr} disconnected")
                    else:
                        print(f"[{addr}] {msg}")
                        conn.send("Message received".encode(self.FORMAT))
            except ConnectionResetError:
                break
        conn.close()

    def connect_to_server(self, server_ip, server_port):
        """Connect to another node acting as a server."""
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((server_ip, server_port))
            print(f"[CONNECTED] Connected to server at {server_ip}:{server_port}")
            return client_socket
        except ConnectionError as e:
            print(f"[ERROR] Failed to connect to {server_ip}:{server_port}: {e}")
            return None

    def send(self, client_socket, msg):
        """Send a message to a connected server."""
        message = msg.encode(self.FORMAT)
        msg_length = len(message)
        send_length = str(msg_length).encode(self.FORMAT)
        send_length += b' ' * (self.HEADER - len(send_length))
        client_socket.send(send_length)
        client_socket.send(message)
        print(client_socket.recv(2048).decode(self.FORMAT))

    def stop_server(self):
        """Stop the server."""
        self.is_server = False
        if self.server_socket:
            self.server_socket.close()
            print(f"[SERVER STOPPED] {self.ip}:{self.port} stopped")


def elect_leader(nodes: List[Node]) -> Node:
    """Elect a leader (server) based on the lowest IP address."""
    nodes.sort(key=lambda node: socket.inet_aton(node.ip))  # Sort by IP address
    leader = nodes[0]
    print(f"[ELECTION] Node {leader.ip}:{leader.port} elected as leader")
    return leader


# Example Usage
if __name__ == "__main__":
    # node1 = Node(port=5501)
    # node2 = Node(port=5502)
    # node3 = Node(port=5503)

    # # List of nodes
    # nodes = [node1, node2, node3]

    # # Elect the leader
    # leader = elect_leader(nodes)

    # # Start the elected node as server
    # leader.start_server()

    # # Connect other nodes to the leader
    # for node in nodes:
    #     if node != leader:
    #         client_socket = node.connect_to_server(leader.ip, leader.port)
    #         if client_socket:
    #             node.send(client_socket, "Hello from client!")
    #             node.send(client_socket, Node.DISCONNECT_MESSAGE)
    
    
    port = int(os.getenv("NODE_PORT", 5500))

    # Créer une instance de Node
    current_node = Node(port=port)

    # Démarrer le nœud comme serveur
    print(f"[NODE] Starting node on port {port}")
    current_node.start_server()
