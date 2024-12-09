# import socket
# import threading

# HEADER = 64
# PORT = 5500
# # SERVER = "192.168.1.195" # My IP V4 adress
# SERVER = socket.gethostbyname(socket.gethostname())
# print(SERVER) # IP V4 adress
# print(socket.gethostname()) # Name of my PC

# ADDR = (SERVER, PORT)
# FORMAT = 'utf-8'
# DISCONNECT_MESSAGE = "!DISCONNECT"

# server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server.bind(ADDR)


# def handle_client(conn, addr): 
#     print(f"[NEW CONNECTION] {addr} connected")
    
#     connected = True
#     while connected: 
#         msg_length = conn.recv(HEADER).decode(FORMAT)
#         if msg_length:
#             msg_length = int(msg_length)
#             msg = conn.recv(msg_length).decode(FORMAT)
            
#             if msg == DISCONNECT_MESSAGE:
#                 connected = False
#             print(f"{addr} {msg}")
#             conn.send(f"Msg received".encode(FORMAT))
        
#     conn.close()

# def start():
#     server.listen()
#     print(f"[LISTENING] server listenig on {SERVER}")
#     while True: 
#         conn, addr = server.accept()
#         thread = threading.Thread(target=handle_client, args=(conn, addr))
#         thread.start()
#         print(f"[ACTIVE CONNECTIONS] {threading.active_count() - 1}")

# print("[STARTING] server is starting...")
# start()




import socket
import threading


class Server:
    HEADER = 64
    FORMAT = 'utf-8'
    DISCONNECT_MESSAGE = "!DISCONNECT"

    def __init__(self, port):
        self.port = port
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_ip = socket.gethostbyname(socket.gethostname())
        self.addr = (self.server_ip, self.port)
        self.server.bind(self.addr)

    def handle_client(self, conn, addr):
        print(f"[NEW CONNECTION] {addr} connected")
        connected = True

        while connected:
            msg_length = conn.recv(self.HEADER).decode(self.FORMAT)
            if msg_length:
                msg_length = int(msg_length)
                msg = conn.recv(msg_length).decode(self.FORMAT)

                if msg == self.DISCONNECT_MESSAGE:
                    connected = False
                print(f"{addr}: {msg}")
                conn.send("Message received".encode(self.FORMAT))

        conn.close()

    def start(self):
        self.server.listen()
        print(f"[LISTENING] Server is listening on {self.server_ip}")

        while True:
            conn, addr = self.server.accept()
            thread = threading.Thread(target=self.handle_client, args=(conn, addr))
            thread.start()
            print(f"[ACTIVE CONNECTIONS] {threading.active_count() - 1}")


if __name__ == "__main__":
    print("[STARTING] Server is starting...")
    server = Server(port=5500)
    server.start()
