# import socket
# import pickle

# HEADER = 64
# PORT = 5500
# SERVER = "192.168.1.195" # My IP V4 adress
# print(SERVER) # IP V4 adress
# print(socket.gethostname()) # Name of my PC

# ADDR = (SERVER, PORT)
# FORMAT = 'utf-8'
# DISCONNECT_MESSAGE = "!DISCONNECT"

# client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# client.connect(ADDR)

# def send(msg):
#     message = msg.encode(FORMAT)
#     msg_length = len(message)
#     send_length = str(msg_length).encode(FORMAT)
#     send_length += b' ' * (HEADER - len(send_length))
#     client.send(send_length)
#     client.send(message)
#     print(client.recv(2048).decode(FORMAT))
    
# send('Hello world !')
# send('Hello Every one !')
# send('Hello Teams !')


# send(DISCONNECT_MESSAGE)




import socket


class Client:
    HEADER = 64
    FORMAT = 'utf-8'
    DISCONNECT_MESSAGE = "!DISCONNECT"

    def __init__(self, server_ip, port):
        self.port = port
        self.server_ip = server_ip
        self.addr = (self.server_ip, self.port)
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect(self.addr)

    def send(self, msg):
        message = msg.encode(self.FORMAT)
        msg_length = len(message)
        send_length = str(msg_length).encode(self.FORMAT)
        send_length += b' ' * (self.HEADER - len(send_length))
        self.client.send(send_length)
        self.client.send(message)
        print(self.client.recv(2048).decode(self.FORMAT))


if __name__ == "__main__":
    client = Client(server_ip="192.168.1.178", port=5500)
    client.send("Hello world!")
    client.send("Hello Everyone!")
    client.send("Hello Teams!")
    client.send(Client.DISCONNECT_MESSAGE)
