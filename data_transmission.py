import socket
class Transfer:
    def __init__(self,data):
        self.data=data
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # todo 使用UDP
        self.serverAddressPort = ('127.0.0.1', 9090)
    def sent(self):
        self.sock.sendto(str.encode(str(self.data)), self.serverAddressPort)