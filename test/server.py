import socket
import os
import struct
import psutil
import time

SERVER_HOST = '127.0.0.1'  # Localhost
SERVER_PORT = 5001
BUFFER_SIZE = 4096

os.makedirs('received_files', exist_ok=True)

def receive_file(conn):
    # Start tracking CPU/time
    cpu_start = psutil.cpu_percent(interval=None)
    start_time = time.time()

    filename_length = struct.unpack('I', conn.recv(4))[0]
    filename = conn.recv(filename_length).decode()

    with open(os.path.join('received_files', filename), 'wb') as f:
        while True:
            bytes_read = conn.recv(BUFFER_SIZE)
            if not bytes_read:
                break
            f.write(bytes_read)

    # Calculate metrics
    cpu_used = psutil.cpu_percent(interval=None) - cpu_start
    transfer_time = time.time() - start_time

    print(f"[SERVER] File received: {filename}")
    print(f"[SERVER] CPU used: {cpu_used:.2f}%")
    print(f"[SERVER] Transfer time: {transfer_time:.2f} sec")

def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((SERVER_HOST, SERVER_PORT))
    server_socket.listen(5)
    print(f"Server listening on {SERVER_HOST}:{SERVER_PORT}")

    conn, addr = server_socket.accept()
    print(f"Connection from {addr} established!")
    receive_file(conn)
    conn.close()

if __name__ == "__main__":
    main()