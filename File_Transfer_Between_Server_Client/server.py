import socket
import os

# --- Configuration ---
# '0.0.0.0' listens on all available network interfaces,
# allowing other computers on the network to connect.
HOST = '0.0.0.0'
PORT = 65432       # Port to listen on (ports > 1023 are generally safe)
BUFFER_SIZE = 4096 # The size of each chunk of data to receive

# Main function to run the server
def main():
    # Create a TCP/IP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Allow the port to be reused immediately after the program exits
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Bind the socket to the host and port
        s.bind((HOST, PORT))
        
        # Listen for incoming connections (allow a queue of up to 5 clients)
        s.listen(5)
        print(f"Server is listening on {HOST}:{PORT}")
        
        # Accept a connection from a client
        conn, addr = s.accept()
        with conn:
            print(f"Connection established with {addr}")
            
            try:
                # Receive the file metadata (filename and filesize)
                received_data = conn.recv(BUFFER_SIZE).decode()
                filename, filesize = received_data.split(':')
                filesize = int(filesize)
                
                print(f"Receiving file: {filename} ({filesize} bytes)...")
                
                # Create a new file to write the received data
                # We'll save it with a new name to avoid overwriting original
                new_filename = f"received_{os.path.basename(filename)}"
                
                with open(new_filename, 'wb') as f:
                    bytes_received = 0
                    while bytes_received < filesize:
                        # Receive a chunk of data
                        chunk = conn.recv(BUFFER_SIZE)
                        if not chunk:
                            # Break if connection is lost
                            break
                        
                        # Write the chunk to the new file
                        f.write(chunk)
                        bytes_received += len(chunk)
                        
                        # Optional: Print progress
                        progress = (bytes_received / filesize) * 100
                        print(f"Progress: {progress:.2f}%", end='\r')
                
                print("\nFile transfer complete!")
                
            except Exception as e:
                print(f"\nAn error occurred: {e}")
                
# Run the main function
if __name__ == "__main__":
    main()

