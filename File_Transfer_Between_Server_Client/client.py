import socket
import os

# --- Configuration ---
# Prompt the user to enter the server's IP address
SERVER_HOST = input("Enter the server's IP address: ")
SERVER_PORT = 65432
BUFFER_SIZE = 4096
FILE_TO_SEND = 'README.txt' # The file you want to send

# Main function to run the client
def main():
    # Create a TCP/IP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            # Connect to the server
            s.connect((SERVER_HOST, SERVER_PORT))
            print(f"Connected to server at {SERVER_HOST}:{SERVER_PORT}")
            
            # Get the file size and filename
            filesize = os.path.getsize(FILE_TO_SEND)
            filename = os.path.basename(FILE_TO_SEND)
            
            # Send the file metadata (filename and size)
            metadata = f"{filename}:{filesize}"
            s.send(metadata.encode())
            
            print(f"Sending file: {filename} ({filesize} bytes)...")
            
            # Open the file and send it in chunks
            with open(FILE_TO_SEND, 'rb') as f:
                bytes_sent = 0
                while bytes_sent < filesize:
                    # Read a chunk from the file
                    chunk = f.read(BUFFER_SIZE)
                    if not chunk:
                        # Break if end of file
                        break
                    
                    # Send the chunk to the server
                    s.sendall(chunk)
                    bytes_sent += len(chunk)
                    
                    # Optional: Print progress
                    progress = (bytes_sent / filesize) * 100
                    print(f"Progress: {progress:.2f}%", end='\r')
            
            print("\nFile sent successfully!")
        
        except FileNotFoundError:
            print(f"Error: The file '{FILE_TO_SEND}' was not found.")
        except ConnectionRefusedError:
            print("Error: Connection refused. Make sure the server is running.")
        except Exception as e:
            print(f"An error occurred: {e}")

# Run the main function
if __name__ == "__main__":
    main()
