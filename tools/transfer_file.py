import os
import subprocess
import sys
import paramiko
import scp

ORIN_LAN_IP = "ORIN_LAN_IP_ADDRESS"  # Replace with the actual IP address of the ORIN device
USERNAME = "ORIN_USERNAME"  # Replace with the actual username for SSH access to the ORIN device

if __name__ == "__main__":
    # Check for command-line arguments
    if sys.argv[1] == "help" or len(sys.argv) < 4:
        print("Usage: python transfer_profiling_results.py  <get|put> <file_or_folder> <file_or_folder>")
        print("Options:")
        print("  get: Transfer files or folders from the remote host to the local machine")
        print("  put: Transfer files or folders from the local machine to the remote host")
        print("  file_or_folder: Path to the file or folder")
        print("Note: Folders will be copied recursively.")
        sys.exit(1)
    elif sys.argv[1] == "get":
        file_orin = sys.argv[2]
        file_laptop = sys.argv[3]
        transfer_direction = "get"
    elif sys.argv[1] == "put":
        file_laptop = sys.argv[2]
        file_orin = sys.argv[3]
        transfer_direction = "put"
    else:
        print("Invalid option. Use 'help' for usage information.")
        sys.exit(1)

    # Create SSH client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        # Connect to the remote host
        ssh.connect(ORIN_LAN_IP, username=USERNAME, allow_agent=True)
        
        # Create SCP client
        scp_client = scp.SCPClient(ssh.get_transport())
        
        if transfer_direction == "put":
            print(f"Transferring {file_laptop} to {ORIN_LAN_IP}:{file_orin}")
            scp_client.put(file_laptop, recursive=True, remote_path=file_orin)
        elif transfer_direction == "get":
            print(f"Transferruing {ORIN_LAN_IP}:{file_orin} to {file_laptop}")
            scp_client.get(file_orin, recursive=True, local_path=file_laptop)
        
        print("Transfer completed successfully.")
        
    except Exception as e:
        print(f"Error during transfer: {e}")
    
    finally:
        # Close SCP and SSH connections
        if "scp_client" in locals():
            scp_client.close()
        if ssh.get_transport() is not None:
            ssh.close()