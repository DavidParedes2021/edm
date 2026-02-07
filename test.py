import time
from datetime import datetime

def print_seconds():
    print("Starting clock... Press Ctrl+C to stop.\n")
    try:
        while True:
            # Get the current time
            now = datetime.now()
            
            # Extract and print just the seconds
            # Use %S for 00-59 format or now.second for an integer
            print(f"Current second: {now.strftime('%S')}", end="\r")
            
            # Pause for 1 second
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nClock stopped.")

if __name__ == "__main__":
    print_seconds()