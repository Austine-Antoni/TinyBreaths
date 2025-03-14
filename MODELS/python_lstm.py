import serial
import time
import re

# Establish Serial connection (Change "COM11" to your actual Arduino port)
try:
    arduino = serial.Serial("COM11", 9600, timeout=2)
    print("Connected to Arduino. Waiting for count_60s...")
except serial.SerialException:
    print("No Arduino found. Please check the connection.")
    exit()

while True:
    try:
        # Read a line from Arduino
        arduino_data = arduino.readline().decode("utf-8").strip()

        # Check if the line contains "Strains in last 60s: <count_60s>"
        match = re.search(r"Strains in last 60s: (\d+)", arduino_data)
        
        if match:
            count_60s = match.group(1)  # Extract the number
            print(f"count_60s: {count_60s}")
        else:
            print("Waiting for count_60s...")

    except serial.SerialException:
        print("Arduino is disconnected. Exiting...")
        break  # Exit the loop when Arduino is unavailable

    except Exception as e:
        print(f"Error reading data: {e}")

    time.sleep(5)  # Check every 5 seconds

print("Program stopped because Arduino is not available.")



