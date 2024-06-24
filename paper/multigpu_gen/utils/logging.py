import csv
import sys
import time
import subprocess
import threading

import time
import subprocess
import threading
import csv
from datetime import datetime

def gpu_monitor(csv_path, interval=0.1, stop_event=None):
    """
    Function to monitor GPU utilization and memory usage, log to wandb, and write to a CSV file.
    Runs in a separate thread with a specified polling interval.
    """
    # Create the CSV file and write the header
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "gpu_util", "gpu_mem"])

    while not stop_event.is_set():
        try:
            # Get current timestamp
            timestamp = str(time.time()) 

            # Get GPU utilization and memory usage
            gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,nounits,noheader"])
            gpu_info = gpu_info.decode('utf-8').strip().split(',')
            gpu_util = float(gpu_info[0])
            gpu_mem = float(gpu_info[1])


            # Write GPU utilization and memory usage to the CSV file
            with open(csv_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, gpu_util, gpu_mem])

            # Sleep for the specified interval
            time.sleep(interval)
        except KeyboardInterrupt:
            # Exit the thread if the main script is interrupted
            print("GPU monitoring thread interrupted.")
            break

    print("GPU monitoring thread stopped.")
class Logger(object):
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


