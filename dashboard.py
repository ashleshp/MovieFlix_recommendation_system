import json
import sys
import os
import subprocess
import zmq
import threading
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QLabel, QGridLayout, QTextEdit
)
from PyQt5.QtCore import Qt
# Configuration for microservices
services = {
    "Data Ingestion": {"script": "data_ingestion.py", "process": None, "log_file": "data_ingestion.log", "port": 12312},
    "Data Preprocessing": {"script": "data_preprocessing_service.py", "process": None, "log_file": "data_preprocessing.log", "port": 12346},
    "User Provider Service": {"script":"user_service.py", "process": None, "log_file": "user_service.log", "port": 12345},
    "Profile Generator": {"script":"profile_generator.py", "process": None, "log_file": "profile_generator.log", "port": 12348},
    "Recommendation Model": {"script":"recommendation_model.py", "process": None, "log_file": "recommendation_model.log", "port": 12349},
    "Two Tower Model": {"script":"two_tower_service.py", "process": None, "log_file": "two_tower_model.log", "port": 12352},
    "Feedback Service": {"script":"feedback_service.py", "process": None, "log_file": "feedback_service.log", "port": 12351},
    "FastAPI Server": {"script": "fast_server.py", "process": None, "log_file": "flask_server.log", "port": 12347},
    "Cache Service" : {"script": "RecommendationCache.py", "process":None, "log_file": "RecommendationCache.log", "port":12350}
}

# ZeroMQ Context
context = zmq.Context()


def check_service_status(port):
    """
    Check if a microservice is running by attempting ZeroMQ communication.
    """
    socket = context.socket(zmq.REQ)
    try:
        socket.connect(f"tcp://127.0.0.1:{port}")
        socket.send_json(json.dumps({"request_type": "status"}))
        socket.RCVTIMEO = 5000
        try:
            reply = json.loads(socket.recv_json())
        except zmq.ZMQError as e:
            print(f"Error receiving JSON: {e}")
            return False
        print(f"Service status of {port}: {reply}")
        return reply.get("status") == "running"
    except zmq.ZMQError:
        return False
    finally:
        socket.close()


def gracefully_terminate_service(port):
    """
    Gracefully terminate a microservice by sending a terminate message.
    """
    socket = context.socket(zmq.REQ)
    try:
        socket.connect(f"tcp://127.0.0.1:{port}")
        socket.send_json(json.dumps({"request_type": "terminate"}))
        reply = json.loads(socket.recv_json())
        return reply.get("status") == "stopped"
    except zmq.ZMQError:
        return False
    finally:
        socket.close()



def start_service(service_name):
    """
    Start a microservice as a subprocess.
    """
    service = services[service_name]
    if service["process"] is None:
        script = service["script"]
        log_file = service["log_file"]
        if os.path.exists(script):  # Ensure the script exists
            with open(log_file, "w") as log:
                process = subprocess.Popen(["python", script], stdout=log, stderr=log)
                service["process"] = process
        else:
            print(f"Error: {script} does not exist.")


def stop_service(service_name):
    """
    Stop a running microservice.
    """
    service = services[service_name]
    process = service["process"]
    if process:
        if gracefully_terminate_service(service["port"]):
            process.terminate()
            process.wait()
            service["process"] = None
            return True
    return False
import psutil
def get_total_resource_usage():
    """
    Calculate the total CPU and RAM usage for all running microservices.
    """
    total_cpu = 0.0
    total_memory = 0.0
    for service in services.values():
        process = service["process"]
        if process and process.poll() is None:  # Check if process is running
            try:
                proc = psutil.Process(process.pid)
                total_cpu += proc.cpu_percent(interval=0.1)
                total_memory += proc.memory_info().rss / (1024 * 1024)  # Convert to MB
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    return total_cpu, total_memory

def get_logs(service_name):
    """
    Read logs for a service.
    """
    log_file = services[service_name]["log_file"]
    if os.path.exists(log_file):
        with open(log_file, "r") as file:
            return file.read()
    return "No logs available."


import time  # Add this import

class StatusPollingThread(threading.Thread):
    """
    A separate thread to poll services for their status.
    """
    def __init__(self, update_status_callback,cpuUpdate):
        super().__init__()
        self.update_status_callback = update_status_callback
        self.cpuUdate = cpuUpdate
        self.running = True

    def run(self):
        while self.running:
            for service_name, service in services.items():
                port = service["port"]
                print(f"Checking status of {service_name}")
                is_running = check_service_status(port)
                status_text = "Running" if is_running else "Stopped"
                self.update_status_callback(service_name, status_text)
                self.cpuUdate()
            time.sleep(0.5)  # Pause for 1 second

    def stop(self):
        self.running = False



class MicroservicesDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Microservices Dashboard")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Title
        self.title_label = QLabel("Microservices Dashboard")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        self.layout.addWidget(self.title_label)

        # Grid layout for services
        self.grid_layout = QGridLayout()
        self.layout.addLayout(self.grid_layout)

        self.service_widgets = {}
        for idx, (service_name, service_info) in enumerate(services.items()):
            # Service name
            service_label = QLabel(service_name)
            service_label.setStyleSheet("font-size: 16px;")
            self.grid_layout.addWidget(service_label, idx, 0)

            # Service status
            status_label = QLabel("Stopped")
            status_label.setStyleSheet("color: red;")
            self.grid_layout.addWidget(status_label, idx, 1)

            # Start button
            start_button = QPushButton("Start")
            start_button.clicked.connect(lambda _, name=service_name: self.start_service(name))
            self.grid_layout.addWidget(start_button, idx, 2)

            # Stop button
            stop_button = QPushButton("Stop")
            stop_button.clicked.connect(lambda _, name=service_name: self.stop_service(name))
            self.grid_layout.addWidget(stop_button, idx, 3)

            # Logs button
            logs_button = QPushButton("Logs")
            logs_button.clicked.connect(lambda _, name=service_name: self.show_logs(name))
            self.grid_layout.addWidget(logs_button, idx, 4)

            self.service_widgets[service_name] = {"status_label": status_label}

        # Start/Stop all buttons
        self.start_all_button = QPushButton("Start All")
        self.start_all_button.clicked.connect(self.start_all_services)
        self.layout.addWidget(self.start_all_button)

        self.stop_all_button = QPushButton("Stop All")
        self.stop_all_button.clicked.connect(self.stop_all_services)
        self.layout.addWidget(self.stop_all_button)

        # Polling Thread
        self.polling_thread = StatusPollingThread(self.update_service_status,self.update_cpu)
        self.polling_thread.start()
        # Add this under the layout setup
        self.resource_label = QLabel("CPU: 0.0% | RAM: 0.0 MB")
        self.layout.addWidget(self.resource_label)
    def update_cpu(self):
        cpu, memory = get_total_resource_usage()
        self.resource_label.setText(f"CPU: {cpu:.2f}% | RAM: {memory:.2f} MB")
    def start_service(self, service_name):
        start_service(service_name)

    def stop_service(self, service_name):
        stop_service(service_name)

    def show_logs(self, service_name):
        logs = get_logs(service_name)
        log_window = QTextEdit()
        log_window.setWindowTitle(f"{service_name} Logs")
        log_window.setReadOnly(True)
        log_window.setText(logs)
        log_window.setMinimumSize(600, 400)
        log_window.show()
        self.log_window = log_window  # Keep a reference to prevent garbage collection

    def start_all_services(self):
        for service_name in services:
            self.start_service(service_name)

    def stop_all_services(self):
        for service_name in services:
            self.stop_service(service_name)

    def update_service_status(self, service_name, status):
        status_label = self.service_widgets[service_name]["status_label"]
        if status == "Running":
            status_label.setText(status)
            status_label.setStyleSheet("color: green;")
        else:
            status_label.setText(status)
            status_label.setStyleSheet("color: red;")

    def closeEvent(self, event):
        self.polling_thread.stop()
        self.polling_thread.join()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dashboard = MicroservicesDashboard()
    dashboard.show()
    sys.exit(app.exec_())
