import os
import faiss
import pickle
import torch
import yaml

def load_config(config_file="config.yaml"):
    """
    Load project configuration from a YAML file.
    """
    if not os.path.exists(config_file):
        print(f"Config file not found: {config_file}")
        return None

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_device(config=None):
    """
    Return the device to run models on: GPU if available, else CPU.
    """
    use_cuda = False
    if config and 'DEVICE' in config:
        use_cuda = config['DEVICE'].upper() == 'GPU'

    if use_cuda and torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def load_faiss_data(config=None):
    """
    Load FAISS index and labels from src/ folder.
    """
    src_folder = os.path.dirname(__file__)
    faiss_file = os.path.join(src_folder, "embeddings.faiss")
    labels_file = os.path.join(src_folder, "labels.pkl")

    if not os.path.exists(faiss_file) or not os.path.exists(labels_file):
        print("FAISS index or labels file not found. Run precompute_embeddings.py first.")
        return None, None

    faiss_index = faiss.read_index(faiss_file)

    with open(labels_file, "rb") as f:
        labels = pickle.load(f)

    return faiss_index, labels


class AttendanceManager:
    """
    Attendance manager with cooldown and optional logging.
    """
    def __init__(self, cooldown_hours=4, log_file=None):
        self.records = {}
        self.cooldown_hours = cooldown_hours
        self.log_file = log_file
        self.last_marked = {}  # name: timestamp

    def mark_attendance(self, name):
        import time
        current_time = time.time()
        if name in self.last_marked:
            elapsed_hours = (current_time - self.last_marked[name]) / 3600
            if elapsed_hours < self.cooldown_hours:
                return  # Skip marking due to cooldown

        self.records[name] = "Present"
        self.last_marked[name] = current_time
        print(f"✅ Attendance marked for {name}")

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(f"{name},Present,{time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    def get_records(self):
        return self.records
