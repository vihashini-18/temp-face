# Face Recognition Project - Complete Line-by-Line Explanation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Configuration File (config.yaml)](#configuration-file)
3. [Utility Functions (src/utils.py)](#utility-functions)
4. [Embedding Generation (src/precompute_embeddings.py)](#embedding-generation)
5. [Face Recognition (src/recognize_faces.py)](#face-recognition)
6. [Video Stream (ui/video_stream.py)](#video-stream)
7. [Data Flow Diagram](#data-flow-diagram)

---

## Project Overview

This is a **Real-Time Face Recognition & Attendance System** that:
- Detects faces using YOLOv8
- Recognizes people using DeepFace embeddings
- Searches efficiently with FAISS vector database
- Tracks attendance with 4-hour cooldown
- Supports multiple cameras (webcam + DroidCam)
- Runs on GPU for fast performance

**Tech Stack:**
- Python 3.10
- PyTorch (CUDA 12.6)
- YOLOv8 (face detection)
- DeepFace (face embeddings)
- FAISS (vector search)
- OpenCV (video processing)

---

## Configuration File (config.yaml)

### Purpose
Central configuration file that stores all settings for paths, models, cameras, and behavior.

### Line-by-Line Explanation

```yaml
# Configuration for Face Recognition Project
```
**Line 1:** Comment describing the file purpose

```yaml
PATHS:
```
**Line 4:** Start of PATHS section - defines all file/folder locations

```yaml
  DATASET_DIR: "dataset"
```
**Line 6:** Folder where training images are stored
- Structure: `dataset/person_name/image.jpg`
- Each person gets their own subfolder
- System scans this to build the face database

```yaml
  EMBEDDINGS_DIR: "embeddings"
```
**Line 7:** Folder where generated embeddings are saved
- Contains `faiss_index.bin` (vector database)
- Contains `labels.pkl` (person names)

```yaml
  FAISS_INDEX_FILE: "faiss_index.bin"
```
**Line 8:** Filename for the FAISS vector database
- Binary file storing face embeddings
- Enables fast similarity search (k-nearest neighbors)

```yaml
  LABELS_FILE: "labels.pkl"
```
**Line 9:** Filename for person labels
- Python pickle file
- Maps FAISS index positions to person names

```yaml
  YOLO_FACE_MODEL: "models/yolov8n-face.pt"
```
**Line 10:** Path to YOLOv8 face detection model weights
- Pre-trained model for detecting faces in images
- `.pt` = PyTorch model file

```yaml
RECOGNITION:
```
**Line 14:** Start of RECOGNITION section - face recognition settings

```yaml
  EMBEDDING_MODEL: "VGG-Face"
```
**Line 15:** DeepFace model to use for face embeddings
- VGG-Face: 4096-dimensional vectors
- Other options: Facenet, ArcFace, DeepFace, OpenFace

```yaml
  DISTANCE_METRIC: "cosine"
```
**Line 16:** How to measure similarity between faces
- `cosine`: Cosine similarity (angle between vectors)
- Other options: euclidean, euclidean_l2

```yaml
  VERIFICATION_THRESHOLD: 0.68
```
**Line 18:** Maximum distance to consider a match
- Lower = stricter (fewer false positives)
- Higher = looser (more false positives)
- 0.68 is standard for VGG-Face + cosine

```yaml
DEVICE: "cuda"
```
**Line 21:** Which hardware to use
- `cuda`: Use NVIDIA GPU
- `cpu`: Use CPU only

```yaml
CAMERA_SOURCES:
   - name: "Webcam-0"
     source: 0
   - name: "Phone-1"
     source: "http://192.168.137.141:4343/video"
   - name: "Phone-2"
     source: "http://192.168.1.102:4747/video"
```
**Lines 25-30:** List of camera inputs
- Each camera has a name and source
- `source: 0` = first webcam (integer index)
- `source: "http://..."` = DroidCam IP stream
- System will open one window per camera

```yaml
CAMERA_SOURCES: []
```
**Line 32:** Override with empty list (defaults to single webcam)
- If empty, uses webcam at index 0

```yaml
DEDUPLICATION:
```
**Line 36:** Settings for preventing double-counting (legacy, used before attendance)

```yaml
  SAME_CAMERA_COOLDOWN_SECONDS: 15
```
**Line 37:** Seconds before re-counting same person in same camera

```yaml
  CROSS_CAMERA_COOLDOWN_SECONDS: 30
```
**Line 38:** Seconds before re-counting same person in different camera

```yaml
  MAX_ACCEPTED_DISTANCE: null
```
**Line 39:** Optional stricter distance filter for counting
- `null` = use only recognition threshold

```yaml
ATTENDANCE:
```
**Line 43:** Attendance marking settings

```yaml
  COOLDOWN_HOURS: 4
```
**Line 44:** Hours before marking same person again
- Prevents duplicate attendance if person returns within 4 hours

```yaml
  LOG_FILE: "attendance/attendance_log.csv"
```
**Line 45:** Path to CSV file for logging attendance
- Format: timestamp, name, camera
- Created automatically if missing

---

## Utility Functions (src/utils.py)

### Purpose
Shared helper functions and classes used across the project.

### Imports (Lines 1-9)

```python
import yaml
```
**Line 1:** Load YAML configuration files

```python
import os
```
**Line 2:** File system operations (paths, directories)

```python
import faiss
```
**Line 3:** Facebook AI Similarity Search (vector database)

```python
import pickle
```
**Line 4:** Serialize/deserialize Python objects

```python
import torch
```
**Line 5:** PyTorch for GPU detection

```python
import time
```
**Line 6:** Unix timestamps for cooldown logic

```python
from typing import Optional, Dict, Tuple
```
**Line 7:** Type hints for better code clarity

```python
import csv
```
**Line 8:** CSV file reading/writing for attendance

```python
from datetime import datetime, timedelta
```
**Line 9:** Date/time handling for attendance

### load_config() Function (Lines 12-21)

```python
def load_config(config_path='config.yaml'):
```
**Line 12:** Function to load configuration
- Default parameter: looks for `config.yaml` in current directory

```python
    """Loads the project configuration from a YAML file."""
```
**Line 13:** Docstring explaining function purpose

```python
    try:
```
**Line 14:** Try-except block to handle errors gracefully

```python
        with open(config_path, 'r') as file:
```
**Line 15:** Open YAML file for reading
- `with` ensures file is closed even if error occurs

```python
            config = yaml.safe_load(file)
```
**Line 16:** Parse YAML into Python dictionary
- `safe_load` prevents code injection attacks

```python
            return config
```
**Line 17:** Return the config dictionary

```python
    except Exception as e:
```
**Line 18:** Catch any errors (file not found, invalid YAML, etc.)

```python
        print(f"Error loading config file: {e}")
```
**Line 19:** Print error message

```python
        return None
```
**Line 20:** Return None to signal failure

### save_faiss_data() Function (Lines 23-42)

```python
def save_faiss_data(faiss_index, labels, config):
```
**Line 23:** Save FAISS index and labels to disk
- `faiss_index`: The vector database object
- `labels`: List of person names
- `config`: Configuration dictionary

```python
    try:
```
**Line 25:** Error handling

```python
        os.makedirs(config['PATHS']['EMBEDDINGS_DIR'], exist_ok=True)
```
**Line 26:** Create embeddings folder if it doesn't exist
- `exist_ok=True`: Don't error if folder already exists

```python
        index_path = os.path.join(config['PATHS']['EMBEDDINGS_DIR'], config['PATHS']['FAISS_INDEX_FILE'])
```
**Line 27:** Build full path to FAISS index file
- Example: `embeddings/faiss_index.bin`

```python
        labels_path = os.path.join(config['PATHS']['EMBEDDINGS_DIR'], config['PATHS']['LABELS_FILE'])
```
**Line 28:** Build full path to labels file
- Example: `embeddings/labels.pkl`

```python
        faiss.write_index(faiss_index, index_path)
```
**Line 31:** Save FAISS index to binary file
- Efficient binary format for fast loading

```python
        print(f"FAISS index saved to: {index_path}")
```
**Line 32:** Confirm save location

```python
        with open(labels_path, 'wb') as f:
```
**Line 35:** Open labels file for binary writing

```python
            pickle.dump(labels, f)
```
**Line 36:** Serialize labels list to file

```python
        print(f"Labels saved to: {labels_path}")
```
**Line 37:** Confirm save location

```python
    except Exception as e:
        print(f"Error saving FAISS data: {e}")
```
**Lines 39-40:** Handle and report errors

### load_faiss_data() Function (Lines 45-69)

```python
def load_faiss_data(config):
```
**Line 45:** Load FAISS index and labels from disk

```python
    index_path = os.path.join(config['PATHS']['EMBEDDINGS_DIR'], config['PATHS']['FAISS_INDEX_FILE'])
    labels_path = os.path.join(config['PATHS']['EMBEDDINGS_DIR'], config['PATHS']['LABELS_FILE'])
```
**Lines 47-48:** Build full paths to both files

```python
    if not os.path.exists(index_path) or not os.path.exists(labels_path):
```
**Line 50:** Check if files exist

```python
        print("FAISS index or labels file not found. Run precompute_embeddings.py first.")
        return None, None
```
**Lines 51-52:** If missing, tell user to generate embeddings first

```python
    try:
        index = faiss.read_index(index_path)
```
**Lines 54-56:** Load FAISS index from binary file

```python
        print(f"FAISS index loaded from: {index_path}")
```
**Line 57:** Confirm load

```python
        with open(labels_path, 'rb') as f:
            labels = pickle.load(f)
```
**Lines 60-61:** Deserialize labels from pickle file

```python
        print(f"Labels loaded from: {labels_path}")
```
**Line 62:** Confirm load

```python
        return index, labels
```
**Line 64:** Return both loaded objects

```python
    except Exception as e:
        print(f"Error loading FAISS data: {e}")
        return None, None
```
**Lines 66-68:** Handle errors, return None for both

### get_device() Function (Lines 71-85)

```python
def get_device(config):
```
**Line 71:** Determine if GPU or CPU should be used

```python
    """
    Determines and returns the device (CPU/CUDA) based on config and availability.
    Prefers root-level DEVICE, then HARDWARE_SETTINGS.DEVICE, defaults to 'cuda'.
    """
```
**Lines 72-75:** Docstring with logic explanation

```python
    requested_device = (
        str(config.get('DEVICE', None) or config.get('HARDWARE_SETTINGS', {}).get('DEVICE', 'cuda'))
        .lower()
    )
```
**Lines 76-79:** Get device preference from config
- Check `DEVICE` key first
- Fall back to `HARDWARE_SETTINGS.DEVICE`
- Default to `'cuda'`
- Convert to lowercase

```python
    if requested_device == 'cuda' and torch.cuda.is_available():
```
**Line 81:** Check if CUDA GPU is requested AND available

```python
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
```
**Lines 82-83:** Use GPU, print GPU name

```python
    else:
        device = 'cpu'
        print("CUDA not available or 'cpu' requested. Using CPU.")
```
**Lines 84-86:** Fall back to CPU

```python
    return device
```
**Line 88:** Return the chosen device string

### DedupeManager Class (Lines 92-132)

```python
class DedupeManager:
```
**Line 92:** Class for preventing double-counting across cameras

```python
    def __init__(self, same_camera_cooldown: int = 15, cross_camera_cooldown: int = 30,
                 max_accepted_distance: Optional[float] = None):
```
**Lines 94-95:** Constructor with default cooldowns
- `same_camera_cooldown`: 15 seconds
- `cross_camera_cooldown`: 30 seconds
- `max_accepted_distance`: Optional distance filter

```python
        self.same_camera_cooldown = same_camera_cooldown
        self.cross_camera_cooldown = cross_camera_cooldown
        self.max_accepted_distance = max_accepted_distance
```
**Lines 96-98:** Store parameters as instance variables

```python
        self._seen_per_cam: Dict[Tuple[str, str], float] = {}
```
**Line 101:** Dictionary tracking last seen time per (label, camera)
- Key: (person_name, camera_name)
- Value: Unix timestamp

```python
        self._seen_global: Dict[str, float] = {}
```
**Line 102:** Dictionary tracking last seen time globally
- Key: person_name
- Value: Unix timestamp

```python
    def should_count(self, label: str, camera_name: str, distance: Optional[float] = None) -> bool:
```
**Line 104:** Check if this sighting should be counted

```python
        now = time.time()
```
**Line 110:** Get current Unix timestamp

```python
        if not label or label == 'Unknown':
            return False
```
**Lines 111-112:** Don't count unknowns

```python
        if self.max_accepted_distance is not None and distance is not None:
            if distance > self.max_accepted_distance:
                return False
```
**Lines 114-116:** Check optional distance threshold

```python
        last_cam_key = (label, camera_name)
        last_cam_seen = self._seen_per_cam.get(last_cam_key, 0.0)
        if now - last_cam_seen < self.same_camera_cooldown:
            return False
```
**Lines 118-121:** Check per-camera cooldown
- If seen in this camera recently, don't count

```python
        last_global_seen = self._seen_global.get(label, 0.0)
        if now - last_global_seen < self.cross_camera_cooldown:
            return False
```
**Lines 123-125:** Check cross-camera cooldown
- If seen in ANY camera recently, don't count

```python
        return True
```
**Line 127:** Passed all checks, should count

```python
    def update_seen(self, label: str, camera_name: str):
```
**Line 129:** Update timestamps after counting

```python
        now = time.time()
        if not label or label == 'Unknown':
            return
        self._seen_per_cam[(label, camera_name)] = now
        self._seen_global[label] = now
```
**Lines 130-134:** Record current time for this person

### AttendanceManager Class (Lines 137-179)

```python
class AttendanceManager:
```
**Line 137:** Class for tracking attendance with cooldown

```python
    def __init__(self, cooldown_hours: int = 4, log_file: Optional[str] = None):
```
**Line 141:** Constructor
- `cooldown_hours`: 4 hours default
- `log_file`: Optional CSV path

```python
        self.cooldown = timedelta(hours=cooldown_hours)
```
**Line 142:** Convert hours to timedelta object

```python
        self.log_file = log_file
```
**Line 143:** Store log file path

```python
        self._last_marked: Dict[str, datetime] = {}
```
**Line 144:** Track last attendance time per person
- Key: person_name
- Value: datetime object

```python
        if self.log_file:
```
**Line 147:** Only if log file is configured

```python
            log_dir = os.path.dirname(self.log_file)
```
**Line 148:** Extract directory from file path

```python
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
```
**Lines 149-150:** Create directory if needed

```python
            if not os.path.exists(self.log_file):
```
**Line 152:** If CSV doesn't exist yet

```python
                with open(self.log_file, mode='w', newline='', encoding='utf-8') as f:
```
**Line 153:** Create file in write mode

```python
                    writer = csv.writer(f)
```
**Line 154:** Create CSV writer

```python
                    writer.writerow(["timestamp", "name", "camera"])  # header
```
**Line 155:** Write column headers

```python
    def should_mark(self, name: str) -> bool:
```
**Line 157:** Check if attendance should be marked

```python
        if not name or name == 'Unknown':
            return False
```
**Lines 158-159:** Don't mark unknowns

```python
        now = datetime.now()
```
**Line 160:** Get current datetime

```python
        last = self._last_marked.get(name)
```
**Line 161:** Get last mark time for this person (or None)

```python
        if last and (now - last) < self.cooldown:
            return False
```
**Lines 162-163:** If marked recently (within cooldown), don't mark again

```python
        return True
```
**Line 164:** Passed checks, should mark

```python
    def mark(self, name: str, camera_name: str):
```
**Line 166:** Record attendance

```python
        if not name or name == 'Unknown':
            return
```
**Lines 167-168:** Don't mark unknowns

```python
        now = datetime.now()
```
**Line 169:** Get current datetime

```python
        self._last_marked[name] = now
```
**Line 170:** Update last marked time

```python
        if self.log_file:
```
**Line 171:** If logging is enabled

```python
            with open(self.log_file, mode='a', newline='', encoding='utf-8') as f:
```
**Line 172:** Open CSV in append mode

```python
                writer = csv.writer(f)
```
**Line 173:** Create CSV writer

```python
                writer.writerow([now.isoformat(timespec='seconds'), name, camera_name])
```
**Line 174:** Write attendance record
- Format: `2025-10-18T14:30:00,John,Webcam-0`

---

## Embedding Generation (src/precompute_embeddings.py)

### Purpose
Generates face embeddings for all people in the dataset and stores them in FAISS index.

### Imports (Lines 1-7)

```python
import os
```
**Line 1:** File system operations

```python
import numpy as np
```
**Line 2:** Numerical operations on arrays

```python
import faiss
```
**Line 3:** Vector similarity search

```python
from tqdm import tqdm
```
**Line 4:** Progress bar for loops

```python
from PIL import Image
```
**Line 5:** Image loading (not used in current version)

```python
from deepface import DeepFace
```
**Line 6:** Face recognition library

```python
from utils import load_config, save_faiss_data
```
**Line 7:** Import helper functions

### precompute_embeddings() Function (Lines 10-105)

```python
def precompute_embeddings():
```
**Line 10:** Main function to generate all embeddings

```python
    config = load_config()
```
**Line 11:** Load configuration from YAML

```python
    if not config:
        return
```
**Lines 12-13:** Exit if config failed to load

```python
    dataset_dir = config['PATHS']['DATASET_DIR']
```
**Line 15:** Get dataset folder path (e.g., "dataset")

```python
    embedding_model = config['RECOGNITION']['EMBEDDING_MODEL']
```
**Line 16:** Get model name (e.g., "VGG-Face")

```python
    print(f"Initializing DeepFace with model: {embedding_model}...")
```
**Line 19:** Inform user which model is being loaded

```python
    all_embeddings = []
    all_labels = []
```
**Lines 22-23:** Initialize lists to collect results

```python
    person_folders = [f.name for f in os.scandir(dataset_dir) if f.is_dir()]
```
**Line 26:** Get list of all person folders in dataset
- Uses `os.scandir()` for efficiency
- Filters to directories only

```python
    if not person_folders:
        print(f"No person folders found in {dataset_dir}. Check your folder structure.")
        return
```
**Lines 28-30:** Exit if no people found

```python
    print(f"Found {len(person_folders)} persons to process.")
```
**Line 32:** Show count

```python
    for person_name in tqdm(person_folders, desc="Generating Embeddings"):
```
**Line 34:** Loop through each person with progress bar

```python
        person_dir = os.path.join(dataset_dir, person_name)
```
**Line 35:** Build full path to person's folder

```python
        image_files = [f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
```
**Line 36:** Get all image files for this person
- Filters by extension

```python
        if not image_files:
            print(f"Warning: No images found for {person_name}. Skipping.")
            continue
```
**Lines 38-40:** Skip person if no images

```python
        person_embeddings = []
```
**Line 44:** List to collect embeddings for this person

```python
        for image_name in image_files:
```
**Line 46:** Loop through each image

```python
            image_path = os.path.join(person_dir, image_name)
```
**Line 47:** Build full path to image

```python
            try:
```
**Line 48:** Error handling for each image

```python
                representations = DeepFace.represent(
                    img_path=image_path,
                    model_name=embedding_model,
                    enforce_detection=False,
                    detector_backend='opencv',
                )
```
**Lines 50-55:** Generate face embedding
- `enforce_detection=False`: Process even if face detection is uncertain
- `detector_backend='opencv'`: Use OpenCV for faster detection
- Returns list of face representations

```python
                if representations:
                    embedding = representations[0]['embedding']
                    person_embeddings.append(np.array(embedding))
```
**Lines 59-61:** Extract first face's embedding
- Convert to numpy array
- Add to person's embedding list

```python
                else:
                    print(f"Could not detect face in {image_path}. Skipping.")
```
**Lines 62-63:** Report if no face found

```python
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
```
**Lines 65-67:** Handle any errors (corrupted image, etc.)

```python
        if person_embeddings:
```
**Line 69:** If we got at least one embedding for this person

```python
            mean_embedding = np.mean(person_embeddings, axis=0)
```
**Line 70:** Average all embeddings for this person
- Creates a representative embedding
- More robust than using a single image

```python
            all_embeddings.append(mean_embedding)
            all_labels.append(person_name)
```
**Lines 72-73:** Add to master lists

```python
    if not all_embeddings:
        print("No embeddings were generated. FAISS index not created.")
        return
```
**Lines 77-79:** Exit if no embeddings were created

```python
    embeddings_matrix = np.array(all_embeddings).astype('float32')
```
**Line 81:** Convert list to numpy matrix
- `float32`: FAISS requires 32-bit floats

```python
    dimension = embeddings_matrix.shape[1]
```
**Line 82:** Get embedding dimension (e.g., 4096 for VGG-Face)

```python
    print(f"Creating FAISS Index (Dimension: {dimension})...")
```
**Line 90:** Inform user

```python
    faiss_index = faiss.IndexFlatL2(dimension)
```
**Line 91:** Create FAISS index
- `IndexFlatL2`: Exact search using L2 (Euclidean) distance
- For small datasets, exact search is fast enough

```python
    faiss_index.add(embeddings_matrix)
```
**Line 94:** Add all embeddings to index

```python
    print(f"Total embeddings added to FAISS: {faiss_index.ntotal}")
```
**Line 95:** Confirm count

```python
    save_faiss_data(faiss_index, all_labels, config)
```
**Line 98:** Save index and labels to disk

```python
if __name__ == "__main__":
    precompute_embeddings()
```
**Lines 101-102:** Run when script is executed directly

---

## Face Recognition (src/recognize_faces.py)

### Purpose
Real-time face detection and recognition using YOLOv8 and DeepFace.

### Imports (Lines 1-14)

```python
import os
import cv2
import numpy as np
from deepface import DeepFace
from ultralytics import YOLO
from PIL import Image
```
**Lines 1-6:** Standard imports

```python
try:
    from src.utils import load_config, load_faiss_data, get_device
except ImportError:
    from utils import load_config, load_faiss_data, get_device
```
**Lines 9-12:** Import with fallback
- Try importing from `src.utils` (when run from project root)
- Fall back to `utils` (when run from src directory)

### FaceRecognizer Class (Lines 15-168)

```python
class FaceRecognizer:
```
**Line 15:** Main class for face recognition

```python
    def __init__(self):
```
**Line 16:** Constructor - initializes all components

```python
        try:
            print("Step 1: Loading configuration...")
```
**Lines 17-18:** Start error-handled initialization with progress

```python
            self.config = load_config()
            if not self.config:
                raise Exception("Failed to load project configuration.")
            print("✓ Configuration loaded")
```
**Lines 19-22:** Load and validate config

```python
            print("Step 2: Getting device...")
            self.device = get_device(self.config)
            print(f"✓ Device set to: {self.device}")
```
**Lines 24-26:** Determine GPU/CPU

```python
            print("Step 3: Loading FAISS Index and Labels...")
            self.faiss_index, self.labels = load_faiss_data(self.config)
            if self.faiss_index is None:
                raise Exception("FAISS index not loaded. Run precompute_embeddings.py first.")
            print(f"✓ FAISS index loaded with {len(self.labels)} persons: {self.labels}")
```
**Lines 28-33:** Load face database

```python
            print("Step 4: Loading YOLOv8 Face Detector...")
            yolo_model_path = self.config['PATHS']['YOLO_FACE_MODEL']
            print(f"   Loading from: {yolo_model_path}")
```
**Lines 35-38:** Get YOLO model path

```python
            self.yolo_model = YOLO(yolo_model_path).to(self.device)
            print(f"✓ YOLOv8 Face Detector loaded on {self.device}")
```
**Lines 41-42:** Load YOLO model to GPU/CPU

```python
            print("Step 5: Configuring DeepFace...")
```
**Line 43:** Start DeepFace setup

```python
            self.embedding_model_name = self.config['RECOGNITION']['EMBEDDING_MODEL']
            self.recognition_threshold = self.config['RECOGNITION']['VERIFICATION_THRESHOLD']
            self.distance_metric = self.config['RECOGNITION']['DISTANCE_METRIC']
```
**Lines 46-48:** Store recognition parameters

```python
            print(f"✓ DeepFace Embedding Model: {self.embedding_model_name}")
            print("\n✓✓✓ FaceRecognizer initialized successfully! ✓✓✓\n")
```
**Lines 51-52:** Confirm success

```python
        except Exception as e:
            import traceback
            print(f"\n❌ Error during initialization at one of the steps:")
            print(f"   {str(e)}")
            print("\nFull traceback:")
            traceback.print_exc()
            raise
```
**Lines 54-60:** Detailed error reporting

```python
    def recognize_face(self, frame: np.ndarray):
```
**Line 63:** Main recognition method
- Input: OpenCV frame (numpy array)
- Output: List of recognition results

```python
        results = []
```
**Line 65:** Initialize results list

```python
        yolo_output = self.yolo_model(frame, verbose=False, device=self.device)
```
**Line 68:** Run YOLO face detection
- `verbose=False`: Suppress debug output
- Returns detections with bounding boxes

```python
        for r in yolo_output:
```
**Line 70:** Loop through YOLO results (usually one per frame)

```python
            for box in r.boxes.xyxy:
```
**Line 72:** Loop through detected face boxes
- `xyxy`: Format is [x1, y1, x2, y2]

```python
                x1, y1, x2, y2 = map(int, box)
```
**Line 73:** Convert box coordinates to integers

```python
                padding = 10
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(frame.shape[1], x2 + padding)
                y2 = min(frame.shape[0], y2 + padding)
```
**Lines 76-80:** Add padding around face
- Helps DeepFace get better context
- Clamp to frame boundaries

```python
                face_crop = frame[y1:y2, x1:x2]
```
**Line 83:** Extract face region from frame

```python
                person_name = "Unknown"
                min_distance = float('inf')
```
**Lines 86-87:** Initialize default values

```python
                try:
```
**Line 89:** Error handling for recognition

```python
                    representations = DeepFace.represent(
                        img_path=face_crop,
                        model_name=self.embedding_model_name,
                        enforce_detection=False
                    )
```
**Lines 91-95:** Generate embedding for detected face
- `enforce_detection=False`: We already detected with YOLO

```python
                    if representations:
                        query_embedding = np.array(representations[0]['embedding']).astype('float32')
```
**Lines 97-98:** Extract embedding vector

```python
                        k = 1
```
**Line 102:** Search for 1 nearest neighbor

```python
                        distances, indices = self.faiss_index.search(query_embedding[np.newaxis, :], k)
```
**Line 104:** Search FAISS index
- `query_embedding[np.newaxis, :]`: Add batch dimension
- Returns distances and indices of nearest neighbors

```python
                        min_distance = distances[0][0]
                        best_match_index = indices[0][0]
```
**Lines 107-108:** Extract closest match

```python
                        if min_distance <= self.recognition_threshold:
                            person_name = self.labels[best_match_index]
```
**Lines 111-112:** If close enough, assign name

```python
                except Exception as e:
                    pass # Keep label as "Unknown"
```
**Lines 117-118:** Ignore recognition errors (keep as Unknown)

```python
                results.append({
                    'box': (x1, y1, x2 - x1, y2 - y1), # (x, y, w, h) format
                    'label': person_name,
                    'distance': min_distance
                })
```
**Lines 121-125:** Add result for this face
- Box in (x, y, width, height) format
- Label (name or "Unknown")
- Distance (for debugging/filtering)

```python
        return results
```
**Line 127:** Return all face results

### draw_results() Function (Lines 130-145)

```python
def draw_results(frame, recognition_results):
```
**Line 130:** Draw bounding boxes and labels on frame

```python
    for result in recognition_results:
```
**Line 131:** Loop through each detected face

```python
        x, y, w, h = result['box']
        label = result['label']
        distance = result['distance']
```
**Lines 132-134:** Extract result data

```python
        color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
```
**Line 136:** Green for known, red for unknown
- OpenCV color format: (B, G, R)

```python
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
```
**Line 139:** Draw rectangle around face
- Thickness: 2 pixels

```python
        text = f"{label} ({distance:.2f})"
```
**Line 142:** Format label text with distance

```python
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
```
**Line 143:** Draw text above box

```python
    return frame
```
**Line 145:** Return annotated frame

### Main Test Block (Lines 148-158)

```python
if __name__ == "__main__":
```
**Line 148:** Only run if script is executed directly

```python
    try:
        recognizer = FaceRecognizer()
        print("FaceRecognizer initialized successfully!")
        print(f"Loaded {len(recognizer.labels)} persons: {recognizer.labels}")
```
**Lines 150-153:** Test initialization

```python
    except Exception as e:
        print(f"Failed to initialize FaceRecognizer: {e}")
        print("Please ensure:")
        print("  1. Dataset is populated in 'dataset/' folder")
        print("  2. Run 'python src/precompute_embeddings.py' first")
        exit()
```
**Lines 154-159:** Handle errors with helpful instructions

---

## Video Stream (ui/video_stream.py)

### Purpose
Real-time video processing with multi-camera support and attendance marking.

### Imports (Lines 1-11)

```python
import cv2
import time
import sys
import os
import threading
```
**Lines 1-5:** Standard imports

```python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
```
**Line 8:** Add parent directory to Python path
- Allows importing from `src` package

```python
from src.recognize_faces import FaceRecognizer, draw_results
from src.utils import load_config, AttendanceManager
```
**Lines 10-11:** Import recognition and attendance components

### _camera_loop() Function (Lines 14-54)

```python
def _camera_loop(source, name, recognizer: FaceRecognizer, attendance: AttendanceManager):
```
**Line 14:** Per-camera processing loop (runs in separate thread)
- `source`: Camera index or IP URL
- `name`: Display name
- `recognizer`: Shared FaceRecognizer instance
- `attendance`: Shared AttendanceManager instance

```python
    cap = cv2.VideoCapture(source)
```
**Line 15:** Open video source

```python
    if not cap.isOpened():
        print(f"Error: Could not open video source '{name}' -> {source}")
        return
```
**Lines 16-18:** Check if camera opened successfully

```python
    prev_time = time.time()
```
**Line 20:** Initialize FPS calculation

```python
    window_name = f"Face Recognition - {name}"
```
**Line 21:** Create unique window title

```python
    while True:
```
**Line 23:** Main video loop

```python
        ret, frame = cap.read()
```
**Line 24:** Read frame from camera
- `ret`: Success boolean
- `frame`: Image data

```python
        if not ret:
            print(f"Can't receive frame from {name} (stream end?). Exiting ...")
            break
```
**Lines 25-27:** Exit if frame read fails

```python
        frame = cv2.flip(frame, 1)
```
**Line 30:** Mirror frame horizontally
- Makes webcam view more natural

```python
        results = recognizer.recognize_face(frame)
```
**Line 33:** Run face recognition on frame

```python
        for r in results:
            label = r.get('label', 'Unknown')
            if attendance.should_mark(label):
                attendance.mark(label, name)
                print(f"{label} is present (camera: {name})")
```
**Lines 36-40:** Check and mark attendance
- Only prints when attendance is actually marked
- Respects 4-hour cooldown

```python
        annotated = draw_results(frame, results)
```
**Line 42:** Draw boxes and labels on frame

```python
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time
```
**Lines 45-47:** Calculate FPS
- `max(..., 1e-6)`: Prevent division by zero

```python
        cv2.putText(annotated, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
```
**Line 48:** Draw FPS counter

```python
        cv2.imshow(window_name, annotated)
```
**Line 50:** Display frame in window

```python
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
```
**Lines 52-53:** Exit if 'q' is pressed
- `cv2.waitKey(1)`: Wait 1ms for key press

```python
    cap.release()
    cv2.destroyWindow(window_name)
```
**Lines 55-56:** Clean up camera and window

### run_video_stream() Function (Lines 59-97)

```python
def run_video_stream():
```
**Line 59:** Main entry point for video streaming

```python
    config = load_config()
```
**Line 64:** Load configuration

```python
    try:
        recognizer = FaceRecognizer()
```
**Lines 65-66:** Initialize face recognition

```python
    except Exception as e:
        print(f"Initialization error: {e}")
        print("ACTION REQUIRED: Ensure 'dataset' is populated and 'src/precompute_embeddings.py' has been run successfully.")
        return
```
**Lines 67-70:** Handle initialization errors

```python
    att_cfg = config.get('ATTENDANCE', {}) if config else {}
    attendance = AttendanceManager(
        cooldown_hours=int(att_cfg.get('COOLDOWN_HOURS', 4)),
        log_file=att_cfg.get('LOG_FILE', None)
    )
```
**Lines 73-77:** Create attendance manager from config

```python
    sources = config.get('CAMERA_SOURCES', []) if config else []
    if not sources:
        sources = [{ 'name': 'Webcam-0', 'source': 0 }]
```
**Lines 79-81:** Get camera sources or use default webcam

```python
    print("Starting video streams... Press 'q' in any window to exit that stream.")
```
**Line 83:** User instructions

```python
    threads = []
```
**Line 84:** List to track camera threads

```python
    for cam in sources:
        name = str(cam.get('name', cam.get('source', 'camera')))
        src = cam.get('source', 0)
        t = threading.Thread(target=_camera_loop, args=(src, name, recognizer, attendance), daemon=True)
        t.start()
        threads.append(t)
```
**Lines 85-90:** Start one thread per camera
- `daemon=True`: Thread exits when main program exits
- Shares recognizer and attendance manager across threads

```python
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.2)
```
**Lines 93-95:** Keep main thread alive while cameras run
- Checks if any camera thread is still running

```python
    finally:
        cv2.destroyAllWindows()
        print("Video streams closed.")
```
**Lines 96-98:** Clean up all windows on exit

```python
if __name__ == "__main__":
    run_video_stream()
```
**Lines 101-102:** Run when script is executed directly

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PHASE                           │
│                   (Run Once or When                         │
│                    Adding New People)                       │
└─────────────────────────────────────────────────────────────┘

1. User adds images to dataset/person_name/
2. Run: python src/precompute_embeddings.py
   ↓
   ├─ Load images from dataset/
   ├─ DeepFace extracts face embeddings (4096D vectors)
   ├─ Average embeddings per person
   ├─ Build FAISS index
   └─ Save: embeddings/faiss_index.bin + labels.pkl

┌─────────────────────────────────────────────────────────────┐
│                    RUNTIME PHASE                            │
│                  (Face Recognition)                         │
└─────────────────────────────────────────────────────────────┘

1. Run: python run_video.py or .\start.ps1
2. FaceRecognizer.__init__():
   ├─ Load config.yaml
   ├─ Detect GPU/CPU
   ├─ Load FAISS index + labels
   └─ Load YOLOv8 model

3. For each camera source (parallel threads):
   └─ _camera_loop():
      ├─ Read frame from camera
      ├─ YOLOv8 detects faces → bounding boxes
      ├─ For each face:
      │  ├─ Crop face region
      │  ├─ DeepFace generates embedding
      │  ├─ FAISS searches nearest neighbor
      │  └─ If distance < threshold: assign name
      ├─ AttendanceManager checks cooldown
      ├─ If allowed: mark attendance + log to CSV
      ├─ Draw boxes and labels
      └─ Display frame

┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT                                   │
└─────────────────────────────────────────────────────────────┘

1. Video window(s) showing:
   - Bounding boxes (green=known, red=unknown)
   - Person names + confidence
   - FPS counter

2. Console output:
   - "John is present (camera: Webcam-0)"
   - Only when attendance is marked (4-hour cooldown)

3. CSV log: attendance/attendance_log.csv
   timestamp,name,camera
   2025-10-18T14:30:00,John,Webcam-0
   2025-10-18T14:32:15,Sarah,Phone-1
```

---

## Key Concepts Explained

### 1. Face Embeddings
- **What**: 4096-dimensional vector representing a face
- **Why**: Allows mathematical comparison of faces
- **How**: Deep neural network (VGG-Face) extracts features

### 2. FAISS (Vector Database)
- **What**: Fast similarity search in high-dimensional space
- **Why**: Quickly find closest matching face from thousands
- **How**: Indexes embeddings for efficient k-NN search

### 3. YOLOv8 Face Detection
- **What**: Real-time object detection specialized for faces
- **Why**: Faster and more accurate than traditional methods
- **How**: Convolutional neural network predicts bounding boxes

### 4. Cooldown Logic
- **What**: Time-based prevention of duplicate attendance
- **Why**: Same person shouldn't be counted multiple times
- **How**: Store last marked timestamp per person

### 5. Multi-Camera Threading
- **What**: Parallel processing of multiple video sources
- **Why**: Handle webcam + DroidCam phones simultaneously
- **How**: Python threading with shared recognizer instance

---

## Performance Optimizations

1. **GPU Acceleration**
   - YOLO runs on CUDA
   - PyTorch operations use GPU
   - ~10x faster than CPU

2. **Shared Models**
   - One FaceRecognizer for all cameras
   - Avoids loading models multiple times
   - Saves VRAM

3. **FAISS Indexing**
   - Pre-computed embeddings
   - Sub-millisecond search
   - Scalable to millions of faces

4. **Efficient Detection**
   - YOLOv8-nano (smallest model)
   - Only process detected faces
   - Skip frames if needed (not currently implemented)

---

## Common Issues & Solutions

### Issue: "FAISS index not loaded"
**Cause**: Embeddings haven't been generated
**Solution**: Run `python src/precompute_embeddings.py`

### Issue: Low FPS
**Cause**: Too many cameras or high resolution
**Solution**: Lower camera resolution, reduce FPS, or use fewer cameras

### Issue: "Unknown" for known people
**Cause**: Threshold too strict or poor training images
**Solution**: Increase `VERIFICATION_THRESHOLD` in config or add better images

### Issue: Multiple attendance marks
**Cause**: Cooldown not working
**Solution**: Check `COOLDOWN_HOURS` in config, ensure CSV is writable

### Issue: Camera won't open
**Cause**: Wrong index or IP unreachable
**Solution**: Try different indices (0, 1, 2) or check DroidCam IP/port

---

**End of Documentation**

This file provides a complete line-by-line explanation of every component in your Face Recognition system. Use it as a reference for understanding, debugging, or extending the project.
