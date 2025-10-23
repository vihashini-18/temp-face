# Face Recognition Attendance System

A real-time face recognition system for automated attendance tracking using deep learning and FAISS vector search. This project implements an efficient face recognition pipeline with a user-friendly Streamlit interface for live video processing.

## 🌟 Features

- **Real-time Face Recognition**: Recognizes faces from live webcam feed with high accuracy
- **FAISS Vector Search**: Ultra-fast face matching using Facebook AI Similarity Search
- **Automated Attendance Logging**: Automatically logs attendance with timestamps to CSV files
- **Streamlit Web Interface**: Interactive video streaming interface with real-time recognition display
- **Pre-computed Embeddings**: Efficient face encoding using pre-computed embeddings for faster recognition
- **Configurable Settings**: Easy configuration through YAML file for model parameters and paths
- **Multiple Person Support**: Handles multiple faces in a single frame
- **Attendance Management**: Organized attendance logs with date-stamped CSV files

## 📁 Project Structure

```
Face-Recognition-Project/
├── src/                           # Core recognition modules
│   ├── precompute_embeddings.py  # Generate face embeddings and FAISS index
│   ├── recognize_faces.py        # Face recognition logic
│   └── utils.py                  # Helper functions and utilities
├── ui/                            # User interface
│   └── video_stream.py           # Streamlit video streaming interface
├── models/                        # Model storage
│   ├── faiss_index.bin           # FAISS similarity search index
│   └── embeddings.pkl            # Pre-computed face embeddings
├── dataset/                       # Training images
│   ├── person1/                  # Individual person directories
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── person2/
│       └── image1.jpg
├── attendance/                    # Attendance logs
│   └── attendance_YYYY-MM-DD.csv # Daily attendance records
├── config.yaml                    # Configuration file
├── requirements.txt               # Python dependencies
├── run_video.py                   # Main entry point
└── start.ps1                      # PowerShell startup script
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Webcam or video input device
- Windows (for PowerShell script) or Unix-based system

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Thiyanesh07/Face-Recognition-Project.git
   cd Face-Recognition-Project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up dataset**
   - Create a directory for each person in the `dataset/` folder
   - Add multiple images of each person (at least 3-5 images recommended)
   - Example structure:
     ```
     dataset/
     ├── John_Doe/
     │   ├── photo1.jpg
     │   ├── photo2.jpg
     │   └── photo3.jpg
     └── Jane_Smith/
         ├── photo1.jpg
         └── photo2.jpg
     ```

4. **Precompute face embeddings**
   ```bash
   python src/precompute_embeddings.py
   ```
   This will:
   - Process all images in the `dataset/` directory
   - Generate face embeddings using deep learning
   - Build a FAISS index for fast similarity search
   - Save results to `models/` directory

5. **Configure settings (optional)**
   Edit `config.yaml` to customize:
   - Model parameters
   - Recognition thresholds
   - File paths
   - Camera settings

### Usage

#### Method 1: Using PowerShell Script (Windows)
```powershell
.\start.ps1
```

#### Method 2: Using Python Directly
```bash
python run_video.py
```

The Streamlit interface will launch automatically in your default web browser at `http://localhost:8501`

### Using the Interface

1. **Start Video Stream**: Click to enable webcam feed
2. **Face Recognition**: System automatically detects and recognizes faces in real-time
3. **Attendance Logging**: Recognized faces are logged with timestamps
4. **View Results**: Check the `attendance/` folder for CSV files with attendance records

## 🛠️ Technical Details

### Core Components

#### 1. Face Embedding Generation (`precompute_embeddings.py`)
- Loads images from the dataset directory
- Extracts facial features using deep learning models
- Generates high-dimensional embedding vectors
- Builds FAISS index for efficient similarity search
- Stores embeddings and person labels for recognition

#### 2. Face Recognition (`recognize_faces.py`)
- Captures frames from video stream
- Detects faces using computer vision techniques
- Generates embeddings for detected faces
- Performs similarity search against FAISS index
- Returns identity with confidence scores
- Implements attendance tracking logic

#### 3. Utility Functions (`utils.py`)
- Configuration loading and management
- File I/O operations
- Image preprocessing
- Attendance CSV handling
- Helper functions for data management

#### 4. Streamlit Interface (`video_stream.py`)
- Real-time video display
- Face bounding box visualization
- Recognition results overlay
- Control buttons and settings
- Responsive web interface

### Technology Stack

- **Deep Learning**: Face embedding generation and recognition
- **FAISS**: Vector similarity search for fast face matching
- **OpenCV**: Computer vision and video processing
- **Streamlit**: Interactive web interface
- **NumPy/Pandas**: Data manipulation and analysis
- **PyYAML**: Configuration management

### Performance Optimizations

- **Pre-computed Embeddings**: Face encodings generated once and reused
- **FAISS Index**: Sub-millisecond face matching even with large datasets
- **Efficient Frame Processing**: Optimized video frame handling
- **Lazy Loading**: Models loaded only when needed

## 📊 Configuration

The `config.yaml` file allows customization of:

```yaml
# Example configuration
model:
  name: "facenet"  # or other supported models
  threshold: 0.6   # Recognition confidence threshold

paths:
  dataset: "dataset/"
  models: "models/"
  attendance: "attendance/"

video:
  camera_index: 0  # Default webcam
  frame_width: 640
  frame_height: 480
  fps: 30
```

## 📝 Attendance Records

Attendance logs are saved as CSV files in the `attendance/` directory:

```csv
Name,Timestamp,Date,Time
John Doe,2025-10-19 09:15:32,2025-10-19,09:15:32
Jane Smith,2025-10-19 09:16:45,2025-10-19,09:16:45
```

## 🔧 Troubleshooting

### Common Issues

1. **Camera not detected**
   - Check webcam connection
   - Verify camera permissions
   - Try changing `camera_index` in config.yaml

2. **Poor recognition accuracy**
   - Add more training images per person
   - Ensure good lighting in training images
   - Adjust recognition threshold in config
   - Verify image quality and face visibility

3. **Slow performance**
   - Reduce frame resolution in config
   - Ensure FAISS index is properly built
   - Check system resources

4. **Module import errors**
   - Verify all dependencies are installed
   - Check Python version compatibility
   - Reinstall requirements: `pip install -r requirements.txt --force-reinstall`

## 🔒 Security & Privacy

- Face images and embeddings are stored locally
- No data is transmitted to external servers
- Attendance logs contain only names and timestamps
- Consider encryption for sensitive deployments
- Ensure compliance with local privacy regulations

## 🚀 Future Enhancements

- [ ] Support for multiple camera sources
- [ ] Cloud storage integration for attendance logs
- [ ] Mobile app interface
- [ ] Advanced analytics and reporting
- [ ] Integration with existing attendance systems
- [ ] GPU acceleration for faster processing
- [ ] Anti-spoofing measures (liveness detection)
- [ ] User management dashboard

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FAISS](https://github.com/facebookresearch/faiss) - Facebook AI Similarity Search for efficient vector matching
- [OpenCV](https://opencv.org/) - Computer vision library for image and video processing
- [Streamlit](https://streamlit.io/) - Framework for building interactive web applications
- Deep learning community for face recognition models

## 📞 Contact & Support

For questions, issues, or suggestions:
- Create an issue in this repository
- Check existing issues for solutions
- Review documentation and troubleshooting guide

---

**Note**: This system is designed for educational purposes and small to medium-scale deployments. For production environments with high traffic or strict security requirements, consider additional hardening, scaling infrastructure, and compliance measures.
