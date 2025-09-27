# 🎨 Interactive Image Mosaic Generator

Transform any image into a beautiful mosaic artwork using advanced computer vision techniques and adaptive grid generation.

## 🌟 Features

- **Adaptive Grid Generation**: Automatically creates optimal tile layouts based on image complexity
- **Multi-Dataset Support**: Choose from Objects, Celebrity Faces, or Artwork Collections
- **Color Transfer**: Advanced color matching to blend tiles seamlessly with the original image
- **Edge Smoothing**: Reduces visible tile boundaries for a more polished look
- **Quality Metrics**: Real-time SSIM and MSE calculations to evaluate mosaic quality
- **Interactive Web Interface**: User-friendly Gradio interface with real-time parameter adjustment

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- 4GB+ RAM recommended for processing large images

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd clean-repo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

The application will automatically download required datasets on first run and launch a web interface at `http://localhost:7860`.

> **Note**: The datasets are not included in this repository due to size constraints. They will be automatically downloaded from HuggingFace Hub when you first run the application.

## 📊 Datasets

The application uses three curated datasets that are automatically downloaded from HuggingFace Hub:

- **Objects** (800+ categories): General objects, animals, and scenes
- **Celebrity Faces** (13,000+ images): Human faces and portraits  
- **Artwork Collection** (40+ artists): Classic and modern art pieces

> **Dataset Download**: On first run, the application will automatically download these datasets (~2-3GB total). The download progress will be shown in the terminal. Datasets are cached locally for faster subsequent runs.

## 🎛️ Parameters Guide

### Image Processing
- **Target Size**: Output resolution (256-1024 pixels)
- **Minimum Cell Size**: Smallest tile size (8-32 pixels)
- **Maximum Cell Size**: Largest tile size (32-64 pixels)

### Quality Control
- **Detail Threshold**: Controls subdivision sensitivity
  - 200-400: Portraits (captures facial features)
  - 400-600: General photos (balanced)
  - 600-800: Landscapes (larger uniform areas)
- **Color Matching Strength**: How much to adjust tile colors (0-100%)

## 🔧 Technical Details

### Algorithm Overview

1. **Feature Extraction**: Multi-resolution feature extraction with color statistics
2. **Adaptive Gridding**: Variance-based subdivision for optimal tile placement
3. **Tile Matching**: Color distance-based best match selection
4. **Color Transfer**: Statistical color matching between tiles and target regions
5. **Edge Smoothing**: Gaussian blur applied selectively to tile boundaries

### Performance Optimizations

- **Caching**: Dataset features cached for faster subsequent runs
- **Multi-resolution**: Tiles stored at multiple sizes for efficient loading
- **Memory Management**: Optimized image processing pipeline

## 📈 Quality Metrics

The application provides real-time quality assessment:

- **MSE (Mean Squared Error)**: Lower values indicate better reconstruction
- **SSIM (Structural Similarity Index)**: Higher values (0-1) indicate better perceptual quality

## 🎯 Best Practices

### For Portraits
- Use "Celebrity Faces" dataset
- Detail threshold: 200-400
- Color matching: 70-80%
- Target size: 512-768 pixels

### For Landscapes
- Use "Objects" or "Artwork Collection" dataset
- Detail threshold: 600-800
- Color matching: 60-70%
- Target size: 768-1024 pixels

### For Abstract Art
- Use "Artwork Collection" dataset
- Detail threshold: 400-600
- Color matching: 80-90%
- Target size: 512-1024 pixels

## 🛠️ Dependencies

- `opencv-python>=4.8.0`: Computer vision operations
- `numpy>=1.24.0`: Numerical computations
- `gradio>=4.0.0`: Web interface
- `Pillow>=9.5.0`: Image processing
- `scikit-image>=0.21.0`: Image quality metrics
- `tqdm>=4.65.0`: Progress bars
- `huggingface-hub>=0.16.0`: Dataset management

## 📁 Project Structure

```
clean-repo/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── cache/                 # Feature cache for faster loading (auto-created)
├── hf_cache/             # HuggingFace dataset cache (auto-created)
├── examples/              # Sample images for testing (auto-downloaded)
├── art_images/            # Artwork dataset (downloaded on first run)
├── object_images/         # Objects dataset (downloaded on first run)
└── people_images/         # Celebrity faces dataset (downloaded on first run)
```

> **Note**: The `art_images/`, `object_images/`, and `people_images/` directories are created automatically when you first run the application. They are not included in the GitHub repository due to size constraints.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Dataset provided by HuggingFace Hub
- Gradio for the interactive interface
- OpenCV and scikit-image for computer vision operations
