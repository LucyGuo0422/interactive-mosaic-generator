import cv2
import os
import numpy as np
import gradio as gr
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import pickle
from pathlib import Path
from tqdm import tqdm
import tarfile
from huggingface_hub import hf_hub_download


class MosaicGenerator:
    """Generate image mosaics using a dataset of tile images."""
    
    def __init__(self, dataset_path, cache_dir="./cache"):
        self.dataset_path = dataset_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.dataset_features = {}
        self.tile_cache = {}
        self.color_matching_strength = 0.7
    
        
    def load_dataset(self):
        """Load and cache dataset features."""
        dataset_name = os.path.basename(self.dataset_path)
        cache_file = self.cache_dir / f"{dataset_name}_cache.pkl"
        
        # Load from cache if available
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self.dataset_features = pickle.load(f)
                if self.dataset_features:
                    return
            except:
                pass
        
        # Build dataset cache
        if not os.path.exists(self.dataset_path):
            return
        
        image_paths = []
        
        # Scan subdirectories for images
        for category in os.listdir(self.dataset_path):
            category_path = os.path.join(self.dataset_path, category)
            if not os.path.isdir(category_path):
                continue
            
            for image_file in os.listdir(category_path):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_paths.append(os.path.join(category_path, image_file))
        
        if not image_paths:
            return
        
        # Process images with progress bar
        self.dataset_features = {}
        for img_path in tqdm(image_paths, desc="Loading dataset"):
            features = self._extract_features(img_path)
            if features:
                self.dataset_features[img_path] = features
        
        # Save cache
        if self.dataset_features:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.dataset_features, f)
    
    def _extract_features(self, img_path):
        """Extract multi-resolution features from an image."""
        try:
            img = cv2.imread(img_path)
            if img is None:
                return None
            
            features = {
                'resolutions': {},
                'avg_color': None
            }
            
            # Store multiple resolutions for efficient tile loading
            for size in [8, 16, 32, 64]:
                features['resolutions'][size] = cv2.resize(
                    img, (size, size), interpolation=cv2.INTER_AREA
                )
            
            # Calculate average color for matching
            base_img = features['resolutions'][32]
            features['avg_color'] = np.mean(base_img.reshape(-1, 3), axis=0)
            
            return features
            
        except:
            return None
    
    def find_best_match(self, target_region):
        """Find the best matching tile for a target region."""
        if not self.dataset_features:
            return None
        
        target_avg = np.mean(target_region.reshape(-1, 3), axis=0)
        best_distance = float('inf')
        best_path = None
        
        for img_path, features in self.dataset_features.items():
            if features and features['avg_color'] is not None:
                distance = np.linalg.norm(target_avg - features['avg_color'])
                if distance < best_distance:
                    best_distance = distance
                    best_path = img_path
        
        return best_path
    
    def apply_color_transfer(self, tile, target):
        """Transfer color statistics from target to tile."""
        strength = self.color_matching_strength
        
        # Calculate statistics
        tile_mean = np.mean(tile, axis=(0, 1))
        target_mean = np.mean(target, axis=(0, 1))
        tile_std = np.std(tile, axis=(0, 1)) + 1e-6
        target_std = np.std(target, axis=(0, 1)) + 1e-6
        
        # Normalize and transfer
        result = tile.astype(np.float32)
        for c in range(3):
            result[:,:,c] = (result[:,:,c] - tile_mean[c]) / tile_std[c]
            result[:,:,c] = result[:,:,c] * target_std[c] + target_mean[c]
        
        # Blend with original
        result = strength * result + (1 - strength) * tile
        
        return np.clip(result, 0, 255).astype(np.uint8)

    def smooth_edges(self, mosaic, grid_cells):
        """Apply edge smoothing to reduce tile boundaries."""
        h, w = mosaic.shape[:2]
        edge_mask = np.zeros((h, w), dtype=np.float32)
        border = 2
        
        # Mark tile boundaries
        for cell in grid_cells:
            x, y = cell['x'], cell['y']
            cw, ch = cell['width'], cell['height']
            
            # Create edge mask
            edge_mask[y:y+border, x:x+cw] = 1.0
            edge_mask[y+ch-border:y+ch, x:x+cw] = 1.0
            edge_mask[y:y+ch, x:x+border] = 1.0
            edge_mask[y:y+ch, x+cw-border:x+cw] = 1.0
        
        # Apply selective blur
        blurred = cv2.GaussianBlur(mosaic, (5, 5), 1.5)
        mask_3ch = np.stack([edge_mask]*3, axis=-1)
        result = blurred * mask_3ch + mosaic * (1 - mask_3ch)
        
        return result.astype(np.uint8)
    
    def generate(self, image, grid_cells):
        """Generate the final mosaic."""
        mosaic = np.zeros_like(image)
        
        for cell in tqdm(grid_cells, desc="Generating mosaic"):
            x, y = cell['x'], cell['y']
            w, h = cell['width'], cell['height']
            
            # Extract target region
            target_region = image[y:y+h, x:x+w]
            
            # Find and load best matching tile
            best_path = self.find_best_match(target_region)
            if not best_path or best_path not in self.dataset_features:
                continue
                
            features = self.dataset_features[best_path]
            
            # Select appropriate resolution
            available_sizes = list(features['resolutions'].keys())
            closest_size = min(available_sizes, key=lambda s: abs(s - max(w, h)))
            
            # Load tile at optimal resolution
            if abs(closest_size - max(w, h)) > 16:
                if best_path not in self.tile_cache:
                    self.tile_cache[best_path] = cv2.imread(best_path)
                tile = self.tile_cache[best_path]
                if tile is None:
                    continue
            else:
                tile = features['resolutions'][closest_size]
            
            # Resize and apply color transfer
            if tile.shape[:2] != (h, w):
                tile = cv2.resize(tile, (w, h), interpolation=cv2.INTER_LINEAR)
            
            tile = self.apply_color_transfer(tile, target_region)
            mosaic[y:y+h, x:x+w] = tile
        
        # Smooth tile edges
        return self.smooth_edges(mosaic, grid_cells)

def setup_datasets():
    """Download and extract dataset files from HuggingFace if not already present."""
    datasets = {
        "art_images": "art_images.tar.gz",
        "object_images": "object_images.tar.gz", 
        "people_images": "people_images.tar.gz"
    }
    
    for folder_name, tar_name in datasets.items():
        # Check if already extracted
        if os.path.exists(f"./{folder_name}"):
            continue
            
        try:
            # Download from HuggingFace
            tar_path = hf_hub_download(
                repo_id="LucyGuo/Mosaic_tiles",
                filename=tar_name,
                repo_type="dataset",
                cache_dir="./hf_cache"
            )
            
            # Extract
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(".")
            
        except Exception as e:
            print(f"Error setting up {folder_name}: {e}")


def create_adaptive_grid(image, min_size=8, max_size=64, threshold=500):
    """Create an adaptive grid based on image complexity."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    grid_cells = []
    
    def subdivide(x, y, size):
        if x >= w or y >= h or size < min_size:
            return
        
        actual_w = min(size, w - x)
        actual_h = min(size, h - y)
        
        if actual_w < min_size or actual_h < min_size:
            return
        
        region = gray[y:y+actual_h, x:x+actual_w]
        
        # Subdivide high-variance regions
        if region.size > 0 and size > min_size:
            if np.var(region) > threshold:
                half = size // 2
                subdivide(x, y, half)
                subdivide(x + half, y, half)
                subdivide(x, y + half, half)
                subdivide(x + half, y + half, half)
                return
        
        grid_cells.append({
            'x': x, 'y': y,
            'width': actual_w, 'height': actual_h,
            'size': size
        })
    
    # Initialize grid subdivision
    for y in range(0, h, max_size):
        for x in range(0, w, max_size):
            subdivide(x, y, max_size)
    
    return grid_cells


def calculate_quality_metrics(original, mosaic):
    """Calculate image quality metrics."""
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    mosaic_rgb = cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB)
    
    mse = np.mean((original_rgb.astype(float) - mosaic_rgb.astype(float)) ** 2)
    ssim_value = ssim(original_rgb, mosaic_rgb, channel_axis=2)
    
    return {'MSE': mse, 'SSIM': ssim_value}


# Global generator instance
generator = None


def process_mosaic(input_image, target_size, min_cell_size, max_cell_size,
                   variance_threshold, color_matching_strength, dataset_choice):
    """Main processing function for Gradio interface."""
    global generator
    
    try:
        # Dataset paths
        datasets = {
            "Objects": "./object_images",
            "Celebrity Faces": "./people_images",
            "Artwork Collection": "./art_images"
        }
        
        dataset_path = datasets.get(dataset_choice, "./object_images")
        
        # Validate dataset
        if not os.path.exists(dataset_path):
            return None, None, f"Dataset not found: {dataset_path}", ""
        
        # Initialize or update generator
        if generator is None or generator.dataset_path != dataset_path:
            generator = MosaicGenerator(dataset_path)
            generator.load_dataset()
        
        if not generator.dataset_features:
            return None, None, f"No images found in {dataset_path}", ""
        
        # Set parameters
        generator.color_matching_strength = color_matching_strength / 100.0
        
        # Prepare input image
        if isinstance(input_image, Image.Image):
            input_image = np.array(input_image)
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        
        image = cv2.resize(input_image, (target_size, target_size))
        
        # Generate adaptive grid
        grid_cells = create_adaptive_grid(
            image, min_cell_size, max_cell_size, variance_threshold
        )
        
        # Generate mosaic
        mosaic = generator.generate(image, grid_cells)
        
        # Convert for display
        original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mosaic_rgb = cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB)
        
        # Calculate metrics
        metrics = calculate_quality_metrics(image, mosaic)
        
        # Generate statistics
        cell_distribution = {}
        for cell in grid_cells:
            size = cell['size']
            cell_distribution[size] = cell_distribution.get(size, 0) + 1
        
        stats_text = f"""Dataset: {dataset_choice}
Total cells: {len(grid_cells)}
Cell distribution: {dict(sorted(cell_distribution.items()))}
Dataset images: {len(generator.dataset_features)}"""
        
        metrics_text = f"""MSE: {metrics['MSE']:.2f}
SSIM: {metrics['SSIM']:.4f}"""
        
        return original_rgb, mosaic_rgb, stats_text, metrics_text
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return None, None, error_msg, error_msg


def create_interface():
    """Create the Gradio interface."""
    with gr.Blocks(title="Image Mosaic Generator", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎨 Interactive Image Mosaic Generator")
        gr.Markdown("Transform your images into beautiful mosaics using a dataset of tile images")
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Image", 
                    type="pil", 
                    height=300
                )
                
                with gr.Accordion("Parameters", open=True):
                    target_size = gr.Slider(
                        256, 1024, 512, step=64, 
                        label="Target Size (pixels)",
                        info="Output image resolution"
                    )
                    
                    min_cell_size = gr.Slider(
                        4, 32, 8, step=4, 
                        label="Minimum Cell Size",
                        info="Smallest tile size in pixels"
                    )
                    
                    max_cell_size = gr.Slider(
                        32, 128, 64, step=16, 
                        label="Maximum Cell Size",
                        info="Largest tile size in pixels"
                    )
                    
                    variance_threshold = gr.Slider(
                        100, 1000, 500, step=50, 
                        label="Detail Threshold",
                        info="Lower values = more tiles in detailed areas"
                    )
                    
                    color_matching_strength = gr.Slider(
                        0, 100, 70, step=10, 
                        label="Color Matching Strength (%)",
                        info="How much to adjust tile colors (0=original, 100=full match)"
                    )
                    
                    dataset_choice = gr.Dropdown(
                        choices=[
                            "Objects",
                            "Celebrity Faces",
                            "Artwork Collection"
                        ],
                        value="Objects",
                        label="Dataset Selection",
                        info="Choose which image collection to use for tiles"
                    )
                
                generate_btn = gr.Button("Generate Mosaic", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("Mosaic"):
                        mosaic_output = gr.Image(label="Mosaic Result", height=400)
                    with gr.TabItem("Original"):
                        original_output = gr.Image(label="Original", height=400)
                
                with gr.Row():
                    stats_output = gr.Textbox(label="Statistics", lines=4)
                    metrics_output = gr.Textbox(label="Quality Metrics", lines=4)
        
        # Example images
        gr.Markdown("### 📸 Example Images")
        gr.Examples(
            examples=[
                ["./examples/test_img1.jpg"],
                ["./examples/test_img2.jpg"],
                ["./examples/test_img3.jpg"],
                ["./examples/test_img4.jpg"],
                ["./examples/test_img5.jpg"]
            ],
            inputs=input_image,
            label="Click to load an example"
        )
        
        gr.Markdown("""
        ### 💡 Tips for Best Results
        
        **Dataset Selection:**
        - **Objects**: General objects and scenes, versatile for most images
        - **Celebrity Faces**: Human faces and portraits, ideal for creating portrait mosaics
        - **Artwork Collection**: Artistic images, perfect for creative and abstract mosaics
        
        **Parameter Guidelines:**
        - **Target Size**: Higher resolution (768-1024) for detailed images, lower (256-512) for faster processing
        - **Cell Sizes**: Smaller cells (4-16) preserve more detail but take longer to generate
        - **Detail Threshold**: 
          - 200-400 for portraits (captures facial features)
          - 400-600 for general photos (balanced)
          - 600-800 for landscapes (larger uniform areas)
        - **Color Matching**: 
          - 50-60% for preserving tile character
          - 70-80% for smooth color transitions
          - 90-100% for maximum color coherence
        """)
        
        generate_btn.click(
            process_mosaic,
            inputs=[
                input_image, target_size, min_cell_size, max_cell_size,
                variance_threshold, color_matching_strength, dataset_choice
            ],
            outputs=[original_output, mosaic_output, stats_output, metrics_output]
        )
    
    return app


if __name__ == "__main__":
    # Setup datasets on first run
    setup_datasets()
    
    # Create and launch interface
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False
    )