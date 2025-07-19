"""
FIXED: Floor Plan Processor Module
The issue was in the create_occupancy_grid method - it was inverting the logic
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union
import matplotlib.pyplot as plt
import cv2
import numpy as np
from typing import Optional, Tuple

class FloorPlanProcessor:
    """
    Processes floor plan images and converts them to binary occupancy grids
    suitable for path planning algorithms.
    """
    
    def __init__(self, 
                 obstacle_threshold: int = 127,
                 blur_kernel_size: int = 5,
                 canny_low: int = 50,
                 canny_high: int = 150):
        """
        Initialize the floor plan processor.
        
        Args:
            obstacle_threshold: Threshold for binary conversion (0-255)
            blur_kernel_size: Gaussian blur kernel size for noise reduction
            canny_low: Lower threshold for Canny edge detection
            canny_high: Upper threshold for Canny edge detection
        """
        self.obstacle_threshold = obstacle_threshold
        self.blur_kernel_size = blur_kernel_size
        self.canny_low = canny_low
        self.canny_high = canny_high
        
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image from file path.
        
        Args:
            image_path: Path to the floor plan image
            
        Returns:
            Loaded image as numpy array
        """
        try:
            # Try loading with OpenCV first
            image = cv2.imread(image_path)
            if image is None:
                # Fallback to PIL
                pil_image = Image.open(image_path)
                image = np.array(pil_image)
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # Convert RGB to BGR for OpenCV compatibility
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image
        except Exception as e:
            raise IOError(f"Failed to load image from {image_path}: {str(e)}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the floor plan image.
        
        Args:
            image: Input floor plan image
            
        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale if necessary
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        if self.blur_kernel_size > 0:
            gray = cv2.GaussianBlur(gray, 
                                   (self.blur_kernel_size, self.blur_kernel_size), 
                                   0)
        
        return gray
    
    def detect_edges(self, image: np.ndarray, method: str = 'canny') -> np.ndarray:
        """
        Detect edges in the floor plan image.
        
        Args:
            image: Preprocessed grayscale image
            method: Edge detection method ('canny', 'sobel', 'laplacian')
            
        Returns:
            Binary edge image
        """
        if method.lower() == 'canny':
            edges = cv2.Canny(image, self.canny_low, self.canny_high)
        elif method.lower() == 'sobel':
            # Sobel edge detection
            sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx**2 + sobely**2)
            edges = np.uint8(edges / edges.max() * 255)
            _, edges = cv2.threshold(edges, self.obstacle_threshold, 255, cv2.THRESH_BINARY)
        elif method.lower() == 'laplacian':
            # Laplacian edge detection
            edges = cv2.Laplacian(image, cv2.CV_64F)
            edges = np.uint8(np.absolute(edges))
            _, edges = cv2.threshold(edges, self.obstacle_threshold, 255, cv2.THRESH_BINARY)
        else:
            raise ValueError(f"Unsupported edge detection method: {method}")
        
        return edges
    
    def compute_homography(self, calibration_points):
        """
        Compute homography matrix for perspective correction.
        calibration_points: List of 16 floats: src (8 coords) then dst (8 coords)
        """
        if len(calibration_points) != 16:
            raise ValueError("Need 16 values: 8 for 4 source points (x,y pairs), 8 for 4 destination points.")
        
        src_pts = np.array(calibration_points[:8], dtype="float32").reshape(-1, 2)  # Reshape to (4, 2)
        dst_pts = np.array(calibration_points[8:], dtype="float32").reshape(-1, 2)
        
        H, _ = cv2.findHomography(src_pts, dst_pts)
        return H

    def apply_homography(self, image: np.ndarray, H: np.ndarray, output_size=(500, 500)) -> np.ndarray:
        """
        Apply homography to rectify image perspective.
        image: Input image (RGB or grayscale)
        H: Homography matrix from compute_homography
        output_size: Desired output dimensions (width, height)
        """
        return cv2.warpPerspective(image, H, output_size)

    def preprocess_real_image(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        mean_brightness = np.mean(gray)
        if mean_brightness < 50:  # Arbitrary threshold for "too dark"
            print("Warning: Image is very dark (mean brightness {:.2f}). Adjust lighting for better detection.".format(mean_brightness))
        
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        return enhanced

    def detect_contours_for_obstacles(self, edges: np.ndarray) -> np.ndarray:
        """
        Use contours to fill obstacles in real images (helps with furniture detection).
        Returns binary mask where 1=obstacle.
        """
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(edges)
        
        # Fill large contours as obstacles (filter small noise)
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:  # Threshold for min obstacle size
                cv2.drawContours(mask, [cnt], -1, 255, -1)
        
        return mask
    
    def create_occupancy_grid(self, 
                            image: np.ndarray, 
                            method: str = 'threshold') -> np.ndarray:
        """
        Create binary occupancy grid from processed image.
        
        Args:
            image: Processed image (grayscale or edges)
            method: Method for creating occupancy grid ('threshold', 'adaptive')
            
        Returns:
            Binary occupancy grid (0 = free space, 1 = obstacle)
        """
        if method == 'threshold':
            # FIXED: Correct logic for occupancy grid
            # For typical floor plans: dark areas (low values) = obstacles
            # Light areas (high values) = free space
            # Occupancy grid: 1 = obstacle, 0 = free space
            occupancy_grid = (image < self.obstacle_threshold).astype(np.uint8)
            
        elif method == 'adaptive':
            # Adaptive thresholding for varying lighting conditions
            binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            # Convert: 0 (dark) = obstacle, 255 (light) = free space
            occupancy_grid = (binary == 0).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported occupancy grid method: {method}")
        
        return occupancy_grid
    
    def apply_morphological_operations(self, 
                                     occupancy_grid: np.ndarray,
                                     operation: str = 'close',
                                     kernel_size: int = 3) -> np.ndarray:
        """
        Apply morphological operations to clean up the occupancy grid.
        
        Args:
            occupancy_grid: Binary occupancy grid
            operation: Morphological operation ('close', 'open', 'dilate', 'erode')
            kernel_size: Size of the morphological kernel
            
        Returns:
            Cleaned occupancy grid
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        if operation == 'close':
            # Closing: dilation followed by erosion (fills small holes)
            result = cv2.morphologyEx(occupancy_grid, cv2.MORPH_CLOSE, kernel)
        elif operation == 'open':
            # Opening: erosion followed by dilation (removes small noise)
            result = cv2.morphologyEx(occupancy_grid, cv2.MORPH_OPEN, kernel)
        elif operation == 'dilate':
            # Dilation: expands obstacles
            result = cv2.dilate(occupancy_grid, kernel, iterations=1)
        elif operation == 'erode':
            # Erosion: shrinks obstacles
            result = cv2.erode(occupancy_grid, kernel, iterations=1)
        else:
            raise ValueError(f"Unsupported morphological operation: {operation}")
        
        return result
    
    def preprocess_real_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhanced preprocessing for real camera images: grayscale, blur, and CLAHE for contrast.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), 0)
        
        # Adaptive histogram equalization for better contrast in varying lighting
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        return enhanced

    def detect_contours_for_obstacles(self, edges: np.ndarray) -> np.ndarray:
        """
        Use contours to fill obstacles in real images (helps with furniture detection).
        Returns binary mask where 1=obstacle.
        """
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(edges)
        
        # Fill large contours as obstacles (filter small noise)
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:  # Threshold for min obstacle size
                cv2.drawContours(mask, [cnt], -1, 255, -1)
        
        return mask

    def process_floor_plan(self, 
                           image_path_or_array,  # Accepts path or np.array
                           edge_method: str = 'canny',
                           occupancy_method: str = 'adaptive',  # Default to adaptive for real imgs
                           apply_morphology: bool = True,
                           is_real_image: bool = False,
                           homography_matrix: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Updated pipeline: Handles path or array input, optional homography, real-image mode.
        """
        if isinstance(image_path_or_array, str):
            original_image = self.load_image(image_path_or_array)
        else:
            original_image = image_path_or_array  # Assume np.array from camera
        
        if homography_matrix is not None:
            original_image = self.apply_homography(original_image, homography_matrix)
        
        if is_real_image:
            preprocessed = self.preprocess_real_image(original_image)
        else:
            preprocessed = self.preprocess_image(original_image)
        
        if edge_method == 'none':
            processed_image = preprocessed
        else:
            processed_image = self.detect_edges(preprocessed, edge_method)
        
        if is_real_image:
            processed_image = self.detect_contours_for_obstacles(processed_image)
        
        occupancy_grid = self.create_occupancy_grid(processed_image, occupancy_method)
        
        if apply_morphology:
            occupancy_grid = self.apply_morphological_operations(occupancy_grid, 'close')
            occupancy_grid = self.apply_morphological_operations(occupancy_grid, 'open')
        
        return original_image, processed_image, occupancy_grid
    
    def visualize_processing_steps(self, 
                                 original: np.ndarray,
                                 processed: np.ndarray,
                                 occupancy_grid: np.ndarray,
                                 save_path: Optional[str] = None):
        """
        Visualize the processing steps.
        
        Args:
            original: Original floor plan image
            processed: Processed/edge image
            occupancy_grid: Final occupancy grid
            save_path: Optional path to save the visualization
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        if len(original.shape) == 3:
            axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        else:
            axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original Floor Plan')
        axes[0].axis('off')
        
        # Processed/edges
        axes[1].imshow(processed, cmap='gray')
        axes[1].set_title('Processed Image')
        axes[1].axis('off')
        
        # Occupancy grid
        axes[2].imshow(occupancy_grid, cmap='RdYlBu_r')  # Red for obstacles, blue for free space
        axes[2].set_title(f'Occupancy Grid\nObstacle ratio: {np.sum(occupancy_grid)/occupancy_grid.size:.2%}')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def create_synthetic_floor_plan(width: int = 500, 
                              height: int = 500,
                              room_config: str = 'simple') -> np.ndarray:
    """
    Create a synthetic floor plan for testing.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        room_config: Configuration type ('simple', 'hallway', 'complex')
        
    Returns:
        Synthetic floor plan image (0=obstacle, 255=free space)
    """
    # Create white background (free space = 255)
    image = np.ones((height, width), dtype=np.uint8) * 255
    
    if room_config == 'simple':
        # Simple rectangular room with walls
        cv2.rectangle(image, (0, 0), (width-1, height-1), 0, 15)  # Border walls
        # Add some furniture
        cv2.rectangle(image, (100, 100), (200, 150), 0, -1)  # Table
        cv2.rectangle(image, (300, 200), (400, 300), 0, -1)  # Sofa
        
    elif room_config == 'hallway':
        # Hallway configuration
        cv2.rectangle(image, (0, 0), (width-1, height-1), 0, 15)  # Outer walls
        cv2.rectangle(image, (100, 200), (400, 250), 0, -1)  # Central obstacle
        cv2.rectangle(image, (200, 100), (250, 400), 0, -1)  # Vertical obstacle
        
    elif room_config == 'complex':
        # Complex multi-room layout
        # Outer walls
        cv2.rectangle(image, (0, 0), (width-1, height-1), 0, 15)
        # Internal walls
        cv2.line(image, (width//2, 20), (width//2, height//2), 0, 10)
        cv2.line(image, (20, height//2), (width//2, height//2), 0, 10)
        cv2.line(image, (width//2, height//2+50), (width-20, height//2+50), 0, 10)
        # Doors (gaps in walls)
        cv2.rectangle(image, (width//2-20, height//2-5), (width//2+20, height//2+5), 255, -1)
        # Furniture
        cv2.rectangle(image, (50, 50), (150, 100), 0, -1)
        cv2.rectangle(image, (300, 300), (450, 450), 0, -1)
    
    return image


if __name__ == "__main__":
    # Test the fixed processor
    processor = FloorPlanProcessor()
    
    # Create a synthetic floor plan for testing
    synthetic_plan = create_synthetic_floor_plan(room_config='complex')
    
    # Save the synthetic floor plan
    cv2.imwrite('fixed_synthetic_floor_plan.png', synthetic_plan)
    
    # Process the floor plan with edge detection disabled for synthetic images
    original, processed, occupancy_grid = processor.process_floor_plan(
        'fixed_synthetic_floor_plan.png',
        edge_method='none',  # Skip edge detection for synthetic images
        occupancy_method='threshold'
    )
    
    # Visualize results
    processor.visualize_processing_steps(original, processed, occupancy_grid)
    
    print(f"Fixed occupancy grid shape: {occupancy_grid.shape}")
    print(f"Obstacle ratio: {np.sum(occupancy_grid) / occupancy_grid.size:.2%}")
    print(f"Free space ratio: {1 - np.sum(occupancy_grid) / occupancy_grid.size:.2%}")
    
    # Test specific positions
    test_positions = [(100, 100), (200, 200), (400, 400)]
    for pos in test_positions:
        if 0 <= pos[0] < occupancy_grid.shape[0] and 0 <= pos[1] < occupancy_grid.shape[1]:
            status = "OBSTACLE" if occupancy_grid[pos] == 1 else "FREE"
            print(f"Position {pos}: {status}")