"""
Visual Navigation System - Main Integration Module
This is the main entry point that integrates image processing and path planning
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

# Import our custom modules
from src.image_processing.floor_plan_processor import FloorPlanProcessor, create_synthetic_floor_plan
from src.path_planning.path_planner import UnifiedPathPlanner, PathPlanningComparator


class VisualNavigationSystem:
    """
    Complete visual navigation system that processes floor plans and finds optimal paths.
    """
    
    def __init__(self, 
                 processor_config: Optional[Dict] = None,
                 planner_config: Optional[Dict] = None):
        """
        Initialize the visual navigation system.
        
        Args:
            processor_config: Configuration for floor plan processor
            planner_config: Configuration for path planner
        """
        # Default configurations
        default_processor_config = {
            'obstacle_threshold': 127,
            'blur_kernel_size': 5,
            'canny_low': 50,
            'canny_high': 150
        }
        
        default_planner_config = {
            'algorithm': 'astar',
            'allow_diagonal': True,
            'heuristic': 'octile'
        }
        
        # Update with user configurations
        if processor_config:
            default_processor_config.update(processor_config)
        if planner_config:
            default_planner_config.update(planner_config)
        
        # Initialize components
        self.processor = FloorPlanProcessor(**default_processor_config)
        self.unified_planner = UnifiedPathPlanner()
        self.planner_config = default_planner_config
        
        # Storage for results
        self.current_occupancy_grid = None
        self.current_original_image = None
        self.current_processed_image = None
        self.navigation_results = []
    
    def load_and_process_floor_plan(self, 
                                   image_path: str,
                                   edge_method: str = 'canny',
                                   occupancy_method: str = 'threshold',
                                   apply_morphology: bool = True) -> Dict:
        """
        Load and process a floor plan image.
        
        Args:
            image_path: Path to floor plan image
            edge_method: Edge detection method
            occupancy_method: Occupancy grid creation method
            apply_morphology: Whether to apply morphological operations
            
        Returns:
            Processing results dictionary
        """
        try:
            start_time = time.time()
            
            # Process the floor plan
            original, processed, occupancy_grid = self.processor.process_floor_plan(
                image_path, edge_method, occupancy_method, apply_morphology
            )
            
            processing_time = time.time() - start_time
            
            # Store results
            self.current_original_image = original
            self.current_processed_image = processed
            self.current_occupancy_grid = occupancy_grid
            
            # Calculate statistics
            total_pixels = occupancy_grid.size
            obstacle_pixels = np.sum(occupancy_grid)
            free_pixels = total_pixels - obstacle_pixels
            
            results = {
                'success': True,
                'processing_time': processing_time,
                'image_shape': original.shape,
                'grid_shape': occupancy_grid.shape,
                'obstacle_ratio': obstacle_pixels / total_pixels,
                'free_space_ratio': free_pixels / total_pixels,
                'total_pixels': total_pixels,
                'obstacle_pixels': int(obstacle_pixels),
                'free_pixels': int(free_pixels)
            }
            
            return results
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def plan_navigation(self, 
                       start: Tuple[int, int],
                       goal: Tuple[int, int],
                       algorithm: str = None,
                       smooth_path: bool = True) -> Dict:
        """
        Plan navigation from start to goal.
        
        Args:
            start: Start position (x, y)
            goal: Goal position (x, y)
            algorithm: Algorithm to use ('astar', 'dijkstra', or None for default)
            smooth_path: Whether to apply path smoothing (A* only)
            
        Returns:
            Navigation results dictionary
        """
        if self.current_occupancy_grid is None:
            return {
                'success': False,
                'error': 'No occupancy grid loaded. Process a floor plan first.'
            }
        
        try:
            start_time = time.time()
            
            # Use specified algorithm or default
            if algorithm is None:
                algorithm = self.planner_config['algorithm']
            
            # Get planner instance
            if algorithm == 'astar':
                planner = self.unified_planner.get_planner('astar', **{
                    k: v for k, v in self.planner_config.items() if k != 'algorithm'
                })
            else:
                planner = self.unified_planner.get_planner(algorithm, 
                    allow_diagonal=self.planner_config.get('allow_diagonal', True)
                )
            
            # Find path
            path, stats = planner.find_path(start, goal, self.current_occupancy_grid)
            
            if path is None:
                return {
                    'success': False,
                    'error': 'No path found',
                    'stats': stats
                }
            
            # Apply path smoothing if requested and using A*
            original_path = path.copy()
            if smooth_path and algorithm == 'astar':
                path = planner.smooth_path(path, self.current_occupancy_grid)
            
            planning_time = time.time() - start_time
            
            # Calculate path metrics
            path_length_euclidean = self._calculate_path_length(path)
            path_length_grid = len(path)
            
            results = {
                'success': True,
                'algorithm': algorithm,
                'path': path,
                'original_path': original_path,
                'start': start,
                'goal': goal,
                'planning_time': planning_time,
                'path_length_grid': path_length_grid,
                'path_length_euclidean': path_length_euclidean,
                'smoothing_reduction': len(original_path) - len(path) if smooth_path else 0,
                'stats': stats
            }
            
            # Store result for later analysis
            self.navigation_results.append(results)
            
            return results
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def compare_algorithms(self,
                          start: Tuple[int, int],
                          goal: Tuple[int, int]) -> Dict:
        """
        Compare different algorithms on the same navigation problem.
        
        Args:
            start: Start position (x, y)
            goal: Goal position (x, y)
            
        Returns:
            Comparison results dictionary
        """
        if self.current_occupancy_grid is None:
            return {
                'success': False,
                'error': 'No occupancy grid loaded. Process a floor plan first.'
            }
        
        try:
            # Use unified planner for comparison
            results = self.unified_planner.compare_algorithms(
                start, goal, self.current_occupancy_grid
            )
            
            return {
                'success': True,
                'results': results,
                'analysis': PathPlanningComparator.analyze_performance(results)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _calculate_path_length(self, path: List[Tuple[int, int]]) -> float:
        """Calculate the Euclidean length of a path."""
        if len(path) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            segment_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_length += segment_length
        
        return total_length
    
    def visualize_complete_pipeline(self, 
                                   navigation_result: Dict,
                                   save_path: Optional[str] = None,
                                   figsize: Tuple[int, int] = (20, 5)):
        """
        Visualize the complete processing and navigation pipeline.
        
        Args:
            navigation_result: Result from plan_navigation
            save_path: Optional path to save the visualization
            figsize: Figure size (width, height)
        """
        if not navigation_result['success']:
            print(f"Cannot visualize: {navigation_result['error']}")
            return
        
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        
        # Original floor plan
        if len(self.current_original_image.shape) == 3:
            axes[0].imshow(cv2.cvtColor(self.current_original_image, cv2.COLOR_BGR2RGB))
        else:
            axes[0].imshow(self.current_original_image, cmap='gray')
        axes[0].set_title('Original Floor Plan')
        axes[0].axis('off')
        
        # Processed image (edges)
        axes[1].imshow(self.current_processed_image, cmap='gray')
        axes[1].set_title('Edge Detection')
        axes[1].axis('off')
        
        # Occupancy grid
        axes[2].imshow(self.current_occupancy_grid, cmap='RdYlBu_r')
        axes[2].set_title('Occupancy Grid')
        axes[2].axis('off')
        
        # Navigation result
        self._plot_navigation_on_axis(axes[3], navigation_result)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _plot_navigation_on_axis(self, ax, navigation_result: Dict):
        """Plot navigation result on a given axis."""
        # Create visualization grid
        viz_grid = self.current_occupancy_grid.copy().astype(float)
        
        path = navigation_result['path']
        start = navigation_result['start']
        goal = navigation_result['goal']
        
        # Color path
        for x, y in path:
            viz_grid[x, y] = 0.6
        
        # Color start and goal
        viz_grid[start[0], start[1]] = 0.8
        viz_grid[goal[0], goal[1]] = 1.0
        
        ax.imshow(viz_grid, cmap='RdYlBu_r', origin='upper')
        
        # Add path line
        if len(path) > 1:
            path_y, path_x = zip(*path)
            ax.plot(path_x, path_y, 'g-', linewidth=2, alpha=0.8)
        
        # Add start and goal markers
        ax.plot(start[1], start[0], 'go', markersize=10, label='Start')
        ax.plot(goal[1], goal[0], 'ro', markersize=10, label='Goal')
        
        # Title with stats
        algorithm = navigation_result.get('algorithm', 'Unknown')
        title = f'{algorithm.upper()} Path Planning\nLength: {len(path)} steps'
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)


def create_test_scenarios() -> List[Dict]:
    """Create a set of test scenarios for benchmarking."""
    scenarios = [
        {
            'name': 'Simple Room',
            'synthetic_config': {
                'width': 300,
                'height': 300,
                'room_config': 'simple'
            },
            'navigation_tests': [
                {'start': (50, 50), 'goal': (250, 250)},
                {'start': (250, 50), 'goal': (50, 250)},
                {'start': (150, 50), 'goal': (150, 250)}
            ]
        },
        {
            'name': 'Hallway Navigation',
            'synthetic_config': {
                'width': 400,
                'height': 400,
                'room_config': 'hallway'
            },
            'navigation_tests': [
                {'start': (80, 80), 'goal': (320, 320)},
                {'start': (320, 80), 'goal': (80, 320)}
            ]
        },
        {
            'name': 'Complex Multi-room',
            'synthetic_config': {
                'width': 500,
                'height': 500,
                'room_config': 'complex'
            },
            'navigation_tests': [
                {'start': (100, 100), 'goal': (400, 400)},
                {'start': (400, 100), 'goal': (100, 400)},
                {'start': (250, 50), 'goal': (250, 450)}
            ]
        }
    ]
    
    return scenarios


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description='Visual Navigation System')
    parser.add_argument('--image', type=str, help='Path to floor plan image')
    parser.add_argument('--start', type=int, nargs=2, metavar=('X', 'Y'), 
                       help='Start position coordinates')
    parser.add_argument('--goal', type=int, nargs=2, metavar=('X', 'Y'),
                       help='Goal position coordinates')
    parser.add_argument('--algorithm', choices=['astar', 'dijkstra'], default='astar',
                       help='Path planning algorithm to use')
    parser.add_argument('--compare', action='store_true', 
                       help='Compare multiple algorithms')
    parser.add_argument('--synthetic', choices=['simple', 'hallway', 'complex'],
                       help='Create synthetic floor plan')
    
    args = parser.parse_args()
    
    # Initialize system
    navigation_system = VisualNavigationSystem()
    
    if args.synthetic:
        # Create and process synthetic floor plan
        synthetic_image = create_synthetic_floor_plan(room_config=args.synthetic)
        temp_path = f'synthetic_{args.synthetic}.png'
        cv2.imwrite(temp_path, synthetic_image)
        
        # For synthetic images, skip edge detection and process directly
        processing_result = navigation_system.load_and_process_floor_plan(
            temp_path,
            edge_method='none',
            occupancy_method='threshold',
            apply_morphology=True
        )
        print(f"Processing result: {processing_result}")
        
        if processing_result['success'] and args.start and args.goal:
            print("Processing successful, continuing to navigation...")
            
            if args.compare:
                # Compare algorithms
                print("Comparing algorithms...")
                comparison = navigation_system.compare_algorithms(args.start, args.goal)
                if comparison['success']:
                    PathPlanningComparator.print_comparison_report(comparison['results'])
                    PathPlanningComparator.visualize_comparison(
                        navigation_system.current_occupancy_grid,
                        comparison['results'],
                        args.start, args.goal
                    )
                else:
                    print(f"Algorithm comparison failed: {comparison['error']}")
            else:
                # Single algorithm
                print(f"Planning navigation with {args.algorithm}...")
                nav_result = navigation_system.plan_navigation(
                    tuple(args.start), tuple(args.goal), algorithm=args.algorithm, smooth_path=False
                )
                if nav_result['success']:
                    print("Navigation successful!")
                    print(f"   Path length: {nav_result['path_length_grid']} steps")
                    print(f"   Path cost: {nav_result['stats']['path_cost']:.2f}")
                    print(f"   Algorithm: {nav_result['algorithm']}")
                    
                    # Create visualization
                    print("Creating visualization...")
                    navigation_system.visualize_complete_pipeline(nav_result)
                    print("Visualization complete!")
                else:
                    print(f"Navigation failed: {nav_result['error']}")
        else:
            if not processing_result['success']:
                print(f"Processing failed: {processing_result.get('error', 'Unknown error')}")
            else:
                print("Missing start/goal coordinates. Use --start X Y --goal X Y")
        
    elif args.image:
        # Process provided image
        processing_result = navigation_system.load_and_process_floor_plan(args.image)
        print(f"Processing result: {processing_result}")
        
        if processing_result['success'] and args.start and args.goal:
            print("Processing successful, continuing to navigation...")
            
            if args.compare:
                # Compare algorithms
                print("Comparing algorithms...")
                comparison = navigation_system.compare_algorithms(args.start, args.goal)
                if comparison['success']:
                    PathPlanningComparator.print_comparison_report(comparison['results'])
                    PathPlanningComparator.visualize_comparison(
                        navigation_system.current_occupancy_grid,
                        comparison['results'],
                        args.start, args.goal
                    )
                else:
                    print(f"Algorithm comparison failed: {comparison['error']}")
            else:
                # Single algorithm
                print(f"Planning navigation with {args.algorithm}...")
                nav_result = navigation_system.plan_navigation(
                    tuple(args.start), tuple(args.goal), algorithm=args.algorithm, smooth_path=False
                )
                if nav_result['success']:
                    print("Navigation successful!")
                    print(f"   Path length: {nav_result['path_length_grid']} steps")
                    print(f"   Path cost: {nav_result['stats']['path_cost']:.2f}")
                    print(f"   Algorithm: {nav_result['algorithm']}")
                    
                    # Create visualization
                    print("Creating visualization...")
                    navigation_system.visualize_complete_pipeline(nav_result)
                    print("Visualization complete!")
                else:
                    print(f"Navigation failed: {nav_result['error']}")
        else:
            if not processing_result['success']:
                print(f"Processing failed: {processing_result.get('error', 'Unknown error')}")
            else:
                print("Missing start/goal coordinates. Use --start X Y --goal X Y")
    
    else:
        print("Please specify --image or --synthetic option")
        parser.print_help()


if __name__ == "__main__":
    main()