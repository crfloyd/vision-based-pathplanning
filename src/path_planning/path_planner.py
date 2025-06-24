"""
Unified Path Planning Interface
Provides a common interface for different path planning algorithms
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod

from .astar import AStarPathPlanner
from .dijkstra import DijkstraPathPlanner


class PathPlannerInterface(ABC):
    """
    Abstract base class for path planning algorithms.
    """
    
    @abstractmethod
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int],
                  occupancy_grid: np.ndarray) -> Tuple[Optional[List[Tuple[int, int]]], Dict[str, Any]]:
        """
        Find path from start to goal.
        
        Args:
            start: Start position (x, y)
            goal: Goal position (x, y)
            occupancy_grid: Occupancy grid (0 = free, 1 = obstacle)
            
        Returns:
            Tuple of (path, statistics)
        """
        pass


class UnifiedPathPlanner:
    """
    Unified interface for accessing different path planning algorithms.
    """
    
    def __init__(self):
        """Initialize the unified path planner."""
        self.algorithms = {
            'astar': None,
            'dijkstra': None
        }
    
    def get_planner(self, algorithm: str, **kwargs) -> PathPlannerInterface:
        """
        Get a path planning algorithm instance.
        
        Args:
            algorithm: Algorithm name ('astar', 'dijkstra')
            **kwargs: Algorithm-specific parameters
            
        Returns:
            Path planner instance
        """
        algorithm = algorithm.lower()
        
        if algorithm == 'astar':
            return AStarPathPlanner(**kwargs)
        elif algorithm == 'dijkstra':
            return DijkstraPathPlanner(**kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: 'astar', 'dijkstra'")
    
    def compare_algorithms(self, 
                          start: Tuple[int, int],
                          goal: Tuple[int, int],
                          occupancy_grid: np.ndarray,
                          algorithms: Optional[List[str]] = None,
                          algorithm_configs: Optional[Dict[str, Dict]] = None) -> Dict[str, Tuple[Optional[List[Tuple[int, int]]], Dict[str, Any]]]:
        """
        Compare multiple path planning algorithms on the same problem.
        
        Args:
            start: Start position (x, y)
            goal: Goal position (x, y)
            occupancy_grid: Occupancy grid
            algorithms: List of algorithm names to compare
            algorithm_configs: Configuration for each algorithm
            
        Returns:
            Dictionary mapping algorithm names to (path, stats) tuples
        """
        if algorithms is None:
            algorithms = ['astar', 'dijkstra']
        
        if algorithm_configs is None:
            algorithm_configs = {
                'astar': {'allow_diagonal': True, 'heuristic': 'octile'},
                'dijkstra': {'allow_diagonal': True}
            }
        
        results = {}
        
        for algorithm in algorithms:
            try:
                config = algorithm_configs.get(algorithm, {})
                planner = self.get_planner(algorithm, **config)
                path, stats = planner.find_path(start, goal, occupancy_grid)
                results[algorithm] = (path, stats)
            except Exception as e:
                results[algorithm] = (None, {'error': str(e), 'success': False})
        
        return results
    
    def find_best_path(self,
                      start: Tuple[int, int],
                      goal: Tuple[int, int],
                      occupancy_grid: np.ndarray,
                      criterion: str = 'path_cost') -> Tuple[str, Optional[List[Tuple[int, int]]], Dict[str, Any]]:
        """
        Find the best path using multiple algorithms and select based on criterion.
        
        Args:
            start: Start position (x, y)
            goal: Goal position (x, y)
            occupancy_grid: Occupancy grid
            criterion: Selection criterion ('path_cost', 'computation_time', 'path_length')
            
        Returns:
            Tuple of (best_algorithm, best_path, best_stats)
        """
        results = self.compare_algorithms(start, goal, occupancy_grid)
        
        best_algorithm = None
        best_path = None
        best_stats = None
        best_value = float('inf') if criterion in ['path_cost', 'computation_time'] else float('inf')
        
        for algorithm, (path, stats) in results.items():
            if path is None or not stats.get('success', False):
                continue
            
            if criterion == 'path_cost':
                value = stats.get('path_cost', float('inf'))
            elif criterion == 'computation_time':
                value = stats.get('computation_time', float('inf'))
            elif criterion == 'path_length':
                value = stats.get('path_length', float('inf'))
            else:
                raise ValueError(f"Unknown criterion: {criterion}")
            
            if value < best_value:
                best_value = value
                best_algorithm = algorithm
                best_path = path
                best_stats = stats
        
        return best_algorithm, best_path, best_stats


class PathPlanningComparator:
    """
    Utility class for comparing and analyzing path planning algorithms.
    """
    
    @staticmethod
    def analyze_performance(results: Dict[str, Tuple[Optional[List[Tuple[int, int]]], Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Analyze performance metrics across algorithms.
        
        Args:
            results: Results from compare_algorithms
            
        Returns:
            Performance analysis dictionary
        """
        analysis = {
            'algorithms': list(results.keys()),
            'success_rates': {},
            'average_metrics': {},
            'best_performers': {}
        }
        
        successful_results = {alg: (path, stats) for alg, (path, stats) in results.items() 
                            if path is not None and stats.get('success', False)}
        
        # Calculate success rates
        for algorithm in results.keys():
            path, stats = results[algorithm]
            analysis['success_rates'][algorithm] = 1.0 if (path is not None and stats.get('success', False)) else 0.0
        
        if not successful_results:
            return analysis
        
        # Calculate average metrics
        metrics = ['path_cost', 'computation_time', 'path_length', 'nodes_explored']
        
        for metric in metrics:
            values = []
            for algorithm, (path, stats) in successful_results.items():
                if metric in stats:
                    values.append((algorithm, stats[metric]))
            
            if values:
                analysis['average_metrics'][metric] = {
                    'values': {alg: val for alg, val in values},
                    'best': min(values, key=lambda x: x[1]),
                    'worst': max(values, key=lambda x: x[1]),
                    'average': sum(val for _, val in values) / len(values)
                }
        
        # Identify best performers
        for metric in metrics:
            if metric in analysis['average_metrics']:
                best_alg, best_val = analysis['average_metrics'][metric]['best']
                analysis['best_performers'][metric] = best_alg
        
        return analysis
    
    @staticmethod
    def print_comparison_report(results: Dict[str, Tuple[Optional[List[Tuple[int, int]]], Dict[str, Any]]]):
        """
        Print a detailed comparison report.
        
        Args:
            results: Results from compare_algorithms
        """
        print("=" * 60)
        print("PATH PLANNING ALGORITHM COMPARISON REPORT")
        print("=" * 60)
        
        analysis = PathPlanningComparator.analyze_performance(results)
        
        # Success rates
        print("\nSUCCESS RATES:")
        print("-" * 30)
        for algorithm, success_rate in analysis['success_rates'].items():
            status = "✅ SUCCESS" if success_rate == 1.0 else "FAILED"
            print(f"{algorithm.upper():<12}: {status}")
        
        # Performance metrics
        successful_results = {alg: (path, stats) for alg, (path, stats) in results.items() 
                            if path is not None and stats.get('success', False)}
        
        if successful_results:
            print("\nPERFORMANCE METRICS:")
            print("-" * 50)
            
            # Header
            algorithms = list(successful_results.keys())
            header = f"{'Metric':<20}"
            for alg in algorithms:
                header += f"{alg.upper():<12}"
            print(header)
            print("-" * 50)
            
            # Metrics rows
            metrics = [
                ('Path Cost', 'path_cost', '.2f'),
                ('Path Length', 'path_length', 'd'),
                ('Nodes Explored', 'nodes_explored', 'd'),
                ('Time (ms)', 'computation_time', '.2f')
            ]
            
            for metric_name, metric_key, format_spec in metrics:
                row = f"{metric_name:<20}"
                for algorithm in algorithms:
                    _, stats = successful_results[algorithm]
                    if metric_key in stats:
                        value = stats[metric_key]
                        if metric_key == 'computation_time':
                            value *= 1000  # Convert to milliseconds
                        row += f"{value:<12{format_spec}}"
                    else:
                        row += f"{'N/A':<12}"
                print(row)
        
        # Best performers
        if 'best_performers' in analysis and analysis['best_performers']:
            print("\nBEST PERFORMERS:")
            print("-" * 30)
            for metric, best_algorithm in analysis['best_performers'].items():
                print(f"{metric.replace('_', ' ').title():<20}: {best_algorithm.upper()}")
        
        print("\n" + "=" * 60)
    
    @staticmethod
    def visualize_comparison(occupancy_grid: np.ndarray,
                           results: Dict[str, Tuple[Optional[List[Tuple[int, int]]], Dict[str, Any]]],
                           start: Tuple[int, int],
                           goal: Tuple[int, int],
                           save_path: Optional[str] = None):
        """
        Visualize comparison of multiple algorithms side by side.
        
        Args:
            occupancy_grid: Binary occupancy grid
            results: Results from compare_algorithms
            start: Start position
            goal: Goal position
            save_path: Optional path to save the figure
        """
        try:
            import matplotlib.pyplot as plt
            
            successful_results = {alg: (path, stats) for alg, (path, stats) in results.items() 
                                if path is not None and stats.get('success', False)}
            
            if not successful_results:
                print("No successful results to visualize")
                return
            
            n_algorithms = len(successful_results)
            fig, axes = plt.subplots(1, n_algorithms, figsize=(5*n_algorithms, 5))
            
            if n_algorithms == 1:
                axes = [axes]
            
            for i, (algorithm_name, (path, stats)) in enumerate(successful_results.items()):
                # Create visualization grid
                viz_grid = occupancy_grid.copy().astype(float)
                
                # Color path
                if path:
                    for x, y in path:
                        viz_grid[x, y] = 0.6
                
                # Color start and goal
                viz_grid[start[0], start[1]] = 0.8
                viz_grid[goal[0], goal[1]] = 1.0
                
                axes[i].imshow(viz_grid, cmap='RdYlBu_r', origin='upper')
                
                # Add path line
                if path and len(path) > 1:
                    path_y, path_x = zip(*path)
                    axes[i].plot(path_x, path_y, 'g-', linewidth=2, alpha=0.8)
                
                # Add start and goal markers
                axes[i].plot(start[1], start[0], 'go', markersize=10, label='Start')
                axes[i].plot(goal[1], goal[0], 'ro', markersize=10, label='Goal')
                
                # Title with stats
                title = f"{algorithm_name.upper()}\nLength: {stats['path_length']}\n"
                title += f"Cost: {stats['path_cost']:.2f}\n"
                title += f"Time: {stats['computation_time']*1000:.1f}ms"
                
                axes[i].set_title(title)
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for visualization")


# Example usage
if __name__ == "__main__":
    # Create test scenario
    grid_size = 50
    occupancy_grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    
    # Add obstacles
    occupancy_grid[10:40, 15:17] = 1  # Vertical wall
    occupancy_grid[20:22, 5:35] = 1   # Horizontal wall
    occupancy_grid[25:45, 30:32] = 1  # Another vertical wall
    
    start = (5, 5)
    goal = (45, 45)
    
    # Initialize unified planner
    unified_planner = UnifiedPathPlanner()
    
    # Compare algorithms
    print("Comparing path planning algorithms...")
    results = unified_planner.compare_algorithms(start, goal, occupancy_grid)
    
    # Print detailed report
    PathPlanningComparator.print_comparison_report(results)
    
    # Visualize comparison
    PathPlanningComparator.visualize_comparison(occupancy_grid, results, start, goal)
    
    # Find best path
    best_algorithm, best_path, best_stats = unified_planner.find_best_path(
        start, goal, occupancy_grid, criterion='path_cost'
    )
    
    if best_path:
        print(f"\n Best algorithm for path cost: {best_algorithm.upper()}")
        print(f"   Path cost: {best_stats['path_cost']:.2f}")
        print(f"   Path length: {best_stats['path_length']} steps")
    else:
        print("\n No successful paths found")