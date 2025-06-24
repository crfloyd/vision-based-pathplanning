"""
A* Path Planning Algorithm Implementation
Implements A* algorithm for pathfinding on occupancy grids
"""

import numpy as np
import heapq
from typing import List, Tuple, Optional, Set
import math
import time
from dataclasses import dataclass, field


@dataclass
class Node:
    """
    Node class for A* pathfinding algorithm.
    """
    x: int
    y: int
    g_cost: float = float('inf')  # Cost from start to current node
    h_cost: float = 0.0          # Heuristic cost from current node to goal
    f_cost: float = field(init=False)  # Total cost (g + h)
    parent: Optional['Node'] = None
    
    def __post_init__(self):
        self.f_cost = self.g_cost + self.h_cost
    
    def __lt__(self, other):
        """Comparison for priority queue"""
        if self.f_cost == other.f_cost:
            return self.h_cost < other.h_cost
        return self.f_cost < other.f_cost
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))


class AStarPathPlanner:
    """
    A* pathfinding algorithm for occupancy grid navigation.
    """
    
    def __init__(self, 
                 allow_diagonal: bool = True,
                 heuristic: str = 'euclidean',
                 tie_breaker: float = 1.01):
        """
        Initialize A* path planner.
        
        Args:
            allow_diagonal: Whether to allow diagonal movement
            heuristic: Heuristic function ('euclidean', 'manhattan', 'chebyshev', 'octile')
            tie_breaker: Small value to break ties in f-costs
        """
        self.allow_diagonal = allow_diagonal
        self.heuristic_name = heuristic
        self.tie_breaker = tie_breaker
        
        # Movement directions: 4-connected or 8-connected
        if allow_diagonal:
            self.directions = [
                (-1, -1), (-1, 0), (-1, 1),  # Top row
                (0, -1),           (0, 1),   # Middle row (skip center)
                (1, -1),  (1, 0),  (1, 1)    # Bottom row
            ]
            self.movement_costs = [
                math.sqrt(2), 1, math.sqrt(2),  # Diagonal, straight, diagonal
                1,               1,              # Straight, straight
                math.sqrt(2), 1, math.sqrt(2)   # Diagonal, straight, diagonal
            ]
        else:
            self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right
            self.movement_costs = [1, 1, 1, 1]
    
    def heuristic(self, node: Node, goal: Node) -> float:
        """
        Calculate heuristic distance between node and goal.
        
        Args:
            node: Current node
            goal: Goal node
            
        Returns:
            Heuristic distance
        """
        dx = abs(node.x - goal.x)
        dy = abs(node.y - goal.y)
        
        if self.heuristic_name == 'manhattan':
            return dx + dy
        elif self.heuristic_name == 'euclidean':
            return math.sqrt(dx*dx + dy*dy)
        elif self.heuristic_name == 'chebyshev':
            return max(dx, dy)
        elif self.heuristic_name == 'octile':  # Good for 8-connected grids
            return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)
        else:
            raise ValueError(f"Unknown heuristic: {self.heuristic_name}")
    
    def is_valid_position(self, x: int, y: int, occupancy_grid: np.ndarray) -> bool:
        """
        Check if a position is valid (within bounds and not an obstacle).
        
        Args:
            x, y: Grid coordinates
            occupancy_grid: Occupancy grid (0 = free, 1 = obstacle)
            
        Returns:
            True if position is valid
        """
        height, width = occupancy_grid.shape
        return (0 <= x < height and 
                0 <= y < width and 
                occupancy_grid[x, y] == 0)
    
    def get_neighbors(self, node: Node, occupancy_grid: np.ndarray) -> List[Tuple[Node, float]]:
        """
        Get valid neighbors of a node.
        
        Args:
            node: Current node
            occupancy_grid: Occupancy grid
            
        Returns:
            List of (neighbor_node, movement_cost) tuples
        """
        neighbors = []
        
        for i, (dx, dy) in enumerate(self.directions):
            new_x, new_y = node.x + dx, node.y + dy
            
            if self.is_valid_position(new_x, new_y, occupancy_grid):
                # Additional check for diagonal movement to prevent corner cutting
                if self.allow_diagonal and abs(dx) == 1 and abs(dy) == 1:
                    # Check if both adjacent cells are free (prevent diagonal through corners)
                    if (occupancy_grid[node.x + dx, node.y] == 1 or 
                        occupancy_grid[node.x, node.y + dy] == 1):
                        continue
                
                neighbor = Node(new_x, new_y)
                movement_cost = self.movement_costs[i]
                neighbors.append((neighbor, movement_cost))
        
        return neighbors
    
    def reconstruct_path(self, node: Node) -> List[Tuple[int, int]]:
        """
        Reconstruct path from goal node to start node.
        
        Args:
            node: Goal node with parent chain
            
        Returns:
            List of (x, y) coordinates representing the path
        """
        path = []
        current = node
        
        while current is not None:
            path.append((current.x, current.y))
            current = current.parent
        
        return path[::-1]  # Reverse to get path from start to goal
    
    def find_path(self, 
                  start: Tuple[int, int], 
                  goal: Tuple[int, int],
                  occupancy_grid: np.ndarray) -> Tuple[Optional[List[Tuple[int, int]]], dict]:
        """
        Find path from start to goal using A* algorithm.
        
        Args:
            start: Start position (x, y)
            goal: Goal position (x, y)
            occupancy_grid: Occupancy grid (0 = free, 1 = obstacle)
            
        Returns:
            Tuple of (path, statistics) where path is list of coordinates or None if no path found
        """
        start_time = time.time()
        
        # Validate inputs
        if not self.is_valid_position(start[0], start[1], occupancy_grid):
            return None, {'error': 'Start position is invalid', 'algorithm': 'A*'}
        
        if not self.is_valid_position(goal[0], goal[1], occupancy_grid):
            return None, {'error': 'Goal position is invalid', 'algorithm': 'A*'}
        
        # Initialize nodes
        start_node = Node(start[0], start[1], g_cost=0)
        goal_node = Node(goal[0], goal[1])
        start_node.h_cost = self.heuristic(start_node, goal_node)
        start_node.f_cost = start_node.g_cost + start_node.h_cost
        
        # Initialize data structures
        open_set = [start_node]  # Priority queue
        closed_set: Set[Tuple[int, int]] = set()
        g_costs = {(start[0], start[1]): 0.0}  # Best known g_costs
        
        # Statistics
        nodes_explored = 0
        nodes_expanded = 0
        
        while open_set:
            # Get node with lowest f_cost
            current_node = heapq.heappop(open_set)
            current_pos = (current_node.x, current_node.y)
            
            # Skip if we've already processed this node
            if current_pos in closed_set:
                continue
            
            closed_set.add(current_pos)
            nodes_explored += 1
            
            # Check if we reached the goal
            if current_pos == goal:
                path = self.reconstruct_path(current_node)
                computation_time = time.time() - start_time
                stats = {
                    'algorithm': 'A*',
                    'heuristic': self.heuristic_name,
                    'nodes_explored': nodes_explored,
                    'nodes_expanded': nodes_expanded,
                    'path_length': len(path),
                    'path_cost': current_node.g_cost,
                    'computation_time': computation_time,
                    'success': True
                }
                return path, stats
            
            # Explore neighbors
            neighbors = self.get_neighbors(current_node, occupancy_grid)
            nodes_expanded += 1
            
            for neighbor, movement_cost in neighbors:
                neighbor_pos = (neighbor.x, neighbor.y)
                
                # Skip if already in closed set
                if neighbor_pos in closed_set:
                    continue
                
                # Calculate tentative g_cost
                tentative_g = current_node.g_cost + movement_cost
                
                # Skip if we've found a better path to this neighbor
                if neighbor_pos in g_costs and tentative_g >= g_costs[neighbor_pos]:
                    continue
                
                # This is the best path to neighbor so far
                g_costs[neighbor_pos] = tentative_g
                neighbor.g_cost = tentative_g
                neighbor.h_cost = self.heuristic(neighbor, goal_node) * self.tie_breaker
                neighbor.f_cost = neighbor.g_cost + neighbor.h_cost
                neighbor.parent = current_node
                
                heapq.heappush(open_set, neighbor)
        
        # No path found
        computation_time = time.time() - start_time
        stats = {
            'algorithm': 'A*',
            'heuristic': self.heuristic_name,
            'nodes_explored': nodes_explored,
            'nodes_expanded': nodes_expanded,
            'path_length': 0,
            'path_cost': float('inf'),
            'computation_time': computation_time,
            'success': False
        }
        return None, stats
    
    def smooth_path(self, path: List[Tuple[int, int]], 
                   occupancy_grid: np.ndarray,
                   max_iterations: int = 100) -> List[Tuple[int, int]]:
        """
        Smooth the path by removing unnecessary waypoints using line-of-sight checks.
        
        Args:
            path: Original path as list of coordinates
            occupancy_grid: Occupancy grid for collision checking
            max_iterations: Maximum number of smoothing iterations
            
        Returns:
            Smoothed path
        """
        if len(path) <= 2:
            return path
        
        smoothed_path = path.copy()
        
        for _ in range(max_iterations):
            if len(smoothed_path) <= 2:
                break
                
            improved = False
            i = 0
            
            while i < len(smoothed_path) - 2:
                start_pos = smoothed_path[i]
                end_pos = smoothed_path[i + 2]
                
                # Check if we can connect start_pos directly to end_pos
                if self._has_line_of_sight(start_pos, end_pos, occupancy_grid):
                    # Remove the intermediate point
                    smoothed_path.pop(i + 1)
                    improved = True
                else:
                    i += 1
            
            if not improved:
                break
        
        return smoothed_path
    
    def _has_line_of_sight(self, start: Tuple[int, int], end: Tuple[int, int],
                          occupancy_grid: np.ndarray) -> bool:
        """
        Check if there's a clear line of sight between two points using Bresenham's algorithm.
        
        Args:
            start: Start position (x, y)
            end: End position (x, y)
            occupancy_grid: Occupancy grid
            
        Returns:
            True if line of sight is clear
        """
        x0, y0 = start
        x1, y1 = end
        
        # Bresenham's line algorithm
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        
        x_step = 1 if x0 < x1 else -1
        y_step = 1 if y0 < y1 else -1
        
        error = dx - dy
        x, y = x0, y0
        
        while True:
            # Check if current position is an obstacle
            if not self.is_valid_position(x, y, occupancy_grid):
                return False
            
            if x == x1 and y == y1:
                break
            
            error2 = 2 * error
            
            if error2 > -dy:
                error -= dy
                x += x_step
            
            if error2 < dx:
                error += dx
                y += y_step
        
        return True


class PathPlanningVisualizer:
    """
    Utility class for visualizing path planning results.
    """
    
    @staticmethod
    def visualize_path(occupancy_grid: np.ndarray,
                      path: Optional[List[Tuple[int, int]]] = None,
                      start: Optional[Tuple[int, int]] = None,
                      goal: Optional[Tuple[int, int]] = None,
                      explored_nodes: Optional[Set[Tuple[int, int]]] = None,
                      title: str = "Path Planning Result"):
        """
        Visualize the occupancy grid with path and waypoints.
        
        Args:
            occupancy_grid: Binary occupancy grid
            path: Found path as list of coordinates
            start: Start position
            goal: Goal position
            explored_nodes: Set of explored nodes for visualization
            title: Plot title
        """
        import matplotlib.pyplot as plt
        
        # Create visualization grid
        viz_grid = occupancy_grid.copy().astype(float)
        
        # Color explored nodes
        if explored_nodes:
            for x, y in explored_nodes:
                if 0 <= x < viz_grid.shape[0] and 0 <= y < viz_grid.shape[1]:
                    if viz_grid[x, y] == 0:  # Only color free space
                        viz_grid[x, y] = 0.3
        
        # Color path
        if path:
            for x, y in path:
                viz_grid[x, y] = 0.6
        
        # Color start and goal
        if start:
            viz_grid[start[0], start[1]] = 0.8
        if goal:
            viz_grid[goal[0], goal[1]] = 1.0
        
        plt.figure(figsize=(10, 10))
        plt.imshow(viz_grid, cmap='RdYlBu_r', origin='upper')
        
        # Add path line
        if path and len(path) > 1:
            path_y, path_x = zip(*path)
            plt.plot(path_x, path_y, 'g-', linewidth=3, alpha=0.7, label='Path')
        
        # Add start and goal markers
        if start:
            plt.plot(start[1], start[0], 'go', markersize=15, label='Start')
        if goal:
            plt.plot(goal[1], goal[0], 'ro', markersize=15, label='Goal')
        
        plt.title(title)
        plt.legend()
        plt.colorbar(label='0=Free, 1=Obstacle')
        plt.grid(True, alpha=0.3)
        plt.show()


if __name__ == "__main__":
    # Create a simple test occupancy grid
    grid_size = 50
    occupancy_grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    
    # Add some obstacles
    occupancy_grid[10:40, 15:17] = 1  # Vertical wall
    occupancy_grid[20:22, 5:35] = 1   # Horizontal wall
    occupancy_grid[25:45, 30:32] = 1  # Another vertical wall
    
    # Create A* planner
    planner = AStarPathPlanner(allow_diagonal=True, heuristic='octile')
    
    # Define start and goal
    start = (5, 5)
    goal = (45, 45)
    
    # Find path
    path, stats = planner.find_path(start, goal, occupancy_grid)
    
    if path:
        print(f"Path found! Length: {len(path)}, Cost: {stats['path_cost']:.2f}")
        print(f"Nodes explored: {stats['nodes_explored']}")
        print(f"Nodes expanded: {stats['nodes_expanded']}")
        print(f"Computation time: {stats['computation_time']*1000:.2f}ms")
        
        # Smooth the path
        smoothed_path = planner.smooth_path(path, occupancy_grid)
        print(f"Smoothed path length: {len(smoothed_path)}")
        
        # Visualize results
        PathPlanningVisualizer.visualize_path(
            occupancy_grid, path, start, goal,
            title="A* Path Planning Result"
        )
        
        PathPlanningVisualizer.visualize_path(
            occupancy_grid, smoothed_path, start, goal,
            title="Smoothed Path"
        )
    else:
        print("No path found!")
        print(f"Error: {stats.get('error', 'Unknown error')}")
        
        # Visualize the problem
        PathPlanningVisualizer.visualize_path(
            occupancy_grid, None, start, goal,
            title="No Path Found"
        )