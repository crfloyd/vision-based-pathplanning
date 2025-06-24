"""
Dijkstra Path Planning Algorithm Implementation
Implements Dijkstra's algorithm for pathfinding on occupancy grids
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
    Node class for Dijkstra pathfinding algorithm.
    """
    x: int
    y: int
    g_cost: float = float('inf')  # Cost from start to current node
    parent: Optional['Node'] = None
    
    def __lt__(self, other):
        """Comparison for priority queue (based on g_cost only)"""
        return self.g_cost < other.g_cost
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))


class DijkstraPathPlanner:
    """
    Dijkstra's pathfinding algorithm for occupancy grid navigation.
    """
    
    def __init__(self, allow_diagonal: bool = True):
        """
        Initialize Dijkstra path planner.
        
        Args:
            allow_diagonal: Whether to allow diagonal movement
        """
        self.allow_diagonal = allow_diagonal
        
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
        Find path from start to goal using Dijkstra's algorithm.
        
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
            return None, {'error': 'Start position is invalid', 'algorithm': 'Dijkstra'}
        
        if not self.is_valid_position(goal[0], goal[1], occupancy_grid):
            return None, {'error': 'Goal position is invalid', 'algorithm': 'Dijkstra'}
        
        # Initialize nodes
        start_node = Node(start[0], start[1], g_cost=0)
        
        # Initialize data structures
        open_set = [start_node]  # Priority queue
        closed_set: Set[Tuple[int, int]] = set()
        g_costs = {(start[0], start[1]): 0.0}  # Best known g_costs
        
        # Statistics
        nodes_explored = 0
        nodes_expanded = 0
        
        while open_set:
            # Get node with lowest g_cost
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
                    'algorithm': 'Dijkstra',
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
                neighbor.parent = current_node
                
                heapq.heappush(open_set, neighbor)
        
        # No path found
        computation_time = time.time() - start_time
        stats = {
            'algorithm': 'Dijkstra',
            'nodes_explored': nodes_explored,
            'nodes_expanded': nodes_expanded,
            'path_length': 0,
            'path_cost': float('inf'),
            'computation_time': computation_time,
            'success': False
        }
        return None, stats


# Example usage and testing
if __name__ == "__main__":
    # Create a simple test occupancy grid
    grid_size = 50
    occupancy_grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    
    # Add some obstacles
    occupancy_grid[10:40, 15:17] = 1  # Vertical wall
    occupancy_grid[20:22, 5:35] = 1   # Horizontal wall
    occupancy_grid[25:45, 30:32] = 1  # Another vertical wall
    
    # Create Dijkstra planner
    planner = DijkstraPathPlanner(allow_diagonal=True)
    start = (5, 5)
    goal = (45, 45)
    
    path, stats = planner.find_path(start, goal, occupancy_grid)
    
    if path:
        print(f"Dijkstra path found! Length: {len(path)}, Cost: {stats['path_cost']:.2f}")
        print(f"Nodes explored: {stats['nodes_explored']}")
        print(f"Nodes expanded: {stats['nodes_expanded']}")
        print(f"Computation time: {stats['computation_time']*1000:.2f}ms")
        
        # Visualize results (using matplotlib)
        try:
            import matplotlib.pyplot as plt
            
            # Create visualization grid
            viz_grid = occupancy_grid.copy().astype(float)
            
            # Color path
            for x, y in path:
                viz_grid[x, y] = 0.6
            
            # Color start and goal
            viz_grid[start[0], start[1]] = 0.8
            viz_grid[goal[0], goal[1]] = 1.0
            
            plt.figure(figsize=(10, 10))
            plt.imshow(viz_grid, cmap='RdYlBu_r', origin='upper')
            
            # Add path line
            if len(path) > 1:
                path_y, path_x = zip(*path)
                plt.plot(path_x, path_y, 'g-', linewidth=3, alpha=0.7, label='Path')
            
            # Add start and goal markers
            plt.plot(start[1], start[0], 'go', markersize=15, label='Start')
            plt.plot(goal[1], goal[0], 'ro', markersize=15, label='Goal')
            
            plt.title("Dijkstra Path Planning Result")
            plt.legend()
            plt.colorbar(label='0=Free, 1=Obstacle')
            plt.grid(True, alpha=0.3)
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for visualization")
    else:
        print("No path found!")
        print(f"Error: {stats.get('error', 'Unknown error')}")