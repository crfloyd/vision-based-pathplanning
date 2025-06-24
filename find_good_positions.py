
import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.image_processing.floor_plan_processor import FloorPlanProcessor, create_synthetic_floor_plan
from src.path_planning.astar import AStarPathPlanner, PathPlanningVisualizer

def analyze_floor_plan_and_find_positions():
    """Analyze the synthetic floor plan and find good start/goal positions."""
    
    print("ANALYZING SYNTHETIC FLOOR PLAN")
    print("=" * 50)
    
    synthetic_image = create_synthetic_floor_plan(500, 500, 'complex')
    cv2.imwrite('analyze_synthetic.png', synthetic_image)
    
    processor = FloorPlanProcessor()
    original, processed, occupancy_grid = processor.process_floor_plan(
        'analyze_synthetic.png',
        edge_method='none',
        occupancy_method='threshold'
    )
    
    print(f"Floor plan analysis:")
    print(f"  Shape: {occupancy_grid.shape}")
    print(f"  Obstacle ratio: {np.sum(occupancy_grid) / occupancy_grid.size:.3f}")
    
    # Find all free space positions
    free_positions = np.where(occupancy_grid == 0)
    free_coords = list(zip(free_positions[0], free_positions[1]))
    
    print(f"  Total free positions: {len(free_coords)}")
    
    sample_positions = []
    
    # Try corners and edges (but away from walls)
    corner_candidates = [
        (30, 30), (30, 470), (470, 30), (470, 470),  # Near corners
        (100, 30), (400, 30), (30, 250), (470, 250), # Edge positions
        (250, 30), (250, 470), (100, 250), (400, 250) # Mid positions
    ]
    
    # Center area candidates
    center_candidates = [
        (120, 120), (150, 150), (200, 200), (300, 300),
        (350, 350), (380, 380), (150, 300), (300, 150),
        (180, 250), (320, 250), (250, 180), (250, 320)
    ]
    
    all_candidates = corner_candidates + center_candidates
    
    # Check which candidates are valid (free space)
    valid_positions = []
    for pos in all_candidates:
        if (0 <= pos[0] < occupancy_grid.shape[0] and 
            0 <= pos[1] < occupancy_grid.shape[1] and 
            occupancy_grid[pos] == 0):
            valid_positions.append(pos)
    
    print(f"  Valid candidate positions: {len(valid_positions)}")
    
    # If not enough valid positions, sample from all free positions
    if len(valid_positions) < 10:
        # Sample random free positions
        random_indices = np.random.choice(len(free_coords), min(20, len(free_coords)), replace=False)
        sampled_free = [free_coords[i] for i in random_indices]
        valid_positions.extend(sampled_free)
    
    # Test some start-goal pairs
    print(f"TESTING START-GOAL PAIRS")
    print("-" * 30)
    
    planner = AStarPathPlanner(allow_diagonal=True, heuristic='octile')
    successful_pairs = []
    
    # Try pairs with good distance between them
    for i in range(0, min(len(valid_positions), 10)):
        for j in range(i+5, min(len(valid_positions), 15)):  # Skip close positions
            start = valid_positions[i]
            goal = valid_positions[j]
            
            # Calculate distance to prefer longer paths
            distance = np.sqrt((start[0] - goal[0])**2 + (start[1] - goal[1])**2)
            if distance < 100:  # Skip if too close
                continue
            
            path, stats = planner.find_path(start, goal, occupancy_grid)
            
            if path:
                successful_pairs.append({
                    'start': start,
                    'goal': goal,
                    'distance': distance,
                    'path_length': len(path),
                    'path_cost': stats['path_cost'],
                    'path': path
                })
                print(f"{start} → {goal}: Distance={distance:.1f}, Path={len(path)} steps")
                
                if len(successful_pairs) >= 5:  # Found enough good pairs
                    break
        
        if len(successful_pairs) >= 5:
            break
    
    if not successful_pairs:
        print("No successful paths found")
        
        # show the occupancy grid with candidate positions
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(synthetic_image, cmap='gray')
        plt.title('Original Synthetic Image')
        for i, pos in enumerate(valid_positions[:10]):
            plt.plot(pos[1], pos[0], 'ro', markersize=8)
            plt.text(pos[1]+10, pos[0], f'{i}', color='red', fontweight='bold')
        
        plt.subplot(1, 2, 2)
        plt.imshow(occupancy_grid, cmap='RdYlBu_r')
        plt.title('Occupancy Grid (Red=Obstacle, Blue=Free)')
        for i, pos in enumerate(valid_positions[:10]):
            plt.plot(pos[1], pos[0], 'go', markersize=8)
            plt.text(pos[1]+10, pos[0], f'{i}', color='green', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('debug_positions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return None
    
    # Show the best paths
    print(f"\nBEST START-GOAL PAIRS")
    print("=" * 40)
    
    # Sort by path length (longer is more interesting)
    successful_pairs.sort(key=lambda x: x['path_length'], reverse=True)
    
    for i, pair in enumerate(successful_pairs[:3]):
        print(f"\n{i+1}. START: {pair['start']}, GOAL: {pair['goal']}")
        print(f"   Distance: {pair['distance']:.1f} pixels")
        print(f"   Path length: {pair['path_length']} steps")
        print(f"   Path cost: {pair['path_cost']:.2f}")
        print(f"   Command: python main.py --synthetic complex --start {pair['start'][0]} {pair['start'][1]} --goal {pair['goal'][0]} {pair['goal'][1]}")
    
    # Visualize the best path
    best_pair = successful_pairs[0]
    print(f"\nVISUALIZING BEST PATH")
    
    PathPlanningVisualizer.visualize_path(
        occupancy_grid, 
        best_pair['path'], 
        best_pair['start'], 
        best_pair['goal'],
        title=f"Best Path: {best_pair['start']} → {best_pair['goal']}"
    )
    
    return successful_pairs[0]

def quick_test_positions():
    
    print(f"\nQUICK TEST WITH MANUAL POSITIONS")
    print("=" * 40)
    
    # Create occupancy grid
    synthetic_image = create_synthetic_floor_plan(500, 500, 'complex')
    cv2.imwrite('quick_test.png', synthetic_image)
    
    processor = FloorPlanProcessor()
    original, processed, occupancy_grid = processor.process_floor_plan(
        'quick_test.png',
        edge_method='none',
        occupancy_method='threshold'
    )
    
    # Try positions that should be in open areas
    test_pairs = [
        ((75, 75), (425, 425)),      # Near corners but in open space
        ((125, 125), (375, 375)),    # Slightly more inward
        ((100, 200), (400, 300)),    # Different quadrants
        ((200, 100), (300, 400)),    # Cross navigation
        ((150, 150), (350, 350)),    # Central areas
    ]
    
    planner = AStarPathPlanner(allow_diagonal=True, heuristic='octile')
    
    for i, (start, goal) in enumerate(test_pairs):
        print(f"\nTest {i+1}: {start} → {goal}")
        
        # Check validity
        start_valid = (occupancy_grid[start] == 0) if (0 <= start[0] < 500 and 0 <= start[1] < 500) else False
        goal_valid = (occupancy_grid[goal] == 0) if (0 <= goal[0] < 500 and 0 <= goal[1] < 500) else False
        
        print(f"  Start valid: {start_valid}")
        print(f"  Goal valid: {goal_valid}")
        
        if start_valid and goal_valid:
            path, stats = planner.find_path(start, goal, occupancy_grid)
            if path:
                print(f"  SUCCESS! Path length: {len(path)}")
                return start, goal  # Return first successful pair
            else:
                print(f"  No path found")
        else:
            print(f"  Invalid positions")
    
    return None, None

if __name__ == "__main__":
    # First try the thorough analysis
    result = analyze_floor_plan_and_find_positions()
    
    if result is None:
        # If that fails, try quick manual test
        start, goal = quick_test_positions()
        if start and goal:
            print(f"\nFOUND WORKING POSITIONS:")
            print(f"python main.py --synthetic complex --start {start[0]} {start[1]} --goal {goal[0]} {goal[1]}")
        else:
            print(f"\nCould not find any working start-goal pairs")
            print(f"There might be an issue with the synthetic floor plan generation")