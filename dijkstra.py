# Importing the necessary libraries
import numpy as np
import cv2
import math
import heapq
import time
import matplotlib.pyplot as plt

# Defining the clearance (clearance = 5)
clr = 5

# Defining the size of canvas
width, height = 1200, 500

# Defining different colors to illustrate obstacles
cyan = [185, 0, 2]
blue = [255, 125, 0]

# Writing function to display the obstacles and clearance on the canvas
def Obstacles(obs):
    center = (650, 250)
    hexagon_side_length = 150
    y = 500 

    hexagon_vertices = []
    for i in range(6):
        angle_rad = math.radians(30 + 60 * i)  # 60 degrees between each vertex
        hexagon_x = int(center[0] + hexagon_side_length * math.cos(angle_rad))
        hexagon_y = int(center[1] + hexagon_side_length * math.sin(angle_rad))
        hexagon_vertices.append((hexagon_x, hexagon_y))

    # Fill the larger hexagon
    cv2.fillPoly(obs, [np.array(hexagon_vertices)], blue)

    # Smaller hexagon inside with clearance
    hexagon_side_length1 = 140
    hexagon_vertices1 = [(int(center[0] + hexagon_side_length1 * math.cos(math.radians(30 + 60 * i))),
                          int(center[1] + hexagon_side_length1 * math.sin(math.radians(30 + 60 * i))))
                         for i in range(6)]

    # Fill the smaller hexagon
    cv2.fillPoly(obs, [np.array(hexagon_vertices1)], cyan)

    rect_inside = [
        ((100 - clr, y - 100 + clr), (175 + clr, y - height)),
        ((275 - clr, y - 0), (350 + clr, y - 400 - clr)),
        ((980 - clr, y - 50 + clr), (1055 + clr, y - 450 - clr)),
        ((850 - clr, y - 50 + clr), (1055 + clr, y - 125 - clr)),
        ((850 - clr, y - 375 + clr), (1055 + clr, y - 450 - clr))
    ]

    rect_outside = [
        ((100, y - 100), (175, y - height)),
        ((275, y - 0), (350, y - 400)),
        ((980, y - 50), (1055, y - 450)),
        ((850, y - 50), (1055, y - 125)),
        ((850, y - 375), (1055, y - 450))
    ]

    for rect in rect_inside:
        cv2.rectangle(obs, rect[0], rect[1], blue, thickness=-1)
    
    for rect in rect_outside:
        cv2.rectangle(obs, rect[0], rect[1], cyan, thickness=-1)
    
    # Defining clearance on the borders
    obs[:clr, :] = blue  # top
    obs[-clr:, :] = blue  # bottom
    obs[:, :clr] = blue  # left
    obs[:, -clr:] = blue  # right

    return obs

# Initialize canvas
canvas = np.zeros((height, width, 3), dtype=np.uint8)

# Call Obstacles function to populate the canvas
obstacle_map = Obstacles(canvas.copy())

# Function to move up
def move_up(node):
    return node[0], node[1] + 1

# Function to move down
def move_down(node):
    return node[0], node[1] - 1

# Function to move right
def move_right(node):
    return node[0] + 1, node[1]

# Function to move left
def move_left(node):
    return node[0] - 1, node[1]

# Function to move up-right (diagonal)
def move_up_right(node):
    return node[0] + 1, node[1] + 1

# Function to move down-right (diagonal)
def move_down_right(node):
    return node[0] + 1, node[1] - 1

# Function to move up-left (diagonal)
def move_up_left(node):
    return node[0] - 1, node[1] + 1

# Function to move down-left (diagonal)
def move_down_left(node):
    return node[0] - 1, node[1] - 1


# Define the action set with corresponding functions
actions = {
    move_up: 1,
    move_down: 1,
    move_right: 1,
    move_left: 1,
    move_up_right: 1.4,
    move_down_right: 1.4,
    move_up_left: 1.4,
    move_down_left: 1.4
}

# Initialize start and goal nodes
while True:
    start_x = int(input("Enter start point, x (6-1194): "))
    start_y = int(input("Enter start point, y (6-494): "))
    start_node = (start_x, start_y)

    if start_node[0] < 6 or start_node[0] >= width or start_node[1] < 6 or start_node[1] >= height:
        print("Out of canvas!!! Provide new coordinates!!")
    elif ((obstacle_map[499 - start_node[1], start_node[0]])).all() or ((obstacle_map[499 - start_node[1], start_node[0]]) == cyan).all():
        print("Obstacle !!! Provide new coordinates!!")
    else:
        break

while True:
    goal_x = int(input("Enter end point, x (6-1194): "))
    goal_y = int(input("Enter end point, y (6-494): "))
    goal_node = (goal_x, goal_y)

    if goal_node[0] < 6 or goal_node[0] >= width or goal_node[1] < 6 or goal_node[1] >= height:
        print("Out of canvas!!! Provide new coordinates!!")
    elif ((obstacle_map[499 - goal_node[1], goal_node[0]])).all() or ((obstacle_map[499 - goal_node[1], goal_node[0]]) == cyan).all():
        print("Obstacle !!! Provide new coordinates!!")
    else:
        break

# Function to perform Dijkstra's algorithm
def dijkstra(start, goal, obstacle_map):
    start_time = time.time()
    distance_dict = {start: 0}
    visited = set()
    priority_queue = [(0, start)]
    # Initialize predecessor dictionary to store the path
    predecessor = {}

    while priority_queue:
        # Pop the node with the minimum distance
        current_distance, current_node = heapq.heappop(priority_queue)

        # If the current node is the goal, break
        if current_node == goal:
            break

        # Skip if the node is already visited
        if current_node in visited:
            continue

        # Mark current node as visited
        visited.add(current_node)

        # Get neighbors of the current node
        neighbors = []
        for action, cost in actions.items():
            neighbor = action(current_node)
            if 0 <= neighbor[0] < width and 0 <= neighbor[1] < height and obstacle_map[height - neighbor[1] - 1, neighbor[0]].tolist() != blue:
                canvas[height - neighbor[1] - 1, neighbor[0]] = (0, 255, 0)  # Change color of explored node
                neighbors.append((neighbor, cost))
                
        # Update distances to neighbors
        for neighbor, action_cost in neighbors:
            new_distance = distance_dict[current_node] + action_cost
            if neighbor not in distance_dict or new_distance < distance_dict[neighbor]:
                distance_dict[neighbor] = new_distance
                heapq.heappush(priority_queue, (new_distance, neighbor))
                predecessor[neighbor] = current_node

    # Measure time taken
    end_time = time.time()
    time_taken = end_time - start_time

    return goal, time_taken, predecessor

# Function to backtrack and find the optimal path
def backtrack_path(start, goal, predecessor):
    current = goal
    path = [current]
    while current != start:
        if current not in predecessor:
            print("No path found!")
            return []
        current = predecessor[current]
        path.append(current)
    return path[::-1]

# Function to visualize the exploration process and final path
def visualize_path(obstacle_map, goal_node, optimal_path):
    vis_map = obstacle_map.copy()
    
    # Visualize exploration path
    for node in optimal_path[:-1]:
        cv2.circle(vis_map, (node[0], height - node[1]), 3, (255, 0, 0), -1)  
        
    # Visualize final path
    for node in optimal_path:
        cv2.circle(vis_map, (node[0], height - node[1]), 3, (0, 255, 0), -1)  
    
    # Visualize start and goal nodes
    cv2.circle(vis_map, (start_node[0], height - start_node[1]), 5, (0, 0, 255), -1) 
    cv2.circle(vis_map, (goal_node[0], height - goal_node[1]), 5, (255, 0, 255), -1) 
    
    plt.imshow(cv2.cvtColor(vis_map, cv2.COLOR_BGR2RGB))
    plt.title("Optimal Path")
    plt.axis('off')
    plt.show()

# Call Dijkstra's algorithm
goal_node, time_taken, predecessor = dijkstra(start_node, goal_node, obstacle_map)

# Backtrack and find optimal path
optimal_path = backtrack_path(start_node, goal_node, predecessor)

# Function to calculate the total cost of the path
def calculate_path_cost(path, actions):
    total_cost = 0
    for i in range(len(path) - 1):
        action = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
        total_cost += actions.get(action, 1)  # Use get method to handle missing keys
    return total_cost

optimal_path_cost = calculate_path_cost(optimal_path, actions)

# Visualize exploration process and optimal path
visualize_path(obstacle_map, goal_node, optimal_path)

print("Time taken:", time_taken, "seconds")
# print("Final cost of the optimal path:", optimal_path_cost)

# Function to create a video of the exploration process and final path
def create_video(obstacle_map, start_node, goal_node, predecessor, optimal_path, frame_skip=10):
    vis_map = obstacle_map.copy()
    height, width, _ = vis_map.shape
    out = cv2.VideoWriter('dijkstra.avi', cv2.VideoWriter_fourcc(*'XVID'), 1000, (width, height))

    visited_nodes = set()
    
    for node in predecessor.keys():
        visited_nodes.add(node)
        current_node = node
        while current_node in predecessor:
            visited_nodes.add(predecessor[current_node])
            current_node = predecessor[current_node]

    frame_counter = 0

    # Visualize the path with frame skipping
    for node in optimal_path:
        cv2.circle(vis_map, (node[0], height - node[1]), 2, (255, 255, 255), 1)  # Final path in white

        if frame_counter % frame_skip == 0:
            out.write(vis_map)
        frame_counter += 1

    # Visualize start and goal nodes
    cv2.circle(vis_map, (start_node[0], height - start_node[1]), 5, (0, 0, 255), -1)  # Start node in red
    cv2.circle(vis_map, (goal_node[0], height - goal_node[1]), 5, (255, 0, 255), -1)  # Goal node in magenta
    # out.write(vis_map)

    out.release()
    cv2.destroyAllWindows()


# Create a video of the exploration process and final path
create_video(obstacle_map, start_node, goal_node, predecessor, optimal_path)


