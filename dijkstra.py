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
canvas = Obstacles(canvas.copy())

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
    elif ((canvas[499 - start_node[1], start_node[0]])).all() or ((canvas[499 - start_node[1], start_node[0]]) == cyan).all():
        print("Obstacle !!! Provide new coordinates!!")
    else:
        break

while True:
    goal_x = int(input("Enter end point, x (6-1194): "))
    goal_y = int(input("Enter end point, y (6-494): "))
    goal_node = (goal_x, goal_y)

    if goal_node[0] < 6 or goal_node[0] >= width or goal_node[1] < 6 or goal_node[1] >= height:
        print("Out of canvas!!! Provide new coordinates!!")
    elif ((canvas[499 - goal_node[1], goal_node[0]])).all() or ((canvas[499 - goal_node[1], goal_node[0]]) == cyan).all():
        print("Obstacle !!! Provide new coordinates!!")
    else:
        break
# Video writer to save the output
output_file = 'dijkstra_exploration.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 30
video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))


# Function to perform Dijkstra's algorithm
def dijkstra(start, goal, canvas):
    start_time = time.time()
    distance_dict = {start: 0}
    visited = set()
    priority_queue = [(0, start)]
    # Initialize predecessor dictionary to store the path
    predecessor = {}

    frame_counter = 0
    
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
            if 0 <= neighbor[0] < width and 0 <= neighbor[1] < height and canvas[height - neighbor[1] - 1, neighbor[0]].tolist() != blue:
                neighbors.append((neighbor, cost))

        # Update distances to neighbors
        for neighbor, action_cost in neighbors:
            new_distance = distance_dict[current_node] + action_cost
            if neighbor not in distance_dict or new_distance < distance_dict[neighbor]:
                distance_dict[neighbor] = new_distance
                heapq.heappush(priority_queue, (new_distance, neighbor))
                predecessor[neighbor] = current_node

                # Change color of explored node
                canvas[height - neighbor[1] - 1, neighbor[0]] = (0, 255, 0)

                # Draw exploration step on the canvas
                if frame_counter % 1000 == 0:
                    exploration_canvas = canvas.copy()
                    video_writer.write(exploration_canvas)
                
                frame_counter += 1

    # Write the final frame
    final_canvas = canvas.copy()
    video_writer.write(final_canvas)
    video_writer.release()
    
    # Measure time taken
    end_time = time.time()
    time_taken = end_time - start_time

    return goal, time_taken, predecessor

# Function to backtrack and find the optimal path
def backtrack_path(start, goal, predecessor, canvas):
    current_node = goal
    optimal_path = [current_node]

    while current_node != start:
        current_node = predecessor[current_node]
        optimal_path.append(current_node)

    optimal_path.reverse()

    # Draw the optimal path
    for node in optimal_path:
        cv2.circle(canvas, (node[0], height - node[1]), 1, (0, 0, 255), -1)

    return optimal_path

# Perform Dijkstra's algorithm and find the optimal path
goal, time_taken, predecessor = dijkstra(start_node, goal_node, canvas)
optimal_path = backtrack_path(start_node, goal, predecessor, canvas)

# Show the final path
plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
plt.title(f"Optimal Path from {start_node} to {goal_node} using Dijkstra's Algorithm")
plt.show()

print(f"Time taken for Dijkstra's algorithm: {time_taken:.2f} seconds")
