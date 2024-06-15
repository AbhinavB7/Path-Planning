# github Link : https://github.com/KomK2/ENPM661/tree/project3

# ENPM 661 - Planning for Autonomous Robots
# Project 3 : Implementing A* Algorithm 
# Author : Kiran Kommaraju, Abhinav Bhamidipati
# UID : komkiran, abhinav7

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
import cv2
from queue import PriorityQueue
import math
import time

# Node class
class Node :
    def __init__ (self, points, orientation,  parent, cost, heuristic):
        self.points = points
        self.orientation = orientation
        self.parent = parent
        self.cost = cost
        self.heuristic = heuristic
        self.total_cost = cost + heuristic

    # Points of the Node
    def getPoints(self):
        return self.points
    
    # Orientation of the Node
    def getOrientation(self):
        return self.orientation
    
    # Parent Node
    def getParent(self):
        return self.parent
    
    # Check if two nodes are equal
    def __eq__(self, other):
        return self.points[0] == other.points[0] and self.points[1] == other.points[1]
    
    def __hash__ (self):
        return hash((self.points[0], self.points[1]))
    
    def __lt__(self, other):
        return self.total_cost < other.total_cost

# Constants
y = 500
goal = None
start_point = None
canvas = np.ones((500, 1200, 3), dtype=np.uint8 )

# color for exploration of nodes
new_color = (255, 0, 0)

# set clearance and step size
clearance = 0
stepSize = 0

# Video writer to save the output
output_file = 'a_star.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 1000
width, height = 1200,500
video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

# Initializing Variables
def initalize_varaibles():
    global clearance
    global stepSize
    
    # Input from the user
    clearance = int (input("Enter the clearance in mm: "))
    stepSize = int (input("Enter the step size between 1 and 10 : "))

# Calculating vertices of the hexagon
def hexagon_vertex() :
    hexagon_vertex = []
    for i in range(6):
        angle_rad = math.radians(30 + 60 * i)  # 60 degrees between each vertex
        x_coordinate = int(650 + 150 * math.cos(angle_rad))
        y_coordinate = int(250 + 150 * math.sin(angle_rad))
        hexagon_vertex.append((x_coordinate, y_coordinate))
    return hexagon_vertex

# Shape definitions
hexagon_points = np.array(hexagon_vertex())
u_shape = np.array([[1200-100-200,y-(450)],[1200-100,y-450],[1200-100,y-50],[1200-100-200,y-(50)],[1200-100-200,y-75-50],[1200-100-200+120,y-75-50],[1200-100-200+120,y-(375)],[1200-100-200,y-375] ])

# Drawing Hexagon
cv2.polylines(canvas, [hexagon_points], True, (255, 0, 255), 5)
cv2.fillPoly(canvas, [hexagon_points], (0, 0, 255))

# Rectangle 1
cv2.rectangle(canvas, (100, y-500), (175, y-100), (255, 0, 255), 5 )
cv2.rectangle(canvas, (100, y-500), (175, y-100), (0, 0, 255), -1)

# Rectangle 2
cv2.rectangle(canvas, (100+75+100, y-400), (100+75+100+75, y), (255, 0, 255), 5)
cv2.rectangle(canvas, (100+75+100, y-400), (100+75+100+75, y), (0, 0, 255), -1)

# C shape
cv2.polylines(canvas, [u_shape], True, (255, 0, 255), 5)
cv2.fillPoly(canvas, [u_shape],(0, 0, 255))


# display the canvas
def show_image():
    plt.imshow( canvas)
    plt.show()

# check if point is inside the rectangle
def is_in_rectangle(top_left, bottom_right, point):
    if point[0] > top_left[0] and point[0] < bottom_right[0] and point[1] > top_left[1] and point[1] < bottom_right[1]:
        return True
    else:
        return False

# Check if point is inside the obstacle space    
def obstacle_space(point):
    # Check if point is inside the hexagon
    hexagon_path = mplPath.Path(hexagon_points)
    if hexagon_path.contains_point(point):
        return True
    # Check if point is inside the rectangle
    if is_in_rectangle((100, y-500), (175, y-100), point):
        return True
    # Check if point is inside the rectangle
    if is_in_rectangle((100+75+100, y-400), (100+75+100+75, y), point):
        return True
    # Check if point is inside the U shape
    u_shape_path = mplPath.Path(u_shape)
    if u_shape_path.contains_point(point):
        return True
    return False

# Checking if the point can move
def canMove(point):
    if point[0] < 6 or point[0] > 1195 or point[1] < 6 or point[1] > 495:
        return False 
    point_color = canvas[point[1], point[0]]
    if point_color[0] == 1 and point_color[1] == 1 and point_color[2] == 1:
        return True
    if point_color[0] == 255 and point_color[1] == 0 and point_color[2] == 0:
        return True
    return False

# Start and End points from the user
def start_end_goals():
    initial_point = int(input("Enter the x coordinate of the initial point: ")), 500 - int(input("Enter the y coordinate of the initial point: "))
    inital_orentation = int(input("Enter the orientation of robot at initial point ( multiple of 30 ): "))

    goal_point = int(input("Enter the x coordinate of the goal point: ")), 500 - int(input("Enter the y coordinate of the goal point: "))
    goal_orentation = int(input("Enter the orientation of robot at goal point ( multiple of 30 ): "))

    if canMove(initial_point) and canMove(goal_point):
        return initial_point, goal_point , inital_orentation, goal_orentation
    else:
        print("Invalid points. Please enter valid points.")
        return start_end_goals()

# Function to move Straight
def moveStraight(node,step_size):
    current = node.getPoints()
    orientation = node.getOrientation()

    new_x = current[0] + int(round((math.cos(math.radians(orientation))*step_size)))
    new_y = current[1] + int(round((math.sin(math.radians(orientation))*step_size)))

    # Calculating the clearance points
    clearance_x =  current[0] + int(round((math.cos(math.radians(orientation))*clearance)))
    clearance_y = current[1] + int(round((math.sin(math.radians(orientation))*clearance)))

    newPoints = (new_x, new_y)
    clearance_cord = (clearance_x, clearance_y)
    if not canMove(newPoints) or not canMove(clearance_cord):
        return False , None ,None , step_size
    return True, newPoints, orientation, step_size    

# Function to move left 30 degrees
def move_left_30(node,step_size):
    current = node.getPoints()
    orientation = node.getOrientation()

    new_x = current[0] + int(round((math.cos(math.radians(orientation-30))*step_size )))
    new_y = current[1] + int(round((math.sin(math.radians(orientation-30))*step_size)))

    # Calculating the clearance points
    clearance_x =  current[0] + int(round((math.cos(math.radians(orientation -30 ))*clearance)))
    clearance_y = current[1] + int(round((math.sin(math.radians(orientation -30))*clearance)))

    newPoints = (new_x, new_y)
    clearance_cord = (clearance_x, clearance_y)
    if not canMove(newPoints) or not canMove(clearance_cord):
        return False , None ,None , step_size
    return True, newPoints, orientation-30, step_size

# Function to move left 60 degrees
def move_left_60(node,step_size):
    current = node.getPoints()
    orientation = node.getOrientation()

    new_x = current[0] + int(round((math.cos(math.radians(orientation-60))*step_size )))
    new_y = current[1] + int(round((math.sin(math.radians(orientation-60))*step_size)))
    
    # Calculating the clearance points
    clearance_x =  current[0] + int(round((math.cos(math.radians(orientation -60))*clearance)))
    clearance_y = current[1] + int(round((math.sin(math.radians(orientation -60 ))*clearance)))

    newPoints = (new_x, new_y)
    clearance_cord = (clearance_x, clearance_y)
    if not canMove(newPoints) or not canMove(clearance_cord):
        return False , None ,None , step_size
    return True, newPoints, orientation-60, step_size

# Function to move right 30 degrees
def move_right_30(node,step_size):
    current = node.getPoints()
    orientation = node.getOrientation()

    new_x = current[0] + int(round((math.cos(math.radians(orientation+30))*step_size )))
    new_y = current[1] + int(round((math.sin(math.radians(orientation+30))*step_size)))

    # Calculating the clearance points
    clearance_x =  current[0] + int(round((math.cos(math.radians(orientation +30))*clearance)))
    clearance_y = current[1] + int(round((math.sin(math.radians(orientation +30 ))*clearance)))

    newPoints = (new_x, new_y)
    clearance_cord = (clearance_x, clearance_y)
    if not canMove(newPoints) or not canMove(clearance_cord):
        return False , None ,None , step_size
    return True, newPoints, orientation+30, step_size

# Function to move right 60 degrees
def move_right_60(node,step_size):
    current = node.getPoints()
    orientation = node.getOrientation()

    new_x = current[0] + int(round((math.cos(math.radians(orientation+60))*step_size )))
    new_y = current[1] + int(round((math.sin(math.radians(orientation+60))*step_size)))

    # Calculating the clearance points
    clearance_x =  current[0] + int(round((math.cos(math.radians(orientation +60 ))*clearance)))
    clearance_y = current[1] + int(round((math.sin(math.radians(orientation +60))*clearance)))

    newPoints = (new_x, new_y)
    clearance_cord = (clearance_x, clearance_y)
    if not canMove(newPoints) or not canMove(clearance_cord):
        return False , None ,None , step_size
    return True, newPoints, orientation+60, step_size

# Check if point is in the goal threshold
def in_goal_thershold(point,goal, thershold):
    x , y = point[0] ,point[1]
    circle_center_x, circle_center_y, = goal[0], goal[1]
    radius = thershold
    distance = math.sqrt((x - circle_center_x) ** 2 + (y - circle_center_y) ** 2)
    return distance <= radius

# Calculate the heuristic value = Eulicdean distance
def heuristic(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# A* Algorithm 
def a_star(initial, final, inital_orentation, goal_orentation ):
    # Initialize lists
    open_list = PriorityQueue()
    closed_list = set()
    visited = dict()

    start_node = Node(initial, inital_orentation ,None, 0, heuristic(initial, final))

    open_list.put((start_node.total_cost, start_node))
    visited[initial] = start_node
    closed_list.add(initial)
    
    frame_skip = 10
    frame_counter = 0
    
    while not open_list.empty():
        current_node = open_list.get()[1]
        current_point = current_node.getPoints()

        # Check if current point is within the goal threshold
        if in_goal_thershold(current_point,final, 1.5):
            path = []
            while current_node is not None:
                path.append(current_node.getPoints())
                canvas[current_node.getPoints()[1], current_node.getPoints()[0]] = (255, 255, 0)
                current_node = current_node.getParent()
                
            return path[::-1]

        # Move in all directions
        for move in [moveStraight, move_left_30, move_left_60, move_right_30, move_right_60]:
            can_move, new_point, new_orentation, new_cost = move(current_node,stepSize)
            if can_move:
                new_node = Node(new_point, new_orentation ,current_node, current_node.cost + new_cost, heuristic(new_point, final))
                if new_point not in closed_list:
                    if new_point not in visited:
                        open_list.put((new_node.total_cost, new_node))
                        visited[new_point] = new_node
                        closed_list.add(new_point)
                        
                        # canvas[new_point[1], new_point[0]] = new_color
                        draw_exploration(new_node)
                        if frame_counter % frame_skip == 0:
                            video_writer.write(canvas)
                        frame_counter += 1
                    else:   
                        if visited[new_point].total_cost > new_node.total_cost:
                            visited[new_point] = new_node
                            open_list.put((new_node.total_cost, new_node))

# Draw the path as arrows 
def draw_arrow(path):
    
    for i in range(len(path) - 1):
        start_node = path[i]
        end_node = path[i + 1]
        cv2.arrowedLine(canvas, start_node, end_node, (0, 255, 0), 1 , tipLength= 0.5)
        # if frame_counter % frame_skip == 0:
        video_writer.write(canvas)
        
        
        
# Draw the exploration of the nodes
def draw_exploration(explored_node):
    if explored_node.getParent() is not None:
        cv2.line(canvas, explored_node.getPoints(), explored_node.getParent().getPoints() , new_color, thickness=1, lineType=cv2.LINE_8, shift=0)

        # cv2.arrowedLine(canvas, explored_node.getPoints(), explored_node.getParent().getPoints() , new_color, 1 , tipLength= 0.5)

# Main function
if __name__ == "__main__":
    intial, final, inital_orentation, goal_orentation = start_end_goals()
    initalize_varaibles()
    start_time = time.time()
    shortest_path = a_star(intial, final,inital_orentation, goal_orentation)
    end_time = time.time()
    print("Time taken to find the path: ", end_time - start_time)
    # Visualize start and goal nodes
    cv2.circle(canvas, (intial[0], intial[1]), 5, (0, 255, 0), -1)  # Start node in white
    cv2.circle(canvas, (final[0], final[1]), 5, (0, 255, 0), -1)  # Goal node in green

    draw_arrow(shortest_path)
    video_writer.release()
    show_image()
