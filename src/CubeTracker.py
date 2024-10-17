import cv2
import numpy as np
from scipy.spatial.distance import pdist, squareform
from enum import Enum

def nothing(x):
    pass

# Load the image
image = cv2.imread('fsd_Color.png')


class CubeState(Enum):
    TARGETED = "targeted"
    PICKED_UP = "pickedUp"
    ON_BOARD = "onBoard"
    ON_ANOTHER = "onAnother"
    DROPPED_DOWN = "droppedDown"
    NEE = "onAnother"
    NULL = "null"
    
    @classmethod
    def from_string(cls, state_str):
        """
        Search for an enum member by its string value.
        :param state_str: The string value to search for.
        :return: Corresponding CubeState member or None if not found.
        """
        for state in cls:
            if state.value == state_str:
                return state
        return None
    
    
    
class Cube:
    def __init__(self, position, color_name, contour, vertices, is_square, angle):
        self.position = position  # (x, y) position of the cube
        self.color = color_name   # Identified color of the cube
        self.contour = contour    # Contour for visual representation
        self.vertices = vertices  # Vertices of the detected contour
        self.is_square = is_square  # Boolean flag indicating if the shape is square-like
        self.id = id(self)   
        self.angle = angle    
        self.state = CubeState.ON_BOARD
        
    def set_picked (self, state:CubeState):
        self.state = state
        
    def get_state(self) -> CubeState:
        return self.state
        
    def update(self, cube=None):
        match self.state:
            case CubeState.TARGETED:
                
                pass
            
            case CubeState.PICKED_UP:
                self.position = None
                self.contour = None
                self.vertices = None
                self.is_square= None
                self.angle = None
                
            case CubeState.DROPPED_DOWN:
                self.position = cube.position
                self.contour = cube.contour
                self.vertices = cube.vertices
                self.is_square = cube.is_square
                self.angle = cube.angle
            
        return
        
    def is_point_inside_contour(self, point):
        """Check if a given point is inside the cube's contour."""
        return cv2.pointPolygonTest(self.contour, point, False) >= 0

    def is_point_inside_vertices(self, point):
        """Check if a given point is inside the polygon formed by the vertices."""
        if self.vertices is None:
            return False
        polygon = np.array([vertex[0] for vertex in self.vertices])
        return cv2.pointPolygonTest(polygon, point, False) >= 0
    
    
    
class CubeTracker:
    def __init__(self):
        # Define HSV ranges for known cube colors
        self.color_ranges_lenient = {
            'red': 
                # [
                # ((31, 0, 0), (72, 43, 135)),  # Lower and upper range for red
                ((11, 0, 0), (31, 82, 255))   ,
            # ],
            
            'green': ((30, 43, 0), (101, 255, 255)),  # Lower and upper range for green
            
            'blue': ((75, 0, 0), (114, 255, 255)),  # Lower and upper range for blue
            
            'yellow': ((19, 30, 0), (49, 255, 255)),
            
            'orange': ((10, 15, 61), (78, 239, 125)), 
            
            'purple': ((94, 0, 0), (176, 111, 255)),
        }
        
        self.detected_cubes = [] 
        self.selected_cube = None
        self.limbo = []
        # strict ranges
        self.color_ranges = {
            'red': 
                # [
                # ((31, 6, 0), (78, 43, 255)),  # Lower and upper range for red
                ((12, 11, 41), (30, 62, 139))   ,
            # ],
            
            'green': ((64, 57, 55), (78, 119, 141)),  # Lower and upper range for green
            
            'blue': ((88, 36, 34), (106, 170, 137)),  # Lower and upper range for blue
            
            'yellow': ((21, 63, 65), (35, 194, 200)),
            
            'orange': ((10, 16, 0), (74, 228, 125)),
            
            'purple': ((97, 1, 34), (152, 71, 126)),
            
            'orange1': ((0, 37, 67), (42, 90, 138)),
            
            'orange2': ((7, 51, 61), (68, 91, 127)),
        }

        # self.color_ranges = {
        #     # 'red': ([0, 50, 50], [10, 255, 255]),  # Two ranges for red due to HSV wrap-around
        #     # 'green': ([35, 50, 50], [85, 255, 255]),
        #     'green': ((35, 66, 89), (85, 255, 255)),
        #     'blue': ([90, 50, 50], [130, 255, 255]),
        #     # 'yellow': ((21, 63, 65), (35, 194, 200)),
        #     # 'orange': ([0, 0, 126], [17, 255, 255]),
        #     'purple': ((120, 0, 40), (160, 255, 255)),
        #     'red1': ([160, 50, 50], [180, 255, 255]),
        #     }
        
        self.color_ranges = {
           'red1': ([0, 50, 50], [10, 255, 255]),  # Lower red range
            'red2': ([170, 50, 50], [180, 255, 255]),  # Upper red range for wrap-around

            'green': ([35, 66, 89], [85, 255, 255]),

            'blue': ([73, 40, 90], [131, 255, 255]),

            'yellow': ([21, 63, 65], [35, 194, 200]),

            'orange': ([10, 183, 0], [17, 255, 255]),

            'purple': ([120, 0, 40], [160, 255, 255])

            }
        
        self.reject = {
            "black": ([0, 0, 0], [180, 255, 38])
        }
    # def preprocess_image(self, image):
    #     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #     lower_gray = np.array([0, 0, 55])
    #     upper_gray = np.array([180, 255, 255])

    #     # Create a mask for the gray color range
    #     gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
    #     non_gray_mask = cv2.bitwise_not(gray_mask)
    #     image_no_gray = cv2.bitwise_not(image, image, mask=non_gray_mask)

    #     # Visualize the image after rejecting gray areas
    #     cv2.imshow("Image with Gray Rejected", image_no_gray)
    #     return image_no_gray
        
    def sanity_check(self, image):
        """
        Perform a sanity check to update cubes that have been moved, 
        match and update as many as possible, and handle newly detected cubes.
        """
        detected_cubes = self.detect_cubes(image)
        image_shape = image.shape[:2]  # Get the shape of the image for overlap calculations
        unmatched_cubes = []
        matched_cubes = set()  # Keep track of cubes that have been matched or updated

        # Step 1: Try to match detected cubes with existing cubes by collision detection
        for detected_cube in detected_cubes:
            # Try to update the cube by checking for collisions with existing cubes
            if not self.collide_check_update(detected_cube, image_shape):
                matched_cubes.add(detected_cube)  # If a collision was found, it's considered matched
                continue  # Move on to the next detected cube
            unmatched_cubes.append(detected_cube)  # Collect unmatched cubes for further processing

        # Step 2: Try to match unmatched cubes by color and smallest distance
        for new_cube in unmatched_cubes:
            closest_cube = None
            min_distance = float('inf')

            # Compare with all existing cubes based on color and find the closest one
            for known_cube in self.detected_cubes:
                if new_cube.color == known_cube.color:
                    # Calculate the distance between the new cube and the known cube
                    distance = np.linalg.norm(np.array(new_cube.position) - np.array(known_cube.position))

                    # Track the cube with the smallest distance
                    if distance < min_distance:
                        closest_cube = known_cube
                        min_distance = distance
            
            # Step 3: If a match is found (smallest distance), update the known cube
            if closest_cube is not None:
                self.adjust_found(existing_cube=closest_cube, new_cube=new_cube)
                matched_cubes.add(closest_cube)  # Mark this known cube as matched
            else:
                # If no match is found, treat it as a completely new cube and add it
                self.add_cube(new_cube, image_shape)

        # Step 4: Remove unmatched cubes from the detected_cubes list
        self.detected_cubes = [cube for cube in self.detected_cubes if cube in matched_cubes]

        return


    
    def update_cube(self, cube, image, coord):
        match self.state:
           
            case CubeState.PICKED_UP:
                self.selected_cube = cube
                self.detected_cubes.remove(cube)
                
                cube.update()
                pass
            case CubeState.DROPPED_DOWN:
                small_dict = {cube.color: self.color_ranges[cube.color]}
                familiar = self.detect_cubes(image, color_ranges=small_dict)
                found = False
                tmp = None
                for fc in familiar:
                    if fc.is_point_inside_contour(coord) or fc.is_point_inside_vertices(coord):
                        found = True
                        tmp = fc
                        
                        break
                if not found:
                    self.sanity_check()
                    self.limbo.append(cube)
                    cube.state=CubeState.NULL
                else:   
                    self.detected_cubes.append(cube)
                    
                if tmp is not None:
                    cube.update(tmp)
                else:
                    cube.update()
                pass
        
        return
    
    def get_selected_cube(self):
        return self.selected_cube
        
    def find_object_direction(self, contour):
        # Ensure the contour is a 2D array of points
        contour = np.squeeze(contour)
        
        # Check if the contour has valid dimensions (should be (n, 2) where n is the number of points)
        if len(contour.shape) != 2 or contour.shape[1] != 2:
            raise ValueError("Contour should be a 2D array with shape (n, 2)")
        
        # Compute the mean (centroid) of the points
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(contour.astype(np.float32), np.mean(contour, axis=0).reshape(1, -1).astype(np.float32))
        
        # The direction is given by the first eigenvector
        direction_vector = eigenvectors[0]
        
        # Optionally, return the angle of this vector relative to some reference (e.g., x-axis)
        angle = np.arctan2(direction_vector[1], direction_vector[0]) * 180 / np.pi
        
        return mean, direction_vector, angle

    def draw_direction_line(self, image, center, direction_vector, length=40):
        # Normalize the direction vector and calculate the endpoint of the line
        direction_vector = direction_vector / np.linalg.norm(direction_vector)  # Normalize to unit vector
        
        start_point = tuple(center.astype(int))
        end_point = (int(center[0] + direction_vector[0] * length), int(center[1] + direction_vector[1] * length))

        # Draw a gradient line from red to green
        for i in range(length):
            # Interpolate color from red to green
            r = int(255 * (1 - i / length))  # Red decreases
            g = int(255 * (i / length))      # Green increases
            color = (0, g, r)

            # Interpolate the point along the line
            point = (int(center[0] + direction_vector[0] * i), int(center[1] + direction_vector[1] * i))

            # Draw the point (with a 2-pixel width for better visibility)
            cv2.circle(image, point, 2, color, -1)

        return image

    def get_cubes(self):
        """Return the list of detected cube objects."""
        return self.detected_cubes

    def get_cube_by_id(self, cube_id):
        """Return a specific cube from the list based on its ID."""
        for cube in self.detected_cubes:
            if cube.id == cube_id:
                return cube
        return None
    
    def __add_cube(self, center, color_name, contour, vertices, is_square):
        """Add a new cube to the list of detected cubes."""
        new_cube = Cube(center, color_name, contour, vertices, is_square)
        self.detected_cubes.append(new_cube)
    
    def calculate_overlap(self, contour1, contour2, image_shape):
        """Calculate the overlap area percentage between two contours."""
        # Create masks for the two contours using the size of the image
        mask1 = np.zeros(image_shape, dtype=np.uint8)
        mask2 = np.zeros(image_shape, dtype=np.uint8)
        cv2.drawContours(mask1, [contour1], -1, 255, -1)  # Fill contour1
        cv2.drawContours(mask2, [contour2], -1, 255, -1)  # Fill contour2

        # Calculate the intersection and union areas
        intersection = cv2.bitwise_and(mask1, mask2)
        intersection_area = np.sum(intersection == 255)

        area1 = cv2.contourArea(contour1)
        area2 = cv2.contourArea(contour2)

        # Calculate the union area
        union_area = area1 + area2 - intersection_area

        # Calculate the percentage of overlap
        overlap_percentage = (intersection_area / union_area) * 100 if union_area > 0 else 0

        return overlap_percentage
    
    def check_if_square(self, contour):
        """Check if a given contour can be approximated as a square."""
        epsilon = 0.05 * cv2.arcLength(contour, True)  # Tolerance level for approximation
        approx = cv2.approxPolyDP(contour, epsilon, True)
        new_center = -1
        # Check if the contour has four vertices and looks square-like
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.8 <= aspect_ratio <= 1.2:  # Adjust range to match square-like aspect ratio
                return True, approx, None
            points = [tuple(pt[0]) for pt in approx]
            # Sort points based on the y-coordinate (vertical position)
            points = [tuple(pt[0]) for pt in approx]

    # Calculate pairwise distances between all points
            distances = squareform(pdist(points))

            # Identify the three points that form the most consistent group
            # Sum of distances to other points
            distance_sums = np.sum(distances, axis=1)
            
            # Find the index of the point with the highest sum of distances (outlier)
            outlier_index = np.argmax(distance_sums)

            # Define the outlier point and the base points
            outlier_point = points[outlier_index]
            base_points = [pt for i, pt in enumerate(points) if i != outlier_index]
            # find outlier
            apex = np.array(points[(outlier_index+2)%4])
            mid = (np.array(points[(outlier_index+1)%4]) + np.array(points[(outlier_index+3)%4])) / 2

            new_center = mid.astype(int)
            return False, approx, new_center
        return False, approx, None

    
    def adjust_found(self, existing_cube:Cube, new_cube:Cube):
        # Update existing cube properties
        existing_cube.position = new_cube.position
        # existing_cube.color = new_cube.color
        existing_cube.contour = new_cube.contour
        existing_cube.vertices = new_cube.vertices
        existing_cube.is_square = new_cube.is_square
        existing_cube.angle = new_cube.angle
        # updated_cubes.append(existing_cube)
                    
    def collide_check_update(self, new_cube:Cube, image_shape, adjust=True):
        found_collision = False
        for existing_cube in self.detected_cubes:
            try:
                # Check area overlap percentage between the new and existing cubes
                overlap_percentage = self.calculate_overlap(new_cube.contour, existing_cube.contour, image_shape)
                # if existing_cube.color == "yellow" and new_cube.color == "orange" and overlap_percentage > 0:
                #     print (existing_cube.contour, new_cube.contour)
                #     print(overlap_percentage)
                if overlap_percentage > 80:  # Threshold for considering it the same cube
                    found_collision = True
                    if adjust:
                        self.adjust_found(existing_cube=existing_cube, new_cube=new_cube)
                    return False
            except Exception as e:
                print("weird cube state", e)
                
        if not found_collision:
            return True
        return False
    
    def add_cube(self, new_cube:Cube, image_shape, check = True):
        """Update the list of cubes, handle collisions based on area overlap."""
        if check and not self.collide_check_update(new_cube, image_shape):
            # If no collision is found, add the new cube
            return False
        self.detected_cubes.append(new_cube)
        return True
        # Replace the current detected cubes with the updated list

    def detect_cubes(self, image, new_cubes = [], color_ranges = None):
        if color_ranges is None:
            color_ranges = self.color_ranges
        # Convert the image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # hsv = cv2.GaussianBlur(hsv, (3, 3), 0)
        # Create a window


        found_cubes = []
        for color_name, (lower, upper) in color_ranges.items():
            # if color_name != "red":
            #     continue
            mask = cv2.inRange(hsv, tuple(lower), tuple(upper))
            # mask = cv2.erode(mask, None, iterations=1)
            # mask = cv2.dilate(mask, None, iterations=1)
        # Find contours in the masks
            # reject= cv2.inRange(hsv, tuple(self.reject["black"][0]), tuple (self.reject["black"][1]))
            # # Invert the mask to reject values in the range [0, 38]
            # inverted_mask = cv2.bitwise_not(reject)

            # # Combine this new mask with your existing mask using a bitwise AND
            # # Make sure that both masks are of the same dimensions
            # final_mask = cv2.bitwise_and(mask, inverted_mask)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw contours on the original image
            # output_image = image.copy()
            for contour in contours:
                if cv2.contourArea(contour) < 300 or cv2.contourArea(contour) > 1000:  # Filter small contours
                    continue
                # if color_name == "ARM":
                #     cv2.drawContours(image, contour, -1, (0, 255, 0), 2)
                #     self.find_object_direction(contour)
                    
                    continue
                rect = cv2.minAreaRect(contour)  # Get the minimum area bounding rectangle
                angle = rect[2]
                
                # if angle < -45:
                # angle_vs_y_axis = 90 + angle
                # else:
                #     angle_vs_y_axis = -angle

                # Display the angle information
                # Get the center of the rectangle
                center = (int(rect[0][0]), int(rect[0][1]))
                
                box = cv2.boxPoints(rect)
                box = np.int16(box)
                center = np.int16(rect[0])
                if box is not None and len(box) > 0:
                    for i in range(4):
                        start_point = tuple(box[i])
                        end_point = tuple(box[(i + 1) % 4])
                        # if i == 0:
                        # cv2.line(image, start_point, end_point, (0, 255, 0), 2)  # Draw each line in green
                        # cv2.putText(image, color_name, (center[0], center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        is_square, vertices, newcent = self.check_if_square(contour)        
                        if not is_square and newcent is not None:
                            center = newcent                    
                new_cube = Cube(center, color_name, contour, vertices, is_square, angle)
                found_cubes.append(new_cube)
        return found_cubes
        # Update detected cubes with collision handling based on area overlap
                
    def update_cubes(self, found_cubes, shape):
        for cube in found_cubes:
            self.add_cube(cube, shape)
        
    def detect_and_update(self, image):
        self.update_cubes(self.detect_cubes(image), image.shape[:2])
        
    def display_selected_cube(self, cube:Cube, image):
        try:
            if cube.is_square:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            # cv2.drawContours(image, [cube.contour], -1, color, 2)
            # print(cube.position)
            cv2.putText(image, f"{cube.color} {'Square' if cube.is_square else ''}", 
                        (cube.position[0], cube.position[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # cv2.circle(image, cube.position, 5, (0, 255, 0), -1)
            # Optionally, draw vertices if available
            if cube.vertices is not None:
                for vertex in cube.vertices:
                    cv2.circle(image, tuple(vertex[0]), 5, (255, 0, 0), -1)
                    pass
            # self.display_angles(cube,image)
        except Exception as e:
            print("Visualize this cube failed", cube.id, "Error:  ", e)
                
        
    def display_cube(self, image):
        """Display the tracked cubes on the image."""
        for cube in self.detected_cubes:
            self.display_selected_cubes(cube, image)
                
        resized_up = cv2.resize(image, (1280, 768), interpolation= cv2.INTER_LINEAR)
        cv2.imshow('Cube Tracking', resized_up)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def cubeAngle_to_world(self, angle):
        angle+=180
        if angle < 225:
            angle += 90
            
        return angle
    
    def display_angles(self, cube:Cube, image):
        center = cube.position  
        angle = self.cubeAngle_to_world(cube.angle)
        
        # cv2.putText(image, str(angle-270), (center[0], center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Length of the line representing the angle
        line_length = 40

        # Compute the end point of the angle line
        end_x = int(center[0] + line_length * np.cos(np.radians(angle)))
        end_y = int(center[1] + line_length * np.sin(np.radians(angle)))

        # cv2.line(image, center, (center[0], center[1] - 100), (0, 255, 0), 2)

        # Draw the angle line (representing the rotation) in red
        cv2.line(image, center, (end_x, end_y), (0, 0, 255), 2)
        cv2.circle(image, tuple(center), 5, (255, 0, 0), -1)