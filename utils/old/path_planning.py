import numpy as np
import cv2
import math
from utils.image_processing import image_cleanup, find_largest_clusters
from utils.segmentation import ImageSegmentation
import yaml
import matplotlib.pyplot as plt
from skimage.draw import line, line_aa, line_nd

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0

class RRTStar:
    """
    Rapidly-exploring Random Tree (RRT*) algorithm for path planning.

    Args:
        img (numpy.ndarray): The input image.
        start (tuple): The starting point coordinates.
        end (tuple): The ending point coordinates.
        save_path (str): The path to save the animation.
        config (dict): The configuration dictionary.

    Attributes:
        img (numpy.ndarray): The input image.
        binary_img (numpy.ndarray): The binary image generated from the input image.
        start (Node): The starting node.
        end (Node): The ending node.
        max_iter (int): The maximum number of iterations.
        goal_sample_rate (int): The goal sample rate.
        max_distance (float): The maximum distance between nodes.
        nodes (list): The list of nodes.
        path (Path): The path from start to end.
        save_path (str): The path to save the animation.

    """

    def __init__(self, img, start, end, save_path, config):
        self.img = img
        self.binary_img = self.generate_bitmap(img, sam_config=config["sam_config"])
        self.start = Node(start[0], start[1])
        self.end = Node(end[0], end[1])
        self.max_iter = config["Path Planning"]["max_iter"]
        self.goal_sample_rate = config["Path Planning"]["goal_sample_rate"]
        self.max_distance = config["Path Planning"]["max_distance"]
        self.nodes = [self.start]
        self.path = None
        self.save_path = save_path
        self.shortest_path = float("inf")


    def generate_bitmap(self, img, sam_config):
        """
        Generates a binary image from the input image.

        Args:
            img (numpy.ndarray): The input image.
            sam_config (dict): The configuration dictionary for the image segmentation.

        Returns:
            numpy.ndarray: The binary image.

        """
        sam = ImageSegmentation(sam_config)
        input_point = np.array([[140, 850], [800, 200], [800, 800], [0,0], [500, 500]])
        input_label = np.array([1, 1, 1, 0, 0])
        sam.add_input_points(input_point, input_label)
        masks, _, _ = sam.predict(img)

        return masks[0]

    def plan(self, show_animation=False):
        """
        Plans the path from start to end using the RRT* algorithm.

        Args:
            show_animation (bool): Whether to show the animation or not.

        Returns:
            Path: The path from start to end.

        """
        for i in range(self.max_iter):
            if np.random.randint(0, 100) > self.goal_sample_rate:
                rnd = Node(np.random.randint(0, self.img.shape[1]), np.random.randint(0, self.img.shape[0]))
            else:
                rnd = self.end

            nearest_node = self.get_nearest_node(rnd)
            new_node = self.steer(nearest_node, rnd)

            if self.is_collision_free(nearest_node, new_node):
                near_nodes = self.get_near_nodes(new_node)
                self.nodes.append(new_node)
                self.choose_parent(new_node, near_nodes)
                self.rewire(new_node, near_nodes)

                if self.is_goal_reachable(new_node):
                    path, path_len = self.get_path(new_node)
                    if path_len < self.shortest_path:
                        self.path = path
                        self.shortest_path = path_len
                    if show_animation:
                        self.animation(self.path)
                        plt.show(block=False)
                        plt.pause(0.1)

                elif self.path is None and i % 10 == 0 and show_animation:
                    _temp_path, path_len = self.get_path(new_node)
                    self.animation(_temp_path)
                    plt.show(block=False)
                    plt.pause(0.1)

        if self.path is not None:
            if show_animation:
                self.animation(self.path)
                plt.savefig(self.save_path)
            return self.path

        return None

    def get_nearest_node(self, node):
        """
        Returns the nearest node to the given node.

        Args:
            node (Node): The node to find the nearest node to.

        Returns:
            Node: The nearest node.

        """
        distances = [math.sqrt((n.x - node.x) ** 2 + (n.y - node.y) ** 2) for n in self.nodes]
        nearest_node = self.nodes[np.argmin(distances)]
        return nearest_node

    def steer(self, from_node, to_node):
        """
        Steers the node from the from_node to the to_node.

        Args:
            from_node (Node): The starting node.
            to_node (Node): The ending node.

        Returns:
            Node: The new node.

        """
        distance = math.sqrt((to_node.x - from_node.x) ** 2 + (to_node.y - from_node.y) ** 2)
        if distance > self.max_distance:
            ratio = self.max_distance / distance
            x = int(from_node.x + (to_node.x - from_node.x) * ratio)
            y = int(from_node.y + (to_node.y - from_node.y) * ratio)
            new_node = Node(x, y)
        else:
            new_node = Node(to_node.x, to_node.y)
        new_node.parent = from_node
        new_node.cost = from_node.cost + math.sqrt((new_node.x - from_node.x) ** 2 + (new_node.y - from_node.y) ** 2)
        return new_node

    def is_collision_free(self, from_node, to_node):
        """
        Checks if the path from the from_node to the to_node is collision-free.

        Args:
            from_node (Node): The starting node.
            to_node (Node): The ending node.

        Returns:
            bool: Whether the path is collision-free or not.

        """
        rr, cc, _ = line_aa(from_node.x, from_node.y, to_node.x, to_node.y)
        return np.any(self.binary_img[cc, rr])

    def get_near_nodes(self, node):
        """
        Returns the nodes near the given node.

        Args:
            node (Node): The node to find the near nodes to.

        Returns:
            list: The list of near nodes.

        """
        distances = [math.sqrt((n.x - node.x) ** 2 + (n.y - node.y) ** 2) for n in self.nodes]
        near_nodes = [self.nodes[i] for i in range(len(distances)) if distances[i] < self.max_distance*2]
        if not near_nodes:
            raise(Exception("No near nodes found"))
        return near_nodes

    def choose_parent(self, node, near_nodes):
        """
        Chooses the parent node for the given node.

        Args:
            node (Node): The node to choose the parent for.
            near_nodes (list): The list of near nodes.

        """
        costs = [n.cost + math.sqrt((n.x - node.x) ** 2 + (n.y - node.y) ** 2) for n in near_nodes]
        min_cost_node = near_nodes[np.argmin(costs)]
        node.parent = min_cost_node
        node.cost = min_cost_node.cost + math.sqrt((node.x - min_cost_node.x) ** 2 + (node.y - min_cost_node.y) ** 2)

    def rewire(self, node, near_nodes):
        """
        Rewires the nodes near the given node.

        Args:
            node (Node): The node to rewire the near nodes to.
            near_nodes (list): The list of near nodes.

        """
        for near_node in near_nodes:
            if near_node == node.parent:
                continue
            new_cost = node.cost + math.sqrt((node.x - near_node.x) ** 2 + (node.y - near_node.y) ** 2)
            if new_cost < near_node.cost and self.is_collision_free(node, near_node):
                near_node.parent = node
                near_node.cost = new_cost

    def is_goal_reachable(self, node):
        """
        Checks if the goal is reachable from the given node.

        Args:
            node (Node): The node to check the goal reachability from.

        Returns:
            bool: Whether the goal is reachable or not.

        """
        distance_to_goal = math.sqrt((node.x - self.end.x) ** 2 + (node.y - self.end.y) ** 2)
        if distance_to_goal > self.max_distance:
            return False
        return self.is_collision_free(node, self.end)


    def get_path(self, node):
        """
        Returns the path from start to end.

        Args:
            node (Node): The ending node.

        Returns:
            Path: The path from start to end.

        """
        path = []
        while node is not None:
            path.append((node.x, node.y))
            node = node.parent
        path_len = Path(path).path_length
        return path[::-1], path_len

    def animation(self, path):
        """
        Animates the path.

        Args:
            path (Path): The path to animate.

        """
        plt.cla()
        plt.plot(self.start.x, self.start.y, "bs", linewidth=3)
        plt.plot(self.end.x, self.end.y, "rs", linewidth=3)
        imgplot = plt.imshow(self.img)
        x_coords = [node[0] for node in path]
        y_coords = [node[1] for node in path]
        plt.plot(x_coords, y_coords, "g-", linewidth=0.5)
        plt.savefig(self.save_path)
        


import math

class Path():
    """
    A class representing a path in a 2D space.

    Attributes:
    -----------
    path : list
        A list of tuples representing the nodes in the path.
    min_distance : float
        The minimum distance between two nodes in the path.
    traversed_nodes : set
        A set of nodes that have already been traversed.
    path_length : float
        The total length of the path.
    path_cost : float
        The cost of traversing the path.
    path_time : float
        The time required to traverse the path.
    """

    def __init__(self, path, min_distance=10) -> None:
        """
        Initializes a Path object.

        Parameters:
        -----------
        path : list
            A list of tuples representing the nodes in the path.
        min_distance : float, optional
            The minimum distance between two nodes in the path. Default is 10.
        """
        self.path = path
        self.traversed_nodes = set()
        self.path_length = self._calc_path_lenght()
        self.min_distance = min_distance
        self.path_cost = NotImplemented
        self.path_time = NotImplemented

    def get_closest_node(self, position):
        """
        Returns the closest node in the path to a given position.

        Parameters:
        -----------
        position : tuple
            A tuple representing the position in the 2D space.

        Returns:
        --------
        closest_node : tuple
            A tuple representing the closest node in the path to the given position.
        """
        closest_node = None
        min_distance = float('inf')
        for node in self.path:
            distance = math.sqrt((node[0] - position[0]) ** 2 + (node[1] - position[1]) ** 2)
            if distance < min_distance:
                closest_node = node
                min_distance = distance
        return closest_node
    
    def get_closest_not_traversed_node(self, position):
        """
        Returns the closest node in the path that has not been traversed yet.

        Parameters:
        -----------
        position : tuple
            A tuple representing the position in the 2D space.

        Returns:
        --------
        closest_node : tuple
            A tuple representing the closest node in the path that has not been traversed yet.
        """
        closest_node = None
        min_distance = float('inf')
        for node in self.path:
            distance = math.sqrt((node[0] - position[0]) ** 2 + (node[1] - position[1]) ** 2)
            if distance < min_distance and node not in self.traversed_nodes:
                closest_node = node
                min_distance = distance
        return closest_node
    
    def get_next_node(self, position):
        """
        Returns the next node in the path that has not been traversed yet.

        Parameters:
        -----------
        position : tuple
            A tuple representing the position in the 2D space.

        Returns:
        --------
        closest_node : tuple
            A tuple representing the next node in the path that has not been traversed yet.
        """
        closest_node = self.get_closest_not_traversed_node(position)
        return closest_node
    
    def traverse_node(self, node, position):
        """
        Traverses a node in the path if it is close enough to the current position.

        Parameters:
        -----------
        node : tuple
            A tuple representing the node to be traversed.
        position : tuple
            A tuple representing the current position in the 2D space.

        Returns:
        --------
        bool
            True if the node was traversed, False otherwise.
        """
        distance = math.sqrt((node[0] - position[0]) ** 2 + (node[1] - position[1]) ** 2)
        if distance < self.min_distance:
            self.traversed_nodes.add(node)
            return True
        else:
            return False
    
    def _calc_path_lenght(self):
        """
        Calculates the total length of the path.

        Returns:
        --------
        path_length : float
            The total length of the path.
        """
        path_length = 0
        for i in range(len(self.path)-1):
            path_length += math.sqrt((self.path[i][0] - self.path[i+1][0]) ** 2 + (self.path[i][1] - self.path[i+1][1]) ** 2)
        return path_length      

class RRTStarSafe:
    """
    Rapidly-exploring Random Tree (RRT*) algorithm for path planning.

    Args:
        img (numpy.ndarray): The input image.
        start (tuple): The starting point coordinates.
        end (tuple): The ending point coordinates.
        save_path (str): The path to save the animation.
        config (dict): The configuration dictionary.

    Attributes:
        img (numpy.ndarray): The input image.
        binary_img (numpy.ndarray): The binary image generated from the input image.
        start (Node): The starting node.
        end (Node): The ending node.
        max_iter (int): The maximum number of iterations.
        goal_sample_rate (int): The goal sample rate.
        max_distance (float): The maximum distance between nodes.
        nodes (list): The list of nodes.
        path (Path): The path from start to end.
        save_path (str): The path to save the animation.

    """
    # TODO: Assume the input image is binary numpy array 2D
    def __init__(self, img, start, end, save_path, config):
        #self.img = img
        #self.binary_img = self.generate_bitmap(img, sam_config=config["sam_config"])
        self.binary_img = img 
        self.start = Node(start[0], start[1])
        self.end = Node(end[0], end[1])
        self.max_iter = config["max_iter"]
        self.goal_sample_rate = config["goal_sample_rate"]
        self.max_distance = config["max_distance"]
        self.nodes = [self.start]
        self.path = None
        self.save_path = save_path
        self.shortest_path = float("inf")
        self.safety_threshold = config["safety_threshold"]
        self.dilated_img = self.dilate_bitmap(img, 20)

    def dilate_bitmap(self, bitmap, safety_threshold):
        """
        Dilates the obstacles in the binary image by the safety threshold.

        Args:
            bitmap (numpy.ndarray): The binary image.
            safety_threshold (int): The safety threshold.

        Returns:
            numpy.ndarray: The dilated binary image.

        """
        bitmap_copy = bitmap.copy()
        for i in range(bitmap.shape[0]):
            for j in range(bitmap.shape[1]):
                if bitmap[i][j] == 0 and np.any(bitmap[max(0, i - safety_threshold):min(bitmap.shape[0], i + safety_threshold + 1), max(0, j - safety_threshold):min(bitmap.shape[1], j + safety_threshold + 1)]) == 1:
                    for k in range(max(0, i - safety_threshold), min(bitmap.shape[0], i + safety_threshold + 1)):
                        for l in range(max(0, j - safety_threshold), min(bitmap.shape[1], j + safety_threshold + 1)):
                            bitmap_copy[k][l] = 0
        #cv2.imshow("Dilated Image", bitmap_copy*255)
        #cv2.imshow("Original Image", bitmap*255)
        #cv2.waitKey(0)
        return bitmap_copy
                            
        
        image_copy = bitmap.copy()
        image_copy[image_copy == 0] = 1
        image_copy[bitmap==1] = 0
        struct = np.ones((safety_threshold*2+1, safety_threshold*2+1))
        dilated_img = binary_dilation(image_copy, structure=struct)
        return dilated_img


    def plan(self, show_animation=False):
        for i in range(self.max_iter):
            if np.random.randint(0, 100) > self.goal_sample_rate:
                rnd = Node(np.random.randint(0, self.binary_img.shape[1]), np.random.randint(0, self.binary_img.shape[0]))
            else:
                rnd = self.end

            nearest_node = self.get_nearest_node(rnd)

            try:
                new_node = self.steer(nearest_node, rnd)
            except Exception as e:
                #print(f"Steering failed: {e}")
                continue  # Skip the rest of this iteration and try a new random point

            if new_node and self.is_collision_free(nearest_node, new_node):
                near_nodes = self.get_near_nodes(new_node)
                self.nodes.append(new_node)
                self.choose_parent(new_node, near_nodes)
                self.rewire(new_node, near_nodes)

                if self.is_goal_reachable(new_node):
                    path, path_len = self.get_path(new_node)
                    if path_len < self.shortest_path:
                        self.path = path
                        self.shortest_path = path_len
                    if show_animation:
                        self.animation(self.path)
                        plt.show(block=False)
                        plt.pause(0.1)

                elif self.path is None and i % 10 == 0 and show_animation:
                    _temp_path, path_len = self.get_path(new_node)
                    self.animation(_temp_path)
                    plt.show(block=False)
                    plt.pause(0.1)

        if self.path is not None:
            if show_animation:
                self.animation(self.path)
                plt.savefig(self.save_path)
            return self.path

        return None


    def get_nearest_node(self, node):
        """
        Returns the nearest node to the given node.

        Args:
            node (Node): The node to find the nearest node to.

        Returns:
            Node: The nearest node.

        """
        distances = [math.sqrt((n.x - node.x) ** 2 + (n.y - node.y) ** 2) for n in self.nodes]
        nearest_node = self.nodes[np.argmin(distances)]
        return nearest_node

    # def steer(self, from_node, to_node):
    #     """
    #     Steers the node from the from_node to the to_node.

    #     Args:
    #         from_node (Node): The starting node.
    #         to_node (Node): The ending node.

    #     Returns:
    #         Node: The new node.

    #     """
    #     distance = math.sqrt((to_node.x - from_node.x) ** 2 + (to_node.y - from_node.y) ** 2)
    #     if distance > self.max_distance:
    #         ratio = self.max_distance / distance
    #         x = int(from_node.x + (to_node.x - from_node.x) * ratio)
    #         y = int(from_node.y + (to_node.y - from_node.y) * ratio)
    #         new_node = Node(x, y)
    #     else:
    #         new_node = Node(to_node.x, to_node.y)
    #     new_node.parent = from_node
    #     new_node.cost = from_node.cost + math.sqrt((new_node.x - from_node.x) ** 2 + (new_node.y - from_node.y) ** 2)
    #     return new_node
    def steer(self, from_node, to_node):
        """
        Steers the node from from_node to to_node, considering the maximum allowed distance and obstacle safety.

        Args:
            from_node (Node): The starting node.
            to_node (Node): The target node.

        Returns:
            Node: The new node, considering the safety threshold around obstacles.
        """
        distance = math.sqrt((to_node.x - from_node.x) ** 2 + (to_node.y - from_node.y) ** 2)
        if distance > self.max_distance:
            ratio = self.max_distance / distance
        else:
            ratio = 1.0

        x = from_node.x + (to_node.x - from_node.x) * ratio
        y = from_node.y + (to_node.y - from_node.y) * ratio

        new_node = Node(x, y)
        new_node.parent = from_node
        new_node.cost = from_node.cost + math.sqrt((new_node.x - from_node.x) ** 2 + (new_node.y - from_node.y) ** 2)

        # Assuming is_collision_free has been adjusted to account for the safety threshold.
        if self.is_collision_free(from_node, new_node):
            return new_node
        else:
            raise Exception("Collision detected")

    def is_collision_free(self, from_node, to_node):
        """
        Checks if the path from the from_node to the to_node is collision-free.

        Args:
            from_node (Node): The starting node.
            to_node (Node): The ending node.

        Returns:
            bool: Whether the path is collision-free or not.

        """
        
        # # Check if the line intersects any dilated obstacles.
        # rr, cc = line(from_node.x, from_node.y, to_node.x, to_node.y)
        r0, c0 = int(round(from_node.x)), int(round(from_node.y))
        r1, c1 = int(round(to_node.x)), int(round(to_node.y))
        
        # Check if the line intersects any dilated obstacles.
        rr, cc = line(r0, c0, r1, c1)
        return not np.any(self.dilated_img[rr, cc])
    
    # def is_collision_free(self, bitmap, from_node, to_node):
    #     safety_dist = self.safety_threshold
    #     # Bresenham's algorithm
    #     x0, y0 = int(round(from_node.x)), int(round(from_node.y))
    #     x1, y1 = int(round(to_node.x)), int(round(to_node.y))
    #     dx = abs(x1 - x0)
    #     dy = abs(y1 - y0)
    #     sx = 1 if x0 < x1 else -1
    #     sy = 1 if y0 < y1 else -1
    #     err = dx - dy
    #     line_pixels = []
    #     while True:
    #         line_pixels.append((x0, y0))
    #         if x0 == x1 and y0 == y1:
    #             break
    #         e2 = 2 * err
    #         if e2 > -dy:
    #             err -= dy
    #             x0 += sx
    #         if e2 < dx:
    #             err += dx
    #             y0 += sy

    #     for pixel in line_pixels:
    #         x, y = pixel
    #         for i in range(max(0, x - safety_dist), min(300, x + safety_dist + 1)):
    #             for j in range(max(0, y - safety_dist), min(300, y + safety_dist + 1)):
    #                 if bitmap[i][j] == 0 and np.sqrt((x - i) ** 2 + (y - j) ** 2) <= safety_dist:
    #                     return False
    #     return True

    def get_near_nodes(self, node):
        """
        Returns the nodes near the given node.

        Args:
            node (Node): The node to find the near nodes to.

        Returns:
            list: The list of near nodes.

        """
        distances = [math.sqrt((n.x - node.x) ** 2 + (n.y - node.y) ** 2) for n in self.nodes]
        near_nodes = [self.nodes[i] for i in range(len(distances)) if distances[i] < self.max_distance*2]
        if not near_nodes:
            raise(Exception("No near nodes found"))
        return near_nodes

    def choose_parent(self, node, near_nodes):
        """
        Chooses the parent node for the given node.

        Args:
            node (Node): The node to choose the parent for.
            near_nodes (list): The list of near nodes.

        """
        costs = [n.cost + math.sqrt((n.x - node.x) ** 2 + (n.y - node.y) ** 2) for n in near_nodes]
        min_cost_node = near_nodes[np.argmin(costs)]
        node.parent = min_cost_node
        node.cost = min_cost_node.cost + math.sqrt((node.x - min_cost_node.x) ** 2 + (node.y - min_cost_node.y) ** 2)

    def rewire(self, node, near_nodes):
        """
        Rewires the nodes near the given node.

        Args:
            node (Node): The node to rewire the near nodes to.
            near_nodes (list): The list of near nodes.

        """
        for near_node in near_nodes:
            if near_node == node.parent:
                continue
            new_cost = node.cost + math.sqrt((node.x - near_node.x) ** 2 + (node.y - near_node.y) ** 2)
            if new_cost < near_node.cost and self.is_collision_free(node, near_node):
                near_node.parent = node
                near_node.cost = new_cost

    def is_goal_reachable(self, node):
        """
        Checks if the goal is reachable from the given node.

        Args:
            node (Node): The node to check the goal reachability from.

        Returns:
            bool: Whether the goal is reachable or not.

        """
        distance_to_goal = math.sqrt((node.x - self.end.x) ** 2 + (node.y - self.end.y) ** 2)
        if distance_to_goal > self.max_distance:
            return False
        return self.is_collision_free(node, self.end)


    def get_path(self, node):
        """
        Returns the path from start to end.

        Args:
            node (Node): The ending node.

        Returns:
            Path: The path from start to end.

        """
        path = []
        while node is not None:
            path.append((node.x, node.y))
            node = node.parent
        path_len = Path(path).path_length
        return path[::-1], path_len

    def animation(self, path):
        """
        Animates the path.

        Args:
            path (Path): The path to animate.

        """
        plt.cla()
        plt.plot(self.start.x, self.start.y, "bs", linewidth=3)
        plt.plot(self.end.x, self.end.y, "rs", linewidth=3)
        imgplot = plt.imshow(self.binary_img)
        x_coords = [node[0] for node in path]
        y_coords = [node[1] for node in path]
        plt.plot(x_coords, y_coords, "g-", linewidth=0.5)


class Astar2:
    def __init__(self, bitmap, start, end, save_path, config):
        self.bitmap = bitmap
        self.start = tuple(start)
        self.end = tuple(end)
        self.save_path = save_path
        self.config = config
        self.safety_dist = config["Astar"]["safety_dist"]
        self.lines_dict = self.generate_lines(bitmap, max_distance=config["Astar"]["max_distance"], start=self.start, goal=self.end)
        self.planned_path = self.path(self.start, self.end, self.lines_dict)
        

    def check_collision(self, bitmap, start_pixel, end_pixel, safety_dist):

        # Bresenham's algorithm
        x0, y0 = start_pixel
        x1, y1 = end_pixel
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        line_pixels = []
        while True:
            line_pixels.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        for pixel in line_pixels:
            x, y = pixel
            for i in range(max(0, x - safety_dist), min(300, x + safety_dist + 1)):
                for j in range(max(0, y - safety_dist), min(300, y + safety_dist + 1)):
                    if bitmap[i][j] == 0 and np.sqrt((x - i) ** 2 + (y - j) ** 2) <= safety_dist:
                        return False
        return True
    
    def generate_lines(self, bitmap, max_distance, start, goal):
    # 300 random points in bitmap.
        points = [start, goal]
        while len(points) < 500:
            x = random.randint(20, bitmap.shape[1] - 21)
            y = random.randint(20, bitmap.shape[0] - 21)
            if bitmap[y, x] == 1:
                points.append((x, y))

        lines_dict = {point: [] for point in points}

        for i in range(len(points)):

            # find 10 closest points to form a line with without collision

            distances = [(np.sqrt((points[i][0] - p[0]) ** 2 + (points[i][1] - p[1]) ** 2), p) for p in points if
                        p != points[i]]
            distances.sort()
            closest_points = [p[1] for p in distances[:10]]

            for j in range(len(closest_points)):
                p1, p2 = points[i], closest_points[j]

                if np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) <= max_distance:

                    if not self.check_collision(bitmap, p1, p2, self.safety_dist):
                        continue
                    else:
                        if len(lines_dict[tuple(p1)]) < 15:
                            lines_dict[tuple(p1)].append(tuple(p1))
                            lines_dict[tuple(p1)].append(tuple(p2))
                        if len(lines_dict[tuple(p2)]) < 15:
                            lines_dict[tuple(p2)].append(tuple(p1))
                            lines_dict[tuple(p2)].append(tuple(p2))

        return lines_dict

    # def random_point(self):
    #     x = random.randint(0, self.bitmap.shape[1] - 1)
    #     y = random.randint(0, self.bitmap.shape[0] - 1)
    #     while self.bitmap[y, x] == 0:
    #         x = random.randint(0, self.bitmap.shape[1] - 1)
    #         y = random.randint(0, self.bitmap.shape[0] - 1)
    #     return (x, y)
    
    def heuristic(self, u, v):

            dist = np.sqrt((u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2)
            return dist

    def distance_gen(self, u, v):

        dist = np.sqrt((u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2)
        return dist

    def path(self, start, goal, lines_dict):

        Q = {start: 0}
        cost2reach_d = {start: 0}
        parents = {start: start}
        path0 = []

        while len(Q) > 0:

            Q0 = (list(Q.keys()))[0]

            if Q0 == goal:
                path0 = [goal]
                while parents[path0[0]] != start:
                    path0.insert(0, parents[path0[0]])
                if start != goal:
                    path0.insert(0, start)
                print(path0)
                return path0

            succ = lines_dict[Q0]

            for i in succ:

                wgt = self.distance_gen(Q0, i)
                cost2reach = cost2reach_d[Q0] + wgt
                heur = self.heuristic(i, goal)

                if i in list(cost2reach_d.keys()):
                    if cost2reach_d[i] > cost2reach:
                        cost2reach_d[i] = cost2reach
                        if i in list(Q.keys()):
                            Q[i] = heur/2 + cost2reach
                        parents[i] = Q0

                if i not in list(parents.keys()):
                    parents[i] = Q0
                    cost2reach_d[i] = cost2reach
                    Q[i] = heur + cost2reach

            Q.pop(Q0)
            Q_temp = sorted(Q.items(), key=lambda x: x[1])
            Q = dict(Q_temp)

        return []


if __name__ == "__main__":
    start = np.random.randint((200,200), (250,250), 2)
    goal = np.random.randint((750,700), (800,750), 2)
    #img_path = r"/home/m4/DQN_for_Microrobot_control/test_imgs/img_original_1.png"
    img_path = r"test_imgs/img_1071.png"
    img = cv2.imread(img_path)
    config_path = r"scripts/config.yaml"
    with open(config_path) as p:
        config = yaml.safe_load(p)
    rrt = RRTStar(img, start, goal, save_path=r"test_imgs/path2.png", config=config)
    rrt.plan(show_animation=True)
    