import numpy as np
import cv2
import math
import yaml
import time
import random
import glob
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from skimage.draw import line, line_aa, line_nd
import math

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
    def __init__(self, img, save_path, config, plot_img=None):
        #self.img = img
        #self.binary_img = self.generate_bitmap(img, sam_config=config["sam_config"])
        self.binary_img = img 
        self.start = None
        self.end = None
        self.max_iter = config["max_iter"]
        self.goal_sample_rate = config["goal_sample_rate"]
        self.max_distance = config["max_distance"]
        self.path = None
        self.save_path = save_path
        threshold = config["safety_threshold"]
        if config["invert_dilation"]:
            self.dilated_img = self.inverse_dilation(img, threshold)
            self.dilated_img = self.dilate_bitmap(self.dilated_img, 1)    
        else:
            self.dilated_img = self.dilate_bitmap(img, threshold)
        self.plot_img = plot_img
    
    def set_start(self, start):
        self.start = Node(start[0], start[1])
        self.nodes = [self.start]
        self.shortest_path = float("inf")
    
    def set_end(self, end):
        self.end = Node(end[0], end[1])
    
    def set_start_end(self, start, end):
        self.start = Node(start[0], start[1])
        self.end = Node(end[0], end[1])
        self.shortest_path = float("inf")
        self.nodes = [self.start]
        self.path = None
    
    def reset(self):
        self.start = None
        self.end = None
        self.nodes = []
        self.path = None
        self.shortest_path = float("inf")

    @staticmethod
    def dilate_bitmap(bitmap, threshold):
        """
        Dilates the obstacles in the binary image by the safety threshold.

        Args:
            bitmap (numpy.ndarray): The binary image.

        Returns:
            numpy.ndarray: The dilated binary image.

        """
        safety_threshold = threshold
        bitmap_copy = bitmap.copy()
        for i in range(bitmap.shape[0]):
            for j in range(bitmap.shape[1]):
                if bitmap[i][j] == 0 and np.any(bitmap[max(0, i - safety_threshold):min(bitmap.shape[0], i + safety_threshold + 1), max(0, j - safety_threshold):min(bitmap.shape[1], j + safety_threshold + 1)]) == 1:
                    for k in range(max(0, i - safety_threshold), min(bitmap.shape[0], i + safety_threshold + 1)):
                        for l in range(max(0, j - safety_threshold), min(bitmap.shape[1], j + safety_threshold + 1)):
                            bitmap_copy[k][l] = 0
        # cv2.imshow("Dilated Image", bitmap_copy*255)
        # cv2.imshow("Original Image", bitmap*255)
        # delta = cv2.subtract(bitmap, bitmap_copy)
        # red_delta = np.stack((delta*(255), np.zeros_like(delta), np.zeros_like(delta)), axis=-1)
        # show = cv2.addWeighted(cv2.cvtColor(bitmap*255, cv2.COLOR_GRAY2BGR),
        #                        0.5, red_delta, 0.5, 0.0)
        # cv2.imshow("Delta", show)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return bitmap_copy
    
    @staticmethod
    def inverse_dilation(bitmap, threshold, plot=False):
        """
        Creates an inverse dilation, useful to have the path planning "stick" to the walls
        
        Args:
            bitmap (numpy.ndarray): The binary image.

        Returns:
            numpy.ndarray: The dilated binary image.

        """
        safety_threshold = threshold
        bitmap_copy = bitmap.copy()
        for i in range(bitmap.shape[0]):
            for j in range(bitmap.shape[1]):
                if bitmap[i][j] == 1 and np.all(bitmap[max(0, i - safety_threshold):min(bitmap.shape[0], i + safety_threshold + 1), max(0, j - safety_threshold):min(bitmap.shape[1], j + safety_threshold + 1)]) == 1:
                    bitmap_copy[i][j] = 0
                delta = cv2.subtract(bitmap, bitmap_copy)
        if plot:
            red_delta = np.stack((delta*(255), np.zeros_like(delta), np.zeros_like(delta)), axis=-1)
            show = cv2.addWeighted(cv2.cvtColor(bitmap*255, cv2.COLOR_GRAY2BGR),
                                   0.5, red_delta, 0.5, 0.0)
            cv2.imshow("Delta", show)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return bitmap_copy

    
    def inside_obstacle(self, point):
        """
        Checks if the node is inside an obstacle.

        Args:
            node (Node): The node to check.

        Returns:
            bool: Whether the node is inside an obstacle or not.

        """
        return self.dilated_img[point[1], point[0]] == 0

    def plan(self, show_animation=False, save_folder=None):
        """
        Plans the path from start to end using the RRT* algorithm.

        Args:
            show_animation (bool): Whether to show the animation or not.

        Returns:
            Path: The path from start to end.

        """
        assert self.start is not None, "Start node is not set"
        assert self.end is not None, "End node is not set"

        for i in range(self.max_iter):
            if np.random.randint(0, 100) > self.goal_sample_rate:
                rnd = Node(np.random.randint(0, self.binary_img.shape[1]), np.random.randint(0, self.binary_img.shape[0]))
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
                            print(f"Found path with length {path_len}")
                            self.animation(self.nodes, self.path, save_folder, path_len)
                            plt.show(block=False)
                        # plt.pause(0.1)

                elif self.path is None and i % 25 == 0 and show_animation:
                    _temp_path, path_len = self.get_path(self.start)
                    self.animation(self.nodes, _temp_path, save_folder, np.inf)
                    plt.show(block=False)
                    # plt.pause(0.1)

        if self.path is not None:
            # if show_animation:
            #     self.animation(self.nodes, self.path, save_folder)
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
        return np.any(self.dilated_img[cc, rr])

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
        if distance_to_goal > self.max_distance*2:
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

    def animation(self, nodes, path, save_path=None, path_len=np.inf):
        """
        Animates the path.

        Args:
            path (Path): The path to animate.

        """
        plt.cla()
        plt.plot(self.start.x, self.start.y, "bo", linewidth=3)
        plt.plot(self.end.x, self.end.y, "ro", linewidth=3)
        plt.legend(['Start', 'Goal'])
        imgplot = plt.imshow(self.plot_img)
        
        for node in nodes:
            if node.parent is not None:
                plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "b-", linewidth=0.2)
        x_coords = [node[0] for node in path]
        y_coords = [node[1] for node in path]
        plt.plot(x_coords, y_coords, "r-", linewidth=1)
        plt.title(f"RRT* Path Length: {path_len:.2f}")
        
        if save_path is not None:
            imgs = glob.glob(f"{save_path}/*.png")
            if imgs:
                imgs = sorted(imgs, key=lambda x: int(x.split("_")[-1].split(".")[0]))
                last_img = imgs[-1]
                idx = int(last_img.split("_")[-1].split(".")[0]) + 1
            else:
                idx = 0
            plt.axis('off')
            plt.savefig(f"{save_path}/rrt_path_{idx}.png")




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



def dilate_bitmap_testing(bitmap, safety_threshold):
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

if __name__ == "__main__":
    
    with open(f'scripts/config_continous.yaml', 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    img = "binary_images/default_segmentation_vascular_4_closed.png"

    segmented = (np.asarray(cv2.imread(img, cv2.IMREAD_GRAYSCALE), dtype=np.uint8)/255).astype(np.uint8)
    
    planner = RRTStar(segmented, True, config['Path Planning'])
    