# import cv2
# import numpy as np

# class QuadNode:
#     def __init__(self, x, y, w, h, data):
#         self.x = x
#         self.y = y
#         self.w = w
#         self.h = h
#         self.data = data
#         self.childs = []
#         self.average = None  # To store average pixel value

# class QuadTree:
#     def __init__(self, image):
#         self.root = QuadNode(0, 0, image.shape[1], image.shape[0], image)

#     def calculate_average(self, node):
#         """Calculates the average pixel value for the given node data"""
#         if node.data.size > 0: 
#             return np.mean(node.data)
#         return 0

#     def calculate_error(self, node, average):
#         """Calculates the error based on how much pixel values differ from the average"""
#         if node.data.size > 0:
#             return np.mean((node.data - average) ** 2)
#         return 0 

#     def subdivide(self, node, threshold):
#         """Subdivides the node if the error exceeds the threshold"""
#         average = self.calculate_average(node)
#         node.average = average
        
#         error = self.calculate_error(node, average)

#         if error > threshold:  # Checking if the error exceeds the specified threshold
#             half_w = node.w // 2
#             half_h = node.h // 2
            
#             # Creating child nodes
#             child1 = QuadNode(node.x, node.y, half_w, half_h, node.data[node.y:node.y + half_h, node.x:node.x + half_w])  # Top-left
#             child2 = QuadNode(node.x + half_w, node.y, half_w, half_h, node.data[node.y:node.y + half_h, node.x + half_w:node.x + node.w])  # Top-right
#             child3 = QuadNode(node.x, node.y + half_h, half_w, half_h, node.data[node.y + half_h:node.y + node.h, node.x:node.x + half_w])  # Bottom-left
#             child4 = QuadNode(node.x + half_w, node.y + half_h, half_w, half_h, node.data[node.y + half_h:node.y + node.h, node.x + half_w:node.x + node.w])  # Bottom-right
            
#             # Appending the child nodes to the parent node's list
#             node.childs.extend([child1, child2, child3, child4])

#             # recursively subdivide the child nodes
#             self.subdivide(child1, threshold)
#             self.subdivide(child2, threshold)
#             self.subdivide(child3, threshold)
#             self.subdivide(child4, threshold)

#     def build(self, threshold):
#         """ Builds the quad-tree based on the error threshold"""
#         self.subdivide(self.root, threshold)  # Start subdividing from the root node

#     def create_segmented_image(self, image_shape):
#         """Creates a segmented image based on the quad-tree structure"""
#         segmented_image = np.zeros(image_shape, dtype=np.uint8)
        
#         self._fill_segmented_image(self.root, segmented_image)
        
#         return segmented_image

#     def _fill_segmented_image(self, node, segmented_image):
#         """Recursively fill the segmented image based on the quad-tree nodes"""
#         # Ensure that the node has a valid average before filling the image
#         if node.average is not None:
#             segmented_image[node.y:node.y + node.h, node.x:node.x + node.w] = node.average

#         # Recursively fill for each child node if it has children
#         for child in node.childs:
#             self._fill_segmented_image(child, segmented_image)

#     def visualize(self, image, node, depth=0):
#         """Recursively draw the quad-tree on the image with enhanced visualization"""
#         # Color map for different depths
#         color_map = [
#             (0, 255, 0),    # Depth 0 (green)
#             (255, 0, 0),    # Depth 1 (red)
#             (0, 0, 255),    # Depth 2 (blue)
#             (255, 255, 0),  # Depth 3 (cyan)
#             (255, 0, 255),  # Depth 4 (magenta)
#             (0, 255, 255)   # Depth 5 (yellow)
#         ]

#         # Ensure depth doesn't exceed the color map length
#         color = color_map[depth % len(color_map)]

#         # Draw a rectangle around the current node
#         cv2.rectangle(image, (node.x, node.y), (node.x + node.w, node.y + node.h), color, 1)

#         # Add average pixel value text, but only if it's a valid number
#         if node.average is not None and not np.isnan(node.average):
#             cv2.putText(image, f'{int(node.average)}', (node.x + 5, node.y + 15), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

#         # Recursively draw for each child node if it has children
#         for child in node.childs:
#             self.visualize(image, child, depth + 1)

# # Driver Code
# if __name__ == "__main__":
#     path = r'C:\Users\navee\OneDrive\Desktop\Image_Segmentation_Project\image.jpg'
#  # Paste image path 
#     image = cv2.imread(path)
#     if image is None:
#         print("Image not found")
#     else:
#         #image = cv2.resize(image, dsize=(64, 64))
#         grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#         quad_tree = QuadTree(grayscale_image)
#         error_threshold = 50 
#         quad_tree.build(threshold=error_threshold)  

#         # Create a segmented image based on the quad-tree
#         segmented_image = quad_tree.create_segmented_image(grayscale_image.shape)

#         # Visualize the quad-tree by drawing rectangles on a copy of the grayscale image
#         image_with_quadtree = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)
#         quad_tree.visualize(image_with_quadtree, quad_tree.root)

#         # Displaying the images
#         cv2.imshow("QuadTree Visualization", image_with_quadtree)
#         cv2.imshow("Segmented Image", segmented_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

import cv2
import numpy as np

# Node representing a section of the image
class QuadNode:
    def __init__(self, x, y, w, h, data):
        self.x = x  # X-coordinate in original image
        self.y = y  # Y-coordinate in original image
        self.w = w  # Width of region
        self.h = h  # Height of region
        self.data = data  # Pixel data in this region
        self.childs = []  # List of child nodes
        self.average = None  # Average grayscale value

# Class for building and handling the QuadTree
class QuadTree:
    def __init__(self, image):
        self.root = QuadNode(0, 0, image.shape[1], image.shape[0], image)

    def calculate_average(self, node):
        if node.data.size > 0:
            return np.mean(node.data)
        return 0

    def calculate_error(self, node, average):
        if node.data.size > 0:
            return np.mean((node.data - average) ** 2)
        return 0

    def subdivide(self, node, threshold):
        average = self.calculate_average(node)
        node.average = average
        error = self.calculate_error(node, average)

        # Subdivide only if error is high and region is bigger than 1 pixel
        if error > threshold and node.w > 1 and node.h > 1:
            half_w = node.w // 2
            half_h = node.h // 2
            data = node.data

            # Correct slicing using relative indices
            child1 = QuadNode(node.x, node.y, half_w, half_h, data[0:half_h, 0:half_w])  # Top-left
            child2 = QuadNode(node.x + half_w, node.y, node.w - half_w, half_h, data[0:half_h, half_w:node.w])  # Top-right
            child3 = QuadNode(node.x, node.y + half_h, half_w, node.h - half_h, data[half_h:node.h, 0:half_w])  # Bottom-left
            child4 = QuadNode(node.x + half_w, node.y + half_h, node.w - half_w, node.h - half_h, data[half_h:node.h, half_w:node.w])  # Bottom-right

            node.childs.extend([child1, child2, child3, child4])
            self.subdivide(child1, threshold)
            self.subdivide(child2, threshold)
            self.subdivide(child3, threshold)
            self.subdivide(child4, threshold)

    def build(self, threshold):
        self.subdivide(self.root, threshold)

    def create_segmented_image(self, image_shape):
        # Blank image to store final segmented output
        segmented_image = np.zeros(image_shape, dtype=np.uint8)
        self._fill_segmented_image(self.root, segmented_image)
        return segmented_image

    def _fill_segmented_image(self, node, segmented_image):
        # Fill the region with average grayscale value
        if node.average is not None:
            avg_val = np.clip(node.average, 0, 255).astype(np.uint8)
            segmented_image[node.y:node.y + node.h, node.x:node.x + node.w] = avg_val

            # Draw visible white border around segment
            cv2.rectangle(segmented_image, (node.x, node.y), (node.x + node.w, node.y + node.h), (255,), 1)

        for child in node.childs:
            self._fill_segmented_image(child, segmented_image)

    def visualize(self, image, node, depth=0):
        # Colored border map for depth levels
        color_map = [
            (0, 255, 0),    # green
            (255, 0, 0),    # red
            (0, 0, 255),    # blue
            (255, 255, 0),  # cyan
            (255, 0, 255),  # magenta
            (0, 255, 255)   # yellow
        ]
        color = color_map[depth % len(color_map)]

        # Draw bounding box for current region
        cv2.rectangle(image, (node.x, node.y), (node.x + node.w, node.y + node.h), color, 1)

        # Optionally label with average grayscale value
        if node.average is not None and not np.isnan(node.average):
            cv2.putText(image, f'{int(node.average)}', (node.x + 2, node.y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        for child in node.childs:
            self.visualize(image, child, depth + 1)

# --------------- Main Execution ----------------
if __name__ == "__main__":
    path = r'C:\Users\navee\OneDrive\Desktop\Image_Segmentation_Project\image.jpg'
    image = cv2.imread(path)

    if image is None:
        print("Image not found")
    else:
        # Convert to grayscale for segmentation
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Initialize the QuadTree and build it
        quad_tree = QuadTree(grayscale_image)
        error_threshold = 50  # Lower = more segments, Higher = smoother
        quad_tree.build(threshold=error_threshold)

        # Get segmented image (with white borders for clarity)
        segmented_image = quad_tree.create_segmented_image(grayscale_image.shape)

        # Create a color visualization overlay
        image_with_quadtree = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)
        quad_tree.visualize(image_with_quadtree, quad_tree.root)

        # Optional: Create color-mapped version for enhanced contrast
        colored_segmented = cv2.applyColorMap(segmented_image, cv2.COLORMAP_JET)

        # Show the results
        cv2.imshow("Original Grayscale", grayscale_image)
        cv2.imshow("Segmented (Grayscale + Borders)", segmented_image)
        cv2.imshow("Colored Segmented Image", colored_segmented)
        cv2.imshow("QuadTree Structure Visualization", image_with_quadtree)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
