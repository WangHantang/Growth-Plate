import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize

folder_path = 'C:/jg_C3-PBS/yuqi/WKY 2.1. 7M/gp'
image_files = glob.glob(os.path.join(folder_path, '*.jpg'))
for image_file in image_files:
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    # Add a black border to the image
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    # Erode the dilated mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded_base = cv2.erode(image, kernel, iterations=1)
    # Threshold the image to obtain a binary mask of the white regions
    _, binary_mask = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # Define the structuring element for dilation and erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (85, 85))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    # Dilate the binary mask
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Perform sliding average filter on contours
    # Apply median filtering to contours
    window_size = 19
    dilated_mask = cv2.medianBlur(dilated_mask, window_size)
    # Apply Gaussian smoothing to the image
    dilated_mask = cv2.GaussianBlur(dilated_mask, (85, 85), 15)
    _, dilated_mask = cv2.threshold(dilated_mask, 100, 255, cv2.THRESH_BINARY)
    # Erode the dilated mask
    eroded_mask = cv2.erode(dilated_mask, kernel2, iterations=2)
    # Subtract the orinial image from the eroded mask
    subtracted_image = cv2.subtract(eroded_mask, image)
    skeleton = cv2.imread(os.path.join('C:/jg_C3-PBS/yuqi/WKY 2.1. 7M/skeleton', os.path.basename(image_file)))
    skeleton = cv2.copyMakeBorder(skeleton, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    # Apply morphological skeletonization
    #skeleton = skeletonize(eroded_base)
    # Binarize the skeleton image
    _, skeleton = cv2.threshold(skeleton.astype('uint8'), 128, 255, cv2.THRESH_BINARY)
    skeleton = skeleton[:,:,0]
    # Find isolated white points with all black neighbors in the skeleton image
    isolated_points = []
    for y in range(1, skeleton.shape[0] - 1):
        for x in range(1, skeleton.shape[1] - 1):
            if skeleton[y, x] == 255:
                neighbors = skeleton[y-1:y+2, x-1:x+2]
                if np.sum(neighbors) == 255:
                    isolated_points.append((x, y))

    # Change the right neighbor of each isolated point to white
    for point in isolated_points:
        x, y = point
        skeleton[y, x-1] = 255
        skeleton[y, x+1] = 255
    
    class SearchWeldingLine:
        def __init__(self):
            self.component_ids = None
            self.current_id = 0

        def find_component_id(self, m_src, point):
            if self.component_ids is None:
                self.label_components(m_src)
            return self.component_ids[point[1], point[0]]

        def label_components(self, m_src):
            self.component_ids = np.zeros(m_src.shape, dtype=int)
            self.current_id = 0
            visited = np.zeros(m_src.shape, dtype=bool)
            
            for y in range(m_src.shape[0]):
                for x in range(m_src.shape[1]):
                    if m_src[y, x] == 255 and not visited[y, x]:
                        self.current_id += 1
                        self.dfs(m_src, x, y, visited)

        def dfs(self, m_src, x, y, visited):
            stack = [(x, y)]
            while stack:
                cx, cy = stack.pop()
                if visited[cy, cx]:
                    continue
                visited[cy, cx] = True
                self.component_ids[cy, cx] = self.current_id
                for nx, ny in self.get_neighbors(m_src, cx, cy):
                    if m_src[ny, nx] == 255 and not visited[ny, nx]:
                        stack.append((nx, ny))

        def get_neighbors(self, m_src, x, y):
            neighbors = [
                (x-1, y-1), (x, y-1), (x+1, y-1),
                (x-1, y),           (x+1, y),
                (x-1, y+1), (x, y+1), (x+1, y+1)
            ]
            valid_neighbors = [
                (nx, ny) for nx, ny in neighbors
                if 0 <= nx < m_src.shape[1] and 0 <= ny < m_src.shape[0]
            ]
            return valid_neighbors

        def get_8_neighbor_pt(self, m_src, point):
            if m_src[point[1], point[0]] == 0:
                return [], 0
            v_point = []
            neighbors = [(point[0] - 1, point[1] - 1), (point[0], point[1] - 1),
                        (point[0] + 1, point[1] - 1), (point[0] - 1, point[1]),
                        (point[0] + 1, point[1]), (point[0] - 1, point[1] + 1),
                        (point[0], point[1] + 1), (point[0] + 1, point[1] + 1)]
            for neighbor in neighbors:
                x, y = neighbor
                if 0 <= x < m_src.shape[1] and 0 <= y < m_src.shape[0]:
                    if m_src[y, x] == 255:
                        v_point.append((x, y))
            return v_point, len(v_point)

        def endpoints_of_skeleton(self, m_src):
            W,H = m_src.shape[:2]
            img = m_src.copy()
            endpoints = []
            x,y = np.where(img>0)
        
            for i,j in zip(x,y):
                if i-1<0 or j-1<0 or i+1>= W or j+1>= H:
                    continue      
                img = img.astype(np.int16)
                c_a = (img[i-1,j]+img[i+1,j]+img[i,j-1]+img[i,j+1]) == 255
                c_b = (img[i-1,j-1]+img[i-1,j+1]+img[i+1,j-1]+img[i+1,j+1]) == 255
                if (img[i-1:i+2,j-1:j+2]==255).sum() <= 2:
                    endpoints.append([j,i])
                if c_a and c_b:      
                    if (img[i-1,j-1:j+2]==255).sum() == 2 or (img[i+1,j-1:j+2]==255).sum() == 2 or \
                    (img[i-1:i+2,j-1]==255).sum() == 2 or (img[i-1:i+2,j+1]==255).sum() == 2:
                        endpoints.append([j,i])

            return endpoints
        def is_directly_connected(self, img, pt):
            i, j = pt
            neighbour_point = [(i-1, j-1), (i-1, j), (i-1, j+1), (i, j+1), (i+1, j+1), (i+1, j), (i+1, j-1), (i, j-1)]
            for k in range(8):
                if img[neighbour_point[k][1], neighbour_point[k][0]] == 255 and img[neighbour_point[(k+1)%8][1], neighbour_point[(k+1)%8][0]] == 255:
                    return True
            return False
        def remove_burr(self, m_src):
            m_dst = m_src.copy()
            v_terminal_pt = self.endpoints_of_skeleton(m_src)
            n_t_burr = 18
            endpoint_count = {}
            for terminal_pt in v_terminal_pt:
                component_id = self.find_component_id(m_src, terminal_pt)
                if component_id not in endpoint_count:
                    endpoint_count[component_id] = 0
                endpoint_count[component_id] += 1
    
            for terminal_pt in v_terminal_pt:
                last_pt = terminal_pt
                next_pt = terminal_pt
                n_lenth_branch = 0
                b_flag_end = True
                v_burr_pt = []
                component_id = self.find_component_id(m_src, terminal_pt)

                while b_flag_end:
                    v_point = []
                    v_point, n_neighbors = self.get_8_neighbor_pt(m_src, next_pt)
                    if n_neighbors == 1:
                        v_burr_pt.append(next_pt)
                        last_pt = next_pt
                        next_pt = v_point[0]
                        n_lenth_branch += 1
                    elif n_neighbors == 2:
                        v_burr_pt.append(next_pt)
                        if last_pt != v_point[0]:
                            last_pt = next_pt
                            next_pt = v_point[0]
                        else:
                            last_pt = next_pt
                            next_pt = v_point[1]
                        n_lenth_branch += 1
                    elif n_neighbors >= 3:
                        if self.is_directly_connected(m_src, next_pt) is False:
                            b_flag_end = False
                        else:
                            v_burr_pt.append(next_pt)
                            b_flag_end = False
                            n_lenth_branch += 1
                    if n_lenth_branch > n_t_burr:
                        b_flag_end = False

                if n_lenth_branch < n_t_burr and endpoint_count[component_id] > 2:
                    for burr_pt in v_burr_pt:
                        m_dst[burr_pt[1], burr_pt[0]] = 0
                        endpoint_count[component_id] -= 1 
            return m_dst
    if __name__ == "__main__":
        search_welding_line = SearchWeldingLine()
        result_image = search_welding_line.remove_burr(skeleton)
        _, skeleton_cut = cv2.threshold(result_image.astype('uint8'), 0, 255, cv2.THRESH_BINARY)
    def get_8_neighbor_pt(m_src, point):
            if m_src[point[1], point[0]] == 0:
                return [], 0
            v_point = []
            neighbors = [(point[0] - 1, point[1] - 1), (point[0], point[1] - 1),
                        (point[0] + 1, point[1] - 1), (point[0] - 1, point[1]),
                        (point[0] + 1, point[1]), (point[0] - 1, point[1] + 1),
                        (point[0], point[1] + 1), (point[0] + 1, point[1] + 1)]
            for neighbor in neighbors:
                x, y = neighbor
                if 0 <= x < m_src.shape[1] and 0 <= y < m_src.shape[0]:
                    if m_src[y, x] == 255:
                        v_point.append((x, y))
            return v_point, len(v_point)
    def endpoints_of_skeleton(m_src):
        W,H = m_src.shape
        img = m_src.copy()
        endpoints = []
        x,y = np.where(img>0)
        
        for i,j in zip(x,y):
            if i-1<0 or j-1<0 or i+1>= W or j+1>= H:
                continue      
            img = img.astype(np.int16)
            c_a = (img[i-1,j]+img[i+1,j]+img[i,j-1]+img[i,j+1]) == 255
            c_b = (img[i-1,j-1]+img[i-1,j+1]+img[i+1,j-1]+img[i+1,j+1]) == 255
            if (img[i-1:i+2,j-1:j+2]==255).sum() <= 2:
                endpoints.append([j,i])
            if c_a and c_b:      
                if (img[i-1,j-1:j+2]==255).sum() == 2 or (img[i+1,j-1:j+2]==255).sum() == 2 or \
                (img[i-1:i+2,j-1]==255).sum() == 2 or (img[i-1:i+2,j+1]==255).sum() == 2:
                    endpoints.append([j,i])
        return endpoints
    def connect_endpoints(result_image, endpoints):
        # Sort the endpoints based on x-axis values
        sorted_endpoints = sorted(endpoints, key=lambda pt: pt[0])
        
        # Remove the first and last endpoints
        sorted_endpoints = sorted_endpoints[1:-1]
        connected_image = np.zeros_like(skeleton_cut)
        # Plot the image
        if len(sorted_endpoints) > 0:
            
            # Pair the remaining endpoints
            paired_points = zip(sorted_endpoints[::2], sorted_endpoints[1::2])

            # Iterate over the paired points
            for point1, point2 in paired_points:
                # Draw a white line connecting the paired endpoints
                cv2.line(connected_image, tuple(point1), tuple(point2), (255, 255, 255), 1)

            # Calculate the length of the line
                length = int(np.linalg.norm(np.array(point2) - np.array(point1)))
             # Check if the distance is less than 2
                if length < 50:
                # Treat point1 and point2 as the same point
                    ratio = 0.2
                    x1, y1 = int(point1[0] - (ratio) * point2[1]+ (ratio) * point1[1]), int(point1[1] - (ratio) * point1[0]+ (ratio) * point2[0])
                    x2, y2 = int(point1[0] + (ratio) * point2[1]- (ratio) * point1[1]), int(point1[1] + (ratio) * point1[0]- (ratio) * point2[0])
                    x3, y3 = int(point2[0] + (ratio) * point2[1]- (ratio) * point1[1]), int(point2[1] + (ratio) * point1[0]- (ratio) * point2[0])
                    x4, y4 = int(point2[0] - (ratio) * point2[1]+ (ratio) * point1[1]), int(point2[1] - (ratio) * point1[0]+ (ratio) * point2[0])
                    rectangular = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                    cv2.fillConvexPoly(connected_image, rectangular, (255,255,255))
                else:
                    ratio = 0.05
                    x1, y1 = int(point1[0] - (ratio) * point2[1]+ (ratio) * point1[1]), int(point1[1] - (ratio) * point1[0]+ (ratio) * point2[0])
                    x2, y2 = int(point1[0] + (ratio) * point2[1]- (ratio) * point1[1]), int(point1[1] + (ratio) * point1[0]- (ratio) * point2[0])
                    x3, y3 = int(point2[0] + (ratio) * point2[1]- (ratio) * point1[1]), int(point2[1] + (ratio) * point1[0]- (ratio) * point2[0])
                    x4, y4 = int(point2[0] - (ratio) * point2[1]+ (ratio) * point1[1]), int(point2[1] - (ratio) * point1[0]+ (ratio) * point2[0])
                    rectangular = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                    cv2.fillConvexPoly(connected_image, rectangular, (255,255,255))
            return connected_image
        else:
            connected_image = np.zeros_like(skeleton_cut)
    
    endpoints = endpoints_of_skeleton(skeleton_cut)
    connected_image = connect_endpoints(skeleton_cut, endpoints)
    # Convert connected_image to binary
    if connected_image is not None:
        connected_binary = connected_image.copy()
        # Keep only the white regions in subtracted_image that correspond to the white regions in connected_binary
        bony_tethers = subtracted_image.copy()
        bony_tethers[connected_binary == 0] = 0
        output_file = os.path.join(folder_path, 'processed', os.path.basename(image_file))
        cv2.imwrite(output_file, bony_tethers)
    else:
        output_file = os.path.join(folder_path, 'processed', os.path.basename(image_file))
        black_image = np.zeros_like(skeleton_cut, dtype=np.uint8)
        cv2.imwrite(output_file, black_image)