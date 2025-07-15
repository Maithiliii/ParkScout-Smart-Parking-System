import numpy as np
from math import sqrt
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Define scaling function
def scale_coordinates(coords, image_size, target_size, scale_factor=1.0):
    scale_x = target_size[0] / image_size[0]
    scale_y = target_size[1] / image_size[1]
    return [(point[0] * scale_x * scale_factor, point[1] * scale_y * scale_factor) for point in coords]

# Original parking space coordinates and obstacles
parking_spaces_orig = [
    [[57, 16], [88, 16], [88, 74], [57, 74]],
    [[88, 16], [121, 16], [121, 75], [88, 74]],
    [[122, 16], [156, 16], [155, 76], [121, 75]],
    [[157, 16], [191, 18], [190, 75], [156, 75]],
    [[191, 16], [224, 17], [224, 74], [191, 74]],
    [[59, 144], [88, 143], [87, 198], [58, 198]],
    [[88, 143], [121, 144], [122, 199], [88, 199]],
    [[122, 143], [155, 144], [155, 199], [122, 198]],
    [[155, 144], [189, 144], [189, 199], [155, 198]],
    [[189, 143], [222, 144], [223, 199], [189, 199]],
    [[58, 225], [87, 224], [88, 278], [57, 279]],
    [[86, 224], [121, 224], [121, 279], [88, 278]],
    [[121, 224], [154, 224], [155, 279], [121, 278]],
    [[155, 224], [189, 224], [189, 280], [155, 278]],
    [[189, 224], [223, 224], [223, 280], [189, 279]],
    [[311, 28], [368, 26], [369, 58], [310, 60]],
    [[309, 59], [369, 58], [368, 90], [310, 92]],
    [[310, 92], [368, 90], [368, 121], [311, 123]],
    [[311, 124], [369, 122], [368, 153], [312, 154]],
    [[312, 155], [368, 154], [368, 184], [311, 186]],
    [[311, 187], [368, 185], [368, 216], [310, 218]],
    [[310, 218], [368, 217], [368, 248], [311, 250]],
    [[311, 251], [368, 249], [368, 279], [309, 281]],
    [[309, 281], [368, 280], [368, 310], [311, 312]],
]
obstacles_orig = [
    [[55, 0], [56, 17], [225, 18], [226, 0]],
    [[58, 199], [225, 199], [224, 225], [57, 224]],
    [[386, 25], [369, 25], [370, 340], [386, 341]],
]

image_size = (387, 343)
target_size = (387, 343)

# Scale coordinates
parking_spaces = [scale_coordinates(space, image_size, target_size) for space in parking_spaces_orig]
obstacles = [scale_coordinates(obstacle, image_size, target_size) for obstacle in obstacles_orig]


class ParkingLotDetector:
    def __init__(self, reference_path):
        self.reference_img = cv2.imread(reference_path)
        if self.reference_img is None:
            raise ValueError(f"Could not load reference image from {reference_path}")
        self.reference_img = cv2.resize(self.reference_img, target_size)
        self.reference_img = cv2.cvtColor(self.reference_img, cv2.COLOR_BGR2RGB)

        self.spaces = parking_spaces
        self.obstacles = obstacles
        self.masks = self._create_space_masks()
        self.road_mask = self._create_road_mask()
        self.legend_shown = False  # Flag to ensure the legend is shown only once

    def _create_space_masks(self):
        masks = []
        height, width = target_size[1], target_size[0]
        for space in self.spaces:
            mask = np.zeros((height, width), dtype=np.uint8)
            space_points = np.array(space, dtype=np.int32)
            cv2.fillPoly(mask, [space_points], 255)
            masks.append(mask)
        return masks

    def _create_road_mask(self):
        height, width = target_size[1], target_size[0]
        road_mask = np.ones((height, width), dtype=np.uint8) * 255
        for space in self.spaces:
            space_points = np.array(space, dtype=np.int32)
            cv2.fillPoly(road_mask, [space_points], 0)
        for obstacle in self.obstacles:
            obstacle_points = np.array(obstacle, dtype=np.int32)
            cv2.fillPoly(road_mask, [obstacle_points], 0)
        return road_mask

    def calculate_centroid(self, points):
        x = sum(p[0] for p in points) / len(points)
        y = sum(p[1] for p in points) / len(points)
        return int(x), int(y)

    def detect_occupied_spaces(self, test_img_path, threshold=30):
        test_img = cv2.imread(test_img_path)
        if test_img is None:
            raise ValueError(f"Could not load test image from {test_img_path}")
        test_img = cv2.resize(test_img, target_size)

        ref_gray = cv2.cvtColor(self.reference_img, cv2.COLOR_RGB2GRAY)
        test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(ref_gray, test_gray)

        occupied = set()
        for idx, mask in enumerate(self.masks):
            mean_diff = cv2.mean(diff, mask=mask)[0]
            if mean_diff > threshold:
                occupied.add(idx)

        return occupied

    def calculate_grid_route(self, start_pos, target_pos, follow_vertical_line=False):
        route = []
        current_pos = list(start_pos)

        if follow_vertical_line:
            # Define the vertical line (middle of the road)
            line_x = 269  # x-coordinate for the vertical line

            # Move horizontally to align with the vertical line
            while current_pos[0] != line_x:
                if current_pos[0] < line_x:
                    current_pos[0] += 1
                else:
                    current_pos[0] -= 1
                if self.road_mask[current_pos[1], current_pos[0]] == 0:
                    continue  # Skip blocked path
                route.append(tuple(current_pos))

        # Move vertically towards the target position
        while current_pos[1] != target_pos[1]:
            if current_pos[1] < target_pos[1]:
                current_pos[1] += 1
            else:
                current_pos[1] -= 1
            if self.road_mask[current_pos[1], current_pos[0]] == 0:
                continue  # Skip blocked path
            route.append(tuple(current_pos))

        # Adjust to enter the parking space from the front
        final_pos = list(target_pos)
        final_pos[1] = target_pos[1] - 1  # Shift 1 step vertically to the front
        route.append(tuple(final_pos))

        return np.array(route)

    def display_path_interactive(self, test_img_path):
        occupied_spaces = self.detect_occupied_spaces(test_img_path)
        free_spaces = [idx for idx in range(len(self.spaces)) if idx not in occupied_spaces]

        if not free_spaces:
            print("No free spaces available!")
            return

        # Prepare plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(self.reference_img)

        # Draw parking spaces
        for idx, space in enumerate(self.spaces):
            color = 'red' if idx in occupied_spaces else 'green'
            polygon = Polygon(np.array(space), fill=False, color=color, linewidth=1.5)
            ax.add_patch(polygon)
            cx, cy = self.calculate_centroid(space)
            ax.text(cx, cy, str(idx), color=color, ha='center', va='center')

        # Draw obstacles
        for obstacle in self.obstacles:
            polygon = Polygon(np.array(obstacle), fill=False, color='blue', linewidth=1.5)
            ax.add_patch(polygon)

        # Handle user click to select the car's position
        def onclick(event):
            car_pos = (int(event.xdata), int(event.ydata))
            print(f"Car position: {car_pos}")

    # Find the nearest free space
            min_distance = float('inf')
            nearest_space_idx = None
            for idx in free_spaces:
                centroid = self.calculate_centroid(self.spaces[idx])
                distance = sqrt((car_pos[0] - centroid[0])**2 + (car_pos[1] - centroid[1])**2)
                if distance < min_distance:
                    min_distance = distance
                    nearest_space_idx = idx

            print(f"Nearest free space: {nearest_space_idx}")

    # Calculate route
            target_pos = self.calculate_centroid(self.spaces[nearest_space_idx])
            route = self.calculate_grid_route(car_pos, target_pos)

    # Clear previous route and update with the new one
            ax.clear()
            ax.imshow(self.reference_img)

    # Redraw parking spaces
            for idx, space in enumerate(self.spaces):
                color = 'red' if idx in occupied_spaces else 'green'
                polygon = Polygon(np.array(space), fill=False, color=color, linewidth=1.5)
                ax.add_patch(polygon)
                cx, cy = self.calculate_centroid(space)
                ax.text(cx, cy, str(idx), color=color, ha='center', va='center')

    # Redraw obstacles
            for obstacle in self.obstacles:
                polygon = Polygon(np.array(obstacle), fill=False, color='blue', linewidth=1.5)
                ax.add_patch(polygon)

    # Plot the car position, target position, and route
            ax.plot(car_pos[0], car_pos[1], 'yo', markersize=8, label='Car Position ')
            ax.plot(target_pos[0], target_pos[1], 'go', markersize=8, label='Free Space ')
            ax.plot(route[:, 0], route[:, 1], 'b-', label='Path to Free Space ')

    # Add legend only once
            if not self.legend_shown:
                ax.legend()
                self.legend_shown = True

            fig.canvas.draw()

        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()


# Example usage
detector = ParkingLotDetector("C:/Users/HP/Desktop/lotupdated.jpg")
detector.display_path_interactive("C:/Users/HP/Downloads/lottest.png")
