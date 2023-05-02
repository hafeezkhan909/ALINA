import numpy as np
import os

def calculate_iou(coords1, coords2):
    set1 = set(coords1)
    set2 = set(coords2)
    intersection = set1 & set2
    inter = set1 & set2
    total = inter | set2
    union = set1 | set2
    iou = len(intersection) / len(total)
    return iou

# Directories containing text files
canny_dir = r'C:\Users\assist-lab\Desktop\FV\canny_textfiles'
alina_dir = r'C:\Users\assist-lab\Desktop\FV\ALINA_textfiles1'

# Loop through each text file in the directory
iou_values = []
for filename in os.listdir(canny_dir):
    if filename.endswith('.txt'):
        # Load x and y coordinates from Canny text file
        with open(os.path.join(canny_dir, filename)) as f:
            lines = f.readlines()

        # Extract x and y coordinates and convert to numpy arrays
        x1 = np.zeros((len(lines),), dtype=np.int32)
        y1 = np.zeros((len(lines),), dtype=np.int32)
        for i, line in enumerate(lines):
            x, y = line.strip().split()
            x1[i] = int(x)
            y1[i] = int(y)

        # Load x and y coordinates from Alina text file
        with open(os.path.join(alina_dir, filename)) as f:
            lines = f.readlines()

        # Extract x and y coordinates and convert to numpy arrays
        x2 = np.zeros((len(lines),), dtype=np.int32)
        y2 = np.zeros((len(lines),), dtype=np.int32)
        for i, line in enumerate(lines):
            x, y = line.strip().split()
            x2[i] = int(x)
            y2[i] = int(y)

        # Calculate IoU over x-coordinates
        iou_x = calculate_iou(x1, x2)

        # Calculate IoU over y-coordinates
        iou_y = calculate_iou(y1, y2)

        # Combine the two results
        iou = (iou_x + iou_y) / 2
        iou_values.append(iou)

# Print the average IoU over all text files
avg_iou = np.mean(iou_values)
print("Average IoU:", avg_iou)
