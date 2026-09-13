import numpy as np
import os

def calculate_recall(coords1, coords2):
   set1 = set(coords1)
   set2 = set(coords2)
   TP = set1 & set2
   FN = set1 - set2
   recall = (len(TP) / (len(TP) + len(FN))) * 100
   return recall

def calculate_precision(coords1, coords2):
   set1 = set(coords1)
   set2 = set(coords2)
   TP = set1 & set2
   FP = set2 - set1
   precision = (len(TP) / (len(TP) + len(FP))) * 100
   return precision

# Directories containing text files
canny_dirs = [
   r'C:\Users\assist-lab\Desktop\FV\canny_textfiles\canny_textfiles_1',
   r'C:\Users\assist-lab\Desktop\FV\canny_textfiles\canny_textfiles_2',
   r'C:\Users\assist-lab\Desktop\FV\canny_textfiles\canny_textfiles_3',
]

alina_dirs = [
   r'C:\Users\assist-lab\Desktop\FV\ALINA_textfiles\ALINA_textfiles_1',
   r'C:\Users\assist-lab\Desktop\FV\ALINA_textfiles\ALINA_textfiles_2',
   r'C:\Users\assist-lab\Desktop\FV\ALINA_textfiles\ALINA_textfiles_3',
]

# Loop through each text file in the directory
iterator = 0
recall_values = []
precision_values = []
for dir in canny_dirs:
   for filename in os.listdir(dir):
       if filename.endswith('.txt'):
           # Load x and y coordinates from Canny text file
           with open(os.path.join(dir, filename)) as f:
               lines = f.readlines()

           # Extract x and y coordinates and convert to numpy arrays
           x1 = np.zeros((len(lines),), dtype=np.int32)
           y1 = np.zeros((len(lines),), dtype=np.int32)
           for i, line in enumerate(lines):
               x, y = line.strip().split()
               x1[i] = int(x)
               y1[i] = int(y)

           # Load x and y coordinates from Alina text file
           with open(os.path.join(alina_dirs[iterator], filename)) as f:
               lines = f.readlines()

           # Extract x and y coordinates and convert to numpy arrays
           x2 = np.zeros((len(lines),), dtype=np.int32)
           y2 = np.zeros((len(lines),), dtype=np.int32)
           for i, line in enumerate(lines):
               x, y = line.strip().split()
               x2[i] = int(x)
               y2[i] = int(y)

           # Calculate IoU over x-coordinates
           recall_x = calculate_recall(x1, x2)
           precision_x = calculate_precision(x1, x2)
           # Calculate IoU over y-coordinates
           recall_y = calculate_recall(y1, y2)
           precision_y = calculate_precision(y1, y2)
           # Combine the two results
           recall = (recall_x + recall_y) / 2
           precision = (precision_x + precision_y) / 2
           recall_values.append(recall)
           precision_values.append(precision)
   iterator += 1

# Print the average recall over all text files
avg_recall = np.mean(recall_values)
print("Average Recall:", avg_recall, '%')

# Print the average precision over all text files
avg_precision = np.mean(precision_values)
print("Average Precision:", avg_precision, '%')

