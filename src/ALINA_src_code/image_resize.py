import cv2
import os

# Path to the folder containing images
input_folder_path = "C:\\Users\\assist-lab\\Desktop\\output\\1\\"

# Path to the folder to save the resized images
output_folder_path = "C:\\Users\\assist-lab\\Desktop\\resized_vidd_copy_2\\"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# Set the desired output width and height
output_width = 1920
output_height = 1080

# Loop through all images in the folder
for filename in os.listdir(input_folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):

        # Load the image
        input_img_path = os.path.join(input_folder_path, filename)
        img = cv2.imread(input_img_path)

        # Get the current image dimensions
        height, width, _ = img.shape

        # Calculate the resize ratio
        resize_ratio = min(output_width / width, output_height / height)

        # Calculate the new dimensions
        new_width = int(width * resize_ratio)
        new_height = int(height * resize_ratio)

        # Resize the image
        resized_img = cv2.resize(img, (new_width, new_height))

        # Add black padding if necessary
        if new_width < output_width or new_height < output_height:
            top_pad = int((output_height - new_height) / 2)
            bottom_pad = output_height - new_height - top_pad
            left_pad = int((output_width - new_width) / 2)
            right_pad = output_width - new_width - left_pad
            resized_img = cv2.copyMakeBorder(resized_img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT,
                                             value=(0, 0, 0))

        # Construct the output path and save the resized image
        output_img_path = os.path.join(output_folder_path, filename)
        cv2.imwrite(output_img_path, resized_img)

        # Print a message for each image that is resized
        print(f'Resized {filename}')
