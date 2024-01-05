import cv2
import numpy as np

# Load all images
images = [cv2.imread(f'Preprocessing/Concatenate_Dataset/20230626_1_set_2_1.p_frame_{i}.png') for i in range(153, 163)]

# Assuming all images are of the same size, initialize the result with an array of zeros
# The shape of the result is the same as each image
result = np.zeros_like(images[0])

# Add each image to the result
for img in images:
    result = cv2.add(result, img)

# Normalize the result if necessary (e.g., if you want to scale pixel values to a certain range)

# Display the final image
cv2.imshow('Merged Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()




