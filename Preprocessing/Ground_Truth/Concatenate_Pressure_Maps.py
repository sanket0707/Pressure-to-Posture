import cv2
import numpy as np

# Load all images
images = [cv2.imread(f'Preprocessing/Concatenate_Dataset/20230626_1_set_2_1.p_frame_{i}.png') for i in range(153, 163)]

# Initialize the result with an array of zeros
result = np.zeros_like(images[0], dtype=np.float32)

# Add each image to the result
for img in images:
    result += img.astype(np.float32)

# Normalize the result to the range 0-255
result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)

# Convert back to uint8
result = np.uint8(result)

# Create a custom colormap: yellow for pressure points, blue for others
colormap = np.zeros((256, 1, 3), dtype=np.uint8)
colormap[:, 0, 0] = 255  # Blue channel
colormap[:50, 0, 1] = 255  # Green channel for lower values
colormap[:50, 0, 2] = 255  # Red channel for lower values

# Apply the custom colormap
colormapped_result = cv2.applyColorMap(result, colormap)

# Display the final image
cv2.imshow('Custom Pressure Map', colormapped_result)
cv2.waitKey(0)
cv2.destroyAllWindows()












