import cv2

# Load all images (assuming they are named img1, img2, ..., img10)
images = [cv2.imread(f'Preprocessing/Concatenate_Dataset/20230626_1_set_2_1.p_frame_{i}.png') for i in range(153, 163)]

# Resize images if necessary
# ...
for img in images:
    print(img.shape)


# Initialize the result with the first image
result = images[0]

# Blend each image with the result
for img in images[1:]:
    result = cv2.addWeighted(result, 0.5, img, 0.5, 0)

# Save or display the final image
cv2.imshow('Merged Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
