import cv2
import numpy as np

img_path = "/Users/liamachenbach/Desktop/image_142_state_142.png"
image = cv2.imread(img_path)

image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
iamge_2 = image.astype(np.float32) / 255.0

print(image.shape)
print(image.dtype)
print(image)
print(np.max(image))
print(np.min(image))

print(iamge_2.shape)
print(iamge_2.dtype)
print(iamge_2)
print(np.max(iamge_2))
print(np.min(iamge_2))

# cv2.imshow("image", image)