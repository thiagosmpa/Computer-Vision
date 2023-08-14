"""
This code is used only for plotting the inference image, to compare with the original image and mask
"""

import matplotlib.pyplot as plt
import cv2
import os

image = '08_test_0.png'
original_path = 'new_dataset/test/image/'
mask_path = 'new_dataset/test/mask/'
pred_path = 'results/'

original_image = os.path.join(original_path, image)
mask_path = os.path.join(mask_path, image)
pred_path = os.path.join(pred_path, image)

original = cv2.imread(original_image)
mask = cv2.imread(mask_path)
pred = cv2.imread(pred_path)

plt.subplot(1, 3, 1)
plt.axis(False)
plt.title('Original Image')
original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
plt.imshow(original)

plt.subplot(1, 3, 2)
plt.axis(False)
plt.title('Original Mask')
plt.imshow(mask)

plt.subplot(1, 3, 3)
plt.axis(False)
plt.title('Predicted Mask')
plt.imshow(pred)

plt.show()
