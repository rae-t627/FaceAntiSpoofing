
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image
import copy
import cv2
import os
from skimage import feature

class LocalBinaryPatterns:  # Import this class
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius
        
    def lbp_calculate(self, image):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp_image = feature.local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
        
        lbp_r = np.asarray(np.round(lbp_image)).astype(np.uint8)
        return lbp_r

def LBP(img, numPoints, radius): # Call this Function
    # LBP = LocalBinaryPatterns(8,1)
    '''The bottom one corresponds LBP operators with different parameter values, You can try them as well'''
    # LBP = LocalBinaryPatterns(8,2)
    LBP = LocalBinaryPatterns(numPoints, radius)

    lbp_img = LBP.lbp_calculate(img)

    return lbp_img


# '''Test Image'''

# img1 = np.asarray(Image.open('/home/eeiith/Desktop/Project1/Kaustubh/IVP/SSDG-CVPR2020/data_label/replay_attack_cropped/Dataset/Replay Attack/Dataset/train/real/client001_session01_webcam_authenticate_adverse_1/frame_125.0.jpg').convert('L'))
# img2 = np.asarray(Image.open('/home/eeiith/Desktop/Project1/Kaustubh/IVP/SSDG-CVPR2020/data_label/replay_attack_cropped/Dataset/Replay Attack/Dataset/train/attack/hand/attack_highdef_client001_session01_highdef_photo_adverse/frame_100.0.jpg').convert('L'))

# print(img2.shape)
# # The code snippet you provided is performing Local Binary Pattern (LBP) feature extraction on two
# # images - one real and one fake. Here's a breakdown of what the code is doing:

# numPoints = [8, 16, 24, 32, 40, 48, 56, 64]
# radii = [1, 2, 3, 4, 5]
# for numPoint in numPoints:
#     for radius in radii:
#         print(f'numPoint: {numPoint}, radius: {radius}')
#         lbp_image_real = LBP(img1, numPoint, radius)
#         lbp_image_fake = LBP(img2, numPoint, radius)

#         # plot both the images:
#         plt.figure(figsize = (20, 20))
#         plt.subplot(2,2,1)
#         plt.imshow(img1, cmap='gray')
#         plt.title('Real Image')
#         plt.axis('off')

#         plt.subplot(2,2,2)
#         plt.imshow(lbp_image_real, cmap='gray')
#         plt.title('LBP Real Image')
#         plt.axis('off')

#         plt.subplot(2,2,3)
#         plt.imshow(img2, cmap='gray')
#         plt.title('Fake Image')
#         plt.axis('off')

#         plt.subplot(2,2,4)
#         plt.imshow(lbp_image_fake, cmap='gray')
#         plt.title('LBP Fake Image')
#         plt.axis('off')

#         plt.savefig(f'./LBP_{numPoint}_{radius}.png')
#         plt.close()
#         print('Done')