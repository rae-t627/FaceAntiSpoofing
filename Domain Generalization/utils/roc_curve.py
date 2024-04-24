import numpy as np

import matplotlib.pyplot as plt

# Example data for false positive rate and true positive rate
fpr = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
tpr = np.array([0.0, 0.4, 0.7, 0.85, 0.92, 1.0])

# Plot the ROC curve
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='green', lw=2, label='ROC Curve')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='LR (b=1)')
plt.plot([0, 1], [0.1, 0.1], color='red', lw=2, linestyle='--', label='LR (b=0)')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid()

# Save the ROC curve as an image file
plt.savefig('/home/eeiith/Desktop/Project1/Kaustubh/IVP/SSDG-CVPR2020/utils/roc_curve.png')