import numpy as np
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(1, 2))
img = np.load('label_test.npy'); #random.randint(10, size=(h,w))
fig.add_subplot(1, 2, 1)
plt.imshow(img)
img = np.load('pred_test.npy'); #random.randint(10, size=(h,w))
fig.add_subplot(1, 2, 2)
plt.imshow(img)
plt.show()
