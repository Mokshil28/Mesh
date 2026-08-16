import numpy as np
import matplotlib.pyplot as plt

input_numpy = np.load('/data/sxu7/heatmaps/recorded/heatmaps/train/range/all/bow9.npy')
print (f"\n{input_numpy.shape} {type(input_numpy)} \n {input_numpy}")

plt.figure()
plt.imshow(input_numpy)
# plt.axis('off')
plt.gca().set_frame_on(False)
plt.savefig(
    '/data/sxu7/heatmaps/range.png', 
    # bbox_inches='tight', 
    # pad_inches=0
    )
print(f"\nInput numpy size: {input_numpy.shape}")
