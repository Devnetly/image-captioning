import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = plt.imread('/home/abdelnour/Documents/4eme_anne/S2/NLP/projet/image-captioning/docs/figures/architecture.png')

img[img[:,:,-1] == 0] = np.array([14,14,14,1])

h_padding = np.zeros((100,img.shape[1],4))
h_padding[:,:,0] = 14
h_padding[:,:,1] = 14 
h_padding[:,:,2] = 14
h_padding[:,:,3] = 1

img = np.concatenate([h_padding,img,h_padding], axis=0)

v_padding = np.zeros((img.shape[0],100,4))
v_padding[:,:,0] = 14
v_padding[:,:,1] = 14
v_padding[:,:,2] = 14
v_padding[:,:,3] = 1

img = np.concatenate([v_padding,img,v_padding], axis=1)

print(np.unique(img.flatten()))

# Image.fromarray(img).save('/home/abdelnour/Documents/4eme_anne/S2/NLP/projet/image-captioning/docs/figures/architecture-with-bg.png')

img = plt.imshow(img)
img.set_cmap('hot')
plt.axis('off')
plt.savefig('/home/abdelnour/Documents/4eme_anne/S2/NLP/projet/image-captioning/docs/figures/architecture-with-bg.png', bbox_inches='tight',  dpi=1200)