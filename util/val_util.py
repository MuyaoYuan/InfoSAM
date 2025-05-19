import numpy as np
import matplotlib.pyplot as plt

def show_anns(anns, color=None):
    anns = (anns.squeeze(1).cpu().numpy() > 0.5)
    if len(anns) == 0:
        return
    
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((anns[0].shape[0], anns[0].shape[1], 4))
    img[:, :, 3] = 0
    for ann in anns:
        m = ann
       
        if color is not None:
            color_mask = np.concatenate([np.array(color[:3]), [0.35]])  
        else:
            color_mask = np.concatenate([np.random.random(3), [0.35]]) 
        img[m] = color_mask
    ax.imshow(img)
