import matplotlib.pyplot as plt
import numpy as np

def plot_heatmap(name,weight,file_path):
    weight = weight.cpu().detach().numpy()
    weight2d = weight.reshape(weight.shape[0], -1)
    im = plt.matshow(np.abs(weight2d), cmap=plt.cm.BuPu, aspect='auto')
    plt.colorbar(im)
    plt.title(name)
    plt.savefig(file_path + name + '.jpg', dpi=800)
    plt.show()