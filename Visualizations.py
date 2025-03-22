import matplotlib.pyplot as plt
import numpy as np
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import scienceplots

plt.style.use(['science', 'no-latex'])

def plotActivation(activations):
    x = np.arange(len(activations))
    y = activations

    color_values = np.abs(y)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = mcolors.Normalize(color_values.min(), color_values.max())
    lc = mcoll.LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(color_values)
    lc.set_linewidth(2)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.add_collection(lc)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min() - 0.1, y.max() + 0.1)

    ax.set_xlabel('Latent Dimension', fontsize=12)
    ax.set_ylabel('Activation (z)', fontsize=12)
    ax.set_title('Activations across Dimensions', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()

