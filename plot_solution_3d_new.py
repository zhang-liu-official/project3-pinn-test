from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

###################

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    #%
    cdict = {'red': [], 'green': [], 'blue': []}

    # make a lin_space with the number of records from seq.     
    x = np.linspace(0,1, len(seq))
    #%
    for i in range(len(seq)):
        segment = x[i]
        tone = seq[i]
        cdict['red'].append([segment, tone, tone])
        cdict['green'].append([segment, tone, tone])
        cdict['blue'].append([segment, tone, tone])
    #%
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

###################

# domains
# test data.
data = np.genfromtxt("test.dat", delimiter=' ')
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]
true = data[:,3]
pred = data[:,4]

colors = make_colormap(pred)

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_trisurf(x, y, z, cmap=colors, linewidth=0.2)

plt.show()