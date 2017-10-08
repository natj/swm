import numpy as np
from mayavi import mlab
from scipy.special import sph_harm


# Create a sphere
r = 0.3
pi = np.pi
cos = np.cos
sin = np.sin
phi, theta = np.mgrid[0:pi:101j, 0:2 * pi:101j]

x = r * sin(phi) * cos(theta)
y = r * sin(phi) * sin(theta)
z = r * cos(phi)

mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 300))
mlab.clf()

m = 4
n = 10

s = sph_harm(m, n, theta, phi).real

#print s

mlab.mesh(x, y, z, scalars=s, colormap='plasma')



#mlab.view(90, 70, 6.2, (-1.3, -2.9, 0.25))
#mlab.show()

mlab.savefig("sphere.png")
