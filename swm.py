# Simple spherical harmonic shallow water model toy code based on shtns library.
#
# 
# Refs:
#  "non-linear barotropically unstable shallow water test case"
#  example provided by Jeffrey Whitaker
#  https://gist.github.com/jswhit/3845307
#
#  Galewsky et al (2004, Tellus, 56A, 429-440)
#  "An initial-value problem for testing numerical models of the global
#  shallow-water equations" DOI: 10.1111/j.1600-0870.2004.00071.x
#  http://www-vortex.mcs.st-and.ac.uk/~rks/reprints/galewsky_etal_tellus_2004.pdf
#  
#  shtns/examples/shallow_water.py
#
#  Jakob-Chien et al. 1995:
#  "Spectral Transform Solutions to the Shallow Water Test Set"
#


import numpy as np
import shtns
import matplotlib.pyplot as plt
import time
from spharmt import Spharmt 
import os




##################################################
#prepare figure etc
fig = plt.figure(figsize=(8,12))
gs = plt.GridSpec(5, 10)
gs.update(hspace = 0.2)
gs.update(wspace = 0.4)

axs = []
axs.append( plt.subplot(gs[0, 0:10]) )
axs.append( plt.subplot(gs[1, 0:10]) )
axs.append( plt.subplot(gs[2, 0:10]) )
axs.append( plt.subplot(gs[3, 0:10]) )
axs.append( plt.subplot(gs[4, 0:10]) )



directory = 'out/'
if not os.path.exists(directory):
    os.makedirs(directory)



##################################################
# grid, time step info
nlons = 256              # number of longitudes
ntrunc = int(nlons/3)    # spectral truncation (to make it alias-free)
nlats = int(nlons/2)     # for gaussian grid
dt = 150                 # time step in seconds
itmax = 20*int(86400/dt) # integration length in days



# parameters for test
rsphere = 6.37122e6 # earth radius
omega = 7.292e-5    # rotation rate
grav = 9.80616      # gravity
hbar = 10.e3        # resting depth
umax = 80.          # jet speed

phi0 = np.pi/7.
phi1 = 0.5*np.pi - phi0
phi2 = 0.25*np.pi
en = np.exp(-4.0/(phi1-phi0)**2)
alpha = 1./3.
beta = 1./15.
hamp = 120.         # amplitude of height perturbation to zonal jet

efold = 3.*3600.    # efolding timescale at ntrunc for hyperdiffusion
ndiss = 8           # order for hyperdiffusion



# setup up spherical harmonic instance, set lats/lons of grid
x = Spharmt(nlons,nlats,ntrunc,rsphere,gridtype='gaussian')
lons,lats = np.meshgrid(x.lons, x.lats)
f = 2.*omega*np.sin(lats) # Coriolis

# guide grids for plotting
lons1d = (180./np.pi)*x.lons-180.
lats1d = (180./np.pi)*x.lats

#lonsDeg = (180./np.pi)*x.lons-180.
#latsDeg = (180./np.pi)*x.lats
lonsDeg = (180./np.pi)*lons-180.
latsDeg = (180./np.pi)*lats

# zonal jet
vg = np.zeros((nlats,nlons), np.float)
u1 = (umax/en)*np.exp(1./((x.lats-phi0)*(x.lats-phi1)))
ug = np.zeros((nlats),np.float)
ug = np.where(np.logical_and(x.lats < phi1, x.lats > phi0), u1, ug)
ug.shape = (nlats,1)
ug = ug*np.ones((nlats,nlons),dtype=np.float) # broadcast to shape (nlats,nlonss)
# height perturbation.
hbump = hamp*np.cos(lats)*np.exp(-(lons/alpha)**2)*np.exp(-(phi2-lats)**2/beta)



# initial vorticity, divergence in spectral space
vortSpec, divSpec =  x.getVortDivSpec(ug,vg)
vortg = x.sph2grid(vortSpec)
divg  = x.sph2grid(divSpec)


# create hyperdiffusion factor
hyperdiff_fact = np.exp((-dt/efold)*(x.lap/x.lap[-1])**(ndiss/2))



# solve nonlinear balance eqn to get initial zonal geopotential,
# add localized bump (not balanced).
vortg = x.sph2grid(vortSpec)
tmpg1 = ug*(vortg+f); tmpg2 = vg*(vortg+f)
tmpSpec1, tmpSpec2 = x.getVortDivSpec(tmpg1,tmpg2)
tmpSpec2 = x.grid2sph(0.5*(ug**2+vg**2))
phiSpec = x.invlap*tmpSpec1 - tmpSpec2
phig = grav*(hbar + hbump) + x.sph2grid(phiSpec)
phiSpec = x.grid2sph(phig)




# initialize spectral tendency arrays
ddivdtSpec  = np.zeros(vortSpec.shape+(3,), np.complex)
dvortdtSpec = np.zeros(vortSpec.shape+(3,), np.complex)
dphidtSpec  = np.zeros(vortSpec.shape+(3,), np.complex)
nnew = 0
nnow = 1
nold = 2



def visualizeMap(ax, data, vmin=0.0, vmax=1.0, title=""):

    """ 
    make a contour map plot of the incoming data array (in grid)
    """
    ax.cla()

    print title, " min/max:", data.min(), data.max()

    #colorbar
    #cb=plt.colorbar(cs,orientation='vertical') # add colorbar
    #cb.set_label('potential vorticity')

    #make fancy 
    ax.minorticks_on()
    ax.set_ylabel(title)

    #ax.set_xlabel('longitude')
    #ax.set_ylabel('latitude')

    ax.set_xticks(np.arange(-180,181,60))
    ax.set_yticks(np.linspace(-90,90,10))

    ax.pcolormesh(
            lonsDeg,
            latsDeg,
            data,
            vmin=vmin,
            vmax=vmax,
            cmap='plasma',
            )

    #ax.axis('equal')


def visualizeMapVecs(ax, xx, yy, title=""):

    """ 
    make a quiver map plot of the incoming vector field (in grid)
    """
    ax.cla()
    ax.minorticks_on()
    ax.set_ylabel(title)
    ax.set_xticks(np.arange(-180,181,60))
    ax.set_yticks(np.linspace(-90,90,10))

    M = np.hypot(xx, yy)

    print title, " min/max vec len: ", M.min(), M.max()

    sk = 10
    ax.quiver(
            lonsDeg[::sk, ::sk],
            latsDeg[::sk, ::sk],
            xx[::sk, ::sk], yy[::sk, ::sk],
            #M[::sk, ::sk],
            pivot='mid',
            units='x',
            width=1.0,
            scale=8.0,
            )

    #ax.scatter(
    #        lonsDeg[::sk, ::sk],
    #        latsDeg[::sk, ::sk],
    #        color='k',
    #        s=5,
    #          )



##################################################
# main loop
time1 = time.clock() # time loop

for ncycle in range(itmax+1):
#for ncycle in range(2): #debug option
    t = ncycle*dt

    # get vort,u,v,phi on grid
    vortg = x.sph2grid(vortSpec)
    phig  = x.sph2grid(phiSpec)
    divg  = x.sph2grid(divSpec)

    ug,vg = x.getuv(vortSpec,divSpec)

    print('t=%6.2f hours: min/max %6.2f, %6.2f' % (t/3600.,vg.min(), vg.max()))


    # compute tendencies
    tmpg1 = ug*(vortg+f)
    tmpg2 = vg*(vortg+f)
    ddivdtSpec[:,nnew], dvortdtSpec[:,nnew] = x.getVortDivSpec(tmpg1,tmpg2)
    dvortdtSpec[:,nnew] *= -1
    tmpg = x.sph2grid(ddivdtSpec[:,nnew])
    tmpg1 = ug*phig; tmpg2 = vg*phig
    tmpSpec, dphidtSpec[:,nnew] = x.getVortDivSpec(tmpg1,tmpg2)
    dphidtSpec[:,nnew] *= -1
    tmpSpec = x.grid2sph(phig+0.5*(ug**2+vg**2))
    ddivdtSpec[:,nnew] += -x.lap*tmpSpec

    # update vort,div,phiv with third-order adams-bashforth.
    # forward euler, then 2nd-order adams-bashforth time steps to start
    if ncycle == 0:
        dvortdtSpec[:,nnow] = dvortdtSpec[:,nnew]
        dvortdtSpec[:,nold] = dvortdtSpec[:,nnew]
        ddivdtSpec[:,nnow] = ddivdtSpec[:,nnew]
        ddivdtSpec[:,nold] = ddivdtSpec[:,nnew]
        dphidtSpec[:,nnow] = dphidtSpec[:,nnew]
        dphidtSpec[:,nold] = dphidtSpec[:,nnew]
    elif ncycle == 1:
        dvortdtSpec[:,nold] = dvortdtSpec[:,nnew]
        ddivdtSpec[:,nold] = ddivdtSpec[:,nnew]
        dphidtSpec[:,nold] = dphidtSpec[:,nnew]

    vortSpec += dt*( \
    (23./12.)*dvortdtSpec[:,nnew] - (16./12.)*dvortdtSpec[:,nnow]+ \
    (5./12.)*dvortdtSpec[:,nold] )

    divSpec += dt*( \
    (23./12.)*ddivdtSpec[:,nnew] - (16./12.)*ddivdtSpec[:,nnow]+ \
    (5./12.)*ddivdtSpec[:,nold] )

    phiSpec += dt*( \
    (23./12.)*dphidtSpec[:,nnew] - (16./12.)*dphidtSpec[:,nnow]+ \
    (5./12.)*dphidtSpec[:,nold] )



    # implicit hyperdiffusion for vort and div
    vortSpec *= hyperdiff_fact
    divSpec *= hyperdiff_fact


    # switch indices, do next time step
    nsav1 = nnew; nsav2 = nnow
    nnew = nold; nnow = nsav1; nold = nsav2


    #plot & save
    if (ncycle % 10 == 0):

        visualizeMap(axs[0], vortg, -1.0e-4, 1.0e-4, title="Vorticity")
        visualizeMap(axs[1], divg,  -4.0e-4, 4.0e-4, title="Divergence")
        visualizeMap(axs[2], phig,   8.0e4,  1.0e5,  title="$\\Phi$")
        visualizeMapVecs(axs[3], ug, vg, title="Velocities")


        # dimensionless potential vorticity
        pvg = (0.5*hbar*grav/omega)*(vortg+f)/phig
        print('max/min PV',pvg.min(), pvg.max())
        levs = np.arange(-0.2, 1.8, 0.1)
        visualizeMap(axs[4], pvg, -0.2, 1.8, title="Potential Vort.")





        axs[0].set_title('hour {:6.2f}'.format( t/3600.) )
        scycle = str(ncycle).rjust(6, '0')
        plt.savefig(directory+'swater'+scycle+'.png' ) #, bbox_inches='tight') 


#end of time cycle loop

time2 = time.clock()
print('CPU time = ',time2-time1)


