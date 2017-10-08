# SWM - Shallow Water Model 

Small shallow water hydrodynamic code using spherical harmonics to compute atmospheric circulations.

In practice we solve:

![eq1](https://latex.codecogs.com/gif.latex?\frac{d&space;\vec{V}}{dt}&space;=&space;-f&space;\vec{k}&space;\times&space;\vec{V}&space;-&space;g&space;\nabla&space;h&space;&plus;&space;\nu&space;\nabla^2&space;\vec{V})

![eq2](https://latex.codecogs.com/gif.latex?\frac{d&space;h}{dt}&space;=&space;-h&space;\nabla&space;\cdot&space;\vec{V}&space;&plus;&space;\nu&space;\nabla^2&space;h)

where `V = u + v`, `u`/`v` are the eastward/northwards velocities `f` is the Coriolis parameter, `g` is the acceleration due to gravity and `h` is the thickness of the fluid layer.


## Installation

Spherical harmonic transformation library [shtns](https://bitbucket.org/nschaeff/shtns)

```
./configure --enable-python
make
make install
```


## References

*  "non-linear barotropically unstable shallow water test case"
  example provided by Jeffrey Whitaker
  https://gist.github.com/jswhit/3845307

*  Galewsky et al (2004, Tellus, 56A, 429-440)
  "An initial-value problem for testing numerical models of the global
  shallow-water equations" DOI: 10.1111/j.1600-0870.2004.00071.x
  http://www-vortex.mcs.st-and.ac.uk/~rks/reprints/galewsky_etal_tellus_2004.pdf
  
*  shtns/examples/shallow_water.py

*  Jakob-Chien et al. 1995:
  "Spectral Transform Solutions to the Shallow Water Test Set"

