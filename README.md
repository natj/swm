# SWM - Shallow Water Model 

Small shallow water hydrodynamic code using spherical harmonics to compute atmospheric circulations.

In practice we solve:
![eq1](https://latex.codecogs.com/gif.latex?\frac{d&space;\vec{V}}{dt}&space;=&space;-f&space;\vec{k}&space;\times&space;\vec{V}&space;-&space;g&space;\nabla&space;h&space;&plus;&space;\nu&space;\nabla^2&space;\vec{V})

![eq2](https://latex.codecogs.com/gif.latex?\frac{d&space;h}{dt}&space;=&space;-h&space;\nabla&space;\cdot&space;\vec{V}&space;&plus;&space;\nu&space;\nabla^2&space;h)

where `V = u + v`, `f` is the Coriolis parameter, `g` is the acceleration due to gravity and `h` is the thickness of the fluid layer.


## Installation

Spherical harmonic transformation library [shtns](https://bitbucket.org/nschaeff/shtns)

```
./configure --enable.python
make
make install
```


