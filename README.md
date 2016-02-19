Julia Implementation of CPAB transformations
============================================

This is a Julia implementation of CPAB transformations based on [\[Freifeld et al., ICCV '15\] ](http://people.csail.mit.edu/freifeld/papers/freifeld_ICCV_2015.pdf). For a the original CUDA/Python implementation (with more options than currently appear here), please see https://github.com/freifeld/cpabDiffeo.git

This software is released under the MIT License (included with the software). Note, however, that using this code (and/or the results of running it) to support any form of publication (e.g.,a book, a journal paper, a conference paper, a patent application, etc.) requires you to cite the following paper:

```
@inproceedings{freifeld2015transform,
    title = {Highly-Expressive Spaces of Well-Behaved Transformations: Keeping It Simple},
    author = {Oren Freifeld and S{\o}ren Hauberg and Kayhan Batmanghelich and John W. Fisher III},
    booktitle = {International Conference on Computer Vision (ICCV)},
    address = {Santiago, Chile},
    month = Dec,
    year = {2015}
}
```

Author
------

Angel Yu (email: angelyu@mit.edu)

Requirements
------------

Julia 0.4.2
Julia Packages: DistributedArrays, Distributions, Images, PyPlot, ImageMagick, MAT (only to run the example scripts)
To install these packages run the following commands in a Julia terminal:
```
Pkg.add("DistributedArrays")
Pkg.add("Distributions")
Pkg.add("Images")
Pkg.add("PyPlot")
Pkg.add("ImageMagick")
Pkg.add("MAT")
Pkg.update()
```

Instructions
------------

There are a few demo scripts under "examples" which illustrates the use of this implementation. To run the demos, you would first need to navigate to the "examples" directory and open a Julia terminal there.

To run the demo script showing integration of a 1D CPAB transformation in serial, run the following command:
```
include("demoIntegrate1DSerial.jl")
```
The image that results from running this script is located in "example/result_images/1d_integration.png"

To run the demo script showing integration of a 1D CPAB transformation in parallel using, say, 2 CPU cores (you can use as many cores as your machine has):
```
addprocs(2) # 2 processes
include("demoIntegrate1DParallel.jl")
```

To run the demo script showing integration of a 2D CPAB transformation in serial:
```
include("demoIntegrate2DSerial.jl")
```
The image that results from running this script is located in "example/result_images/2d_integration.png"

To run the demo script showing integration of a 2D CPAB transformation in parallel:
```
addprocs(2) # 2 processes
include("demoIntegrate2DParallel.jl")
```

To run the demo script showing landmark warping inference using Metropolis' Algorithm:
```
include("demoMetropolis2D.jl")
```
The image that results from running this script is located in "example/result_images/metropolis.png"

To run the demo script showing landmark warping inference using Particle Filter in serial:
```
include("demoParticleFilter2DSerial.jl")
```
The image that results from running this script is located in "example/result_images/particle_filter.png"

To run the demo script showing landmark warping inference using Particle Filter in parallel:
```
include("demoParticleFilter2DParallel.jl")
```
