using MAT
cd(dirname(Base.source_path()))
include("../src/utils.jl")
include("../src/1d/cpab1dIntegration.jl")
file = matopen("cpa_data_nCells_5.mat")
As = read(file, "As")
pts_src = read(file, "pts_src")
pts_src = squeeze(pts_src,2)
tw_args = read(file, "tw_args")
nCols = tw_args["nCols"]
nCells = size(As, 1)
close(file)

nSteps = 100
Nsteps = 10
tess = getTess(As, nCells, nCols)
v = getVelocityField1d(pts_src, As, nCells, nCols)

pts_fwd1 = solver1dParallelIntegration(pts_src, 0.1, tess, As, nCells, nCols, Nsteps, nSteps)
pts_fwd2 = solver1dParallelIntegration(pts_src, 0.5, tess, As, nCells, nCols, Nsteps, nSteps)
@time pts_fwd3 = solver1dParallelIntegration(pts_src, 1.0, tess, As, nCells, nCols, Nsteps, nSteps)
pts_fwd4 = solver1dParallelIntegration(pts_src, 2.0, tess, As, nCells, nCols, Nsteps, nSteps)
pts_fwd5 = solver1dParallelIntegration(pts_src, 8.0, tess, As, nCells, nCols, Nsteps, nSteps)

using PyPlot
PyPlot.figure("1D Integration",figsize=(12,6))
PyPlot.subplot(1,2,1)
PyPlot.title("Velocity Field")
PyPlot.plot(pts_src, v, color="blue")
PyPlot.xlabel("x")
PyPlot.ylabel("v")

PyPlot.subplot(1,2,2)
PyPlot.title("1D Integration for different t")
PyPlot.plot(pts_src, pts_src, label="t=0", color="black")
PyPlot.plot(pts_src, pts_fwd1, label="t=0.1", color="red")
PyPlot.plot(pts_src, pts_fwd2, label="t=0.5", color="orange")
PyPlot.plot(pts_src, pts_fwd3, label="t=1.0", color="yellow")
PyPlot.plot(pts_src, pts_fwd4, label="t=2.0", color="green")
PyPlot.plot(pts_src, pts_fwd5, label="t=8.0", color="blue")
PyPlot.xlabel("x")
PyPlot.ylabel("Transformed x")
PyPlot.legend(loc="lower right",fancybox="true")