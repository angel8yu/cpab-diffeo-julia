using MAT
cd(dirname(Base.source_path()))
include("../src/utils.jl")
include("../src/2d/cpab2dIntegration.jl")
include("../src/2d/imageWarp.jl")

file = matopen("tri_R2toR2_MINS_0_0_MAXS_512_512_nc_4_4_Xbdr_0_0_ext_0.mat")
As = read(file, "As")
pts_src = read(file, "pts_src")
tw_args = read(file, "tw_args")
nCols = tw_args["nCols"]
nRows = tw_args["nRows"]
close(file)

nC1 = 4
nC2 = 4
incX = nCols/nC1
incY = nRows/nC2
nPts = size(pts_src,1)
t=0.001
nSteps = 100
Nsteps = 10
Δt = t/Nsteps
ΔtAs = Δt*As
solutionMultiThread = dzeros(nPts, 2)
solutionSingleThread = Array(Float64, nPts, 2)
solutionMultiThreadShared = convert(SharedArray, solutionSingleThread)

@time solver2dParallel(pts_src,solutionMultiThread,t,As,nCols,nRows,incX,incY,Nsteps,Δt,nSteps,ΔtAs);

using PyPlot
PyPlot.figure("CPAB 2D Integration",figsize=(12,6))
PyPlot.subplot(1,2,1)
PyPlot.title("Original Image")
img = load("lena.jpg")
imr = reinterpret(UInt8, separate(img))
PyPlot.imshow(imr.data)

PyPlot.subplot(1,2,2)
PyPlot.title("Destination Image")
newimg1 = reinterpret(UInt8, separate(imwarp(convert(Array, solutionMultiThread), img, nCols, nRows)))
PyPlot.imshow(newimg1.data)