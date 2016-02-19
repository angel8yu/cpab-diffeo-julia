println("Loading packages and data, this may take some time...")
using MAT
cd(dirname(Base.source_path()))
include("../src/utils.jl")
include("../src/2d/cpab2dIntegration.jl")
include("../src/2d/imageWarp.jl")
include("../src/2d/2dInference.jl")

file = matopen("tri_R2toR2_MINS_0_0_MAXS_512_512_nc_4_4_Xbdr_0_0_ext_0.mat")
As = read(file, "As")
@broadcast :As As
pts_src = read(file, "pts_src")
@broadcast :pts_src pts_src
pts_fwd = read(file, "pts_fwd")
@broadcast :pts_fwd pts_fwd
tw_args = read(file, "tw_args")
@broadcast :tw_args tw_args
@everywhere nCols = tw_args["nCols"]
@everywhere nRows = tw_args["nRows"]
B = read(file, "B")
@broadcast :B B
θ = vec(read(file, "theta"))
@broadcast :θ θ
priorCov = read(file, "covariance_of_the_prior")
@broadcast :priorCov priorCov
close(file)

@everywhere nC1 = 4
@everywhere nC2 = 4
@everywhere incX = nCols/nC1
@everywhere incY = nRows/nC2
@everywhere nPts = size(pts_src,1)
@everywhere t=0.001
@everywhere nSteps = 100
@everywhere Nsteps = 10
@everywhere Δt = t/Nsteps
@everywhere ΔtAs = Δt*As

@everywhere function decreaseNumPts(A)
  reshape(reshape(A,512,512,2)[16:32:end,16:32:end,:],256,2)
end
@everywhere pts_src1 = decreaseNumPts(pts_src)
@everywhere pts_fwd1 = decreaseNumPts(pts_fwd)
@everywhere solutionSingleThread1 = zeros(size(pts_src1,1), 2)
@everywhere expT = Array(Float64, size(As,1), 3, 3)
@everywhere Avees = Array(Float64, size(θ,1))

@everywhere function leastSquaresTransformation(θ)
  computeAndEvaluateTransformationSingleThread(B,θ,As,pts_src1,pts_fwd1,solutionSingleThread1,t,nCols,nRows,incX,incY,Nsteps,Δt,nSteps,ΔtAs,expT,Avees)
end

iterations = 200
using PyPlot
println("Running Particle Filter, this may take some time...")
@time optimizedθ, llTrace, lpTrace, lpllTrace = particleFilterParallel(leastSquaresTransformation, rand(size(θ,1)), 1000, iterations, 10.0, priorCov, 0.005*priorCov, false, true)
println("Done!")
optimizedθArray = convert(Array, optimizedθ)

PyPlot.figure("Particle Filter",figsize=(18,6))
PyPlot.subplot(2,6,1)
PyPlot.title("Original Points")
xOriginal, yOriginal = separateAxis2D(pts_src1)
PyPlot.axis([0.0,512.0,0.0,512.0])
PyPlot.scatter(xOriginal, yOriginal, label="Original Points", color="blue")

PyPlot.subplot(2,6,2)
PyPlot.title("Destination Points")
PyPlot.axis([0.0,512.0,0.0,512.0])
xDst, yDst = separateAxis2D(pts_fwd1)
PyPlot.scatter(xDst, yDst, label="Destination Points", color="blue")

PyPlot.subplot(2,6,3)
PyPlot.title("Transformed Points")
PyPlot.axis([0.0,512.0,0.0,512.0])
leastSquaresTransformation(optimizedθArray)
xInferred, yInferred = separateAxis2D(solutionSingleThread1)
PyPlot.scatter(xDst, yDst, label="Destination Points", color="blue")
PyPlot.scatter(xInferred, yInferred, label="Transformed Points", color="red")

PyPlot.subplot(2,6,7)
PyPlot.title("Original Image")
img = load("lena.jpg")
imr = reinterpret(UInt8, separate(img))
PyPlot.imshow(imr.data)

PyPlot.subplot(2,6,8)
PyPlot.title("Destination Image")
newimg1 = reinterpret(UInt8, separate(imwarp(pts_fwd, img, nCols, nRows)))
PyPlot.imshow(newimg1.data)

PyPlot.subplot(2,6,9)
PyPlot.title("Transformed Image")
transformedAs = similar(As)
solutionSingleThread = similar(pts_src)
computeAs(B,optimizedθArray,transformedAs)
solver2d(pts_src,solutionSingleThread,t,transformedAs,nCols,nRows,incX,incY,Nsteps,Δt,nSteps,ΔtAs)
newimg2 = reinterpret(UInt8, separate(imwarp(solutionSingleThread, img, nCols, nRows)))
PyPlot.imshow(newimg2.data)

PyPlot.subplot(1,2,2)
PyPlot.title("Log Probability vs Iterations")
PyPlot.plot(1:iterations, llTrace,label="log likelihood", color="blue")
PyPlot.plot(1:iterations, lpTrace,label="log prior", color="red")
PyPlot.plot(1:iterations, lpllTrace,label="log likelihood + log prior", color="green")
PyPlot.xlabel("Iterations")
PyPlot.ylabel("Log Probability")
PyPlot.legend(loc="lower right",fancybox="true")