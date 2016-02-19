@everywhere using Distributions
include("../utils.jl")

# compute As given Basis B and Parameters θ and store the results in As
@everywhere function computeAs(
    B::Matrix{Float64},
    θ::Array{Float64,1},
    As::Array{Float64,3},
    Avees::Array{Float64,1} = Array(Float64, size(θ,1))
  )

  Avees = B*θ
  n::Int64 = length(Avees)
  nCells::Int64 = round(Int, n/6)
  for i = 1:nCells
    As[i,1,1] = Avees[6*(i-1)+1]
    As[i,1,2] = Avees[6*(i-1)+2]
    As[i,1,3] = Avees[6*(i-1)+3]
    As[i,2,1] = Avees[6*(i-1)+4]
    As[i,2,2] = Avees[6*(i-1)+5]
    As[i,2,3] = Avees[6*(i-1)+6]
  end
end

@everywhere function computeAndEvaluateTransformationSingleThread(
    B::Matrix{Float64},
    θ::Array{Float64,1},
    As::Array{Float64,3},
    X::Array{Float64,2},
    Y::Array{Float64,2},
    Tx::Array{Float64,2},
    t::Float64,
    nCols::Int64,
    nRows::Int64,
    incX::Float64,
    incY::Float64,
    Nsteps::Int64,
    Δt::Float64,
    nSteps::Int64,
    ΔtAs::Array{Float64,3},
    expT::Array{Float64,3} = Array(Float64, size(As,1), 3, 3),
    Avees::Array{Float64,1} = Array(Float64, size(θ,1)),
    bdryConstraints::Bool = false,
  )
  computeAs(B,θ,As,Avees)
  solver2d(X,Tx,t,As,nCols,nRows,incX,incY,Nsteps,Δt,nSteps,ΔtAs,bdryConstraints,expT)

  nPts = size(X,1)
  leastSquares = 0
  for i=1:nPts
    leastSquares += (Tx[i,1]-Y[i,1])^2 + (Tx[i,2]-Y[i,2])^2
  end
  leastSquares
end

@everywhere function computeAndEvaluateTransformationParallel(
    B::Matrix{Float64},
    θ::Array{Float64,1},
    As::Array{Float64,3},
    X::Array{Float64,2},
    Y::Array{Float64,2},
    Tx::DArray{Float64,2},
    t::Float64,
    nCols::Int64,
    nRows::Int64,
    incX::Float64,
    incY::Float64,
    Nsteps::Int64,
    Δt::Float64,
    nSteps::Int64,
    ΔtAs::Array{Float64,3},
    expT::Array{Float64,3} = Array(Float64, size(As,1), 3, 3),
    Avees::Array{Float64,1} = Array(Float64, size(θ,1)),
    bdryConstraints::Bool = false,
  )
  computeAs(B,θ,As,Avees)
  solver2dParallel(X,Tx,t,As,nCols,nRows,incX,incY,Nsteps,Δt,nSteps,ΔtAs,bdryConstraints,expT)


  threads::Array{Int64,1} = workers()
  nThreads::Int64 = length(threads)
  refs::Array{RemoteRef,1} = Array(RemoteRef, nThreads)
  partialLeastSquares::Array{Float64,1} = Array(Float64, nThreads)
  for index::Int64=1:nThreads
    p = threads[index]
    refs[index] = @spawnat(p, (
      indicesOnThread::Array{Int64,1} = localindexes(Tx)[1];
      localArray = localpart(Tx);
      currLeastSquares::Float64 = 0.0;
      for i::Int64 = 1:length(indicesOnThread);
        j = indicesOnThread[i];
        currLeastSquares += (localArray[i,1]-Y[j,1])^2 + (localArray[i,2]-Y[j,2])^2;
      end;
      currLeastSquares
      ))
  end

  # wait for the workers to finish and sum
  map(wait, refs)
  leastSquares::Float64 = 0.0
  for index::Int64=1:nThreads
    leastSquares += fetch(refs[index])
  end
  leastSquares
end

@everywhere function computeAndEvaluateTransformationParallelShared(
    B::Matrix{Float64},
    θ::Array{Float64,1},
    As::Array{Float64,3},
    X::Array{Float64,2},
    Y::Array{Float64,2},
    Tx::SharedArray{Float64,2},
    t::Float64,
    nCols::Int64,
    nRows::Int64,
    incX::Float64,
    incY::Float64,
    Nsteps::Int64,
    Δt::Float64,
    nSteps::Int64,
    ΔtAs::Array{Float64,3},
    expT::Array{Float64,3} = Array(Float64, size(As,1), 3, 3),
    Avees::Array{Float64,1} = Array(Float64, size(θ,1)),
    bdryConstraints::Bool = false,
  )
  computeAs(B,θ,As,Avees)
  solver2dParallelShared(X,Tx,t,As,nCols,nRows,incX,incY,Nsteps,Δt,nSteps,ΔtAs,bdryConstraints,expT)
  nPts = size(X,1)
  leastSquares = 0
  for i=1:nPts
    leastSquares += (Tx[i,1]-Y[i,1])^2 + (Tx[i,2]-Y[i,2])^2
  end
  leastSquares
end

@everywhere function calculateLogProbGaussian(
    θ::Array{Float64,1},
    θmean::Array{Float64,1},
    invCov::Array{Float64,2},
  )
  -0.5*(θ-θmean)'*invCov*(θ-θmean)
end

function metropolis(
    f::Function,
    θ::Array{Float64,1},
    iterations::Int64,
    σ::Float64,
    priorCov::Array{Float64,2},
    jumpCov::Array{Float64,2},
    storeTrace::Bool = false,
  )
  dim::Int64 = size(θ,1)
  invPriorCov::Array{Float64,2} = inv(priorCov)
  priorDist::Distributions.MvNormal = MvNormal(zeros(dim), priorCov)
  jumpDist::Distributions.MvNormal = MvNormal(zeros(dim), jumpCov)
  prevθ::Array{Float64,1} = copy(θ)
  newθ::Array{Float64,1} = Array(Float64, dim)
  prevVal::Float64 = -0.5*f(θ)/(σ^2)
  currVal::Float64 = 0.0
  prevPrior::Float64 = calculateLogProbGaussian(θ, zeros(dim), invPriorCov)[1]  #log(pdf(priorDist,θ))
  currPrior::Float64 = prevPrior
  maxVal::Float64 = prevVal
  bestθ::Array{Float64,1} = copy(θ)
  accepted::Int64 = 0

  if storeTrace
    llTrace::Array{Float64,1} = Array(Float64, iterations)
    acceptanceTrace::Array{Float64,1} = Array(Float64, iterations)
    lpTrace::Array{Float64,1} = Array(Float64, iterations)
    lpllTrace::Array{Float64,1} = Array(Float64, iterations)
  end

  for i=1:iterations
    if i%1000 == 0
      println("Running iteration $i out of $iterations")
    end
    acceptanceRate = accepted/i
    newθ = rand(jumpDist) + prevθ
    currVal = -0.5*f(newθ)/(σ^2)
    currPrior = calculateLogProbGaussian(newθ, zeros(dim), invPriorCov)[1] # log(pdf(priorDist,newθ))
    logRatio = (currVal-prevVal) + (currPrior-prevPrior)
    if logRatio>0 || logRatio < log(rand())
      accepted += 1
      prevθ = newθ
      prevVal = currVal
      prevPrior = currPrior
      if currVal > maxVal
        maxVal = currVal
        bestθ = newθ
      end
    end
    if storeTrace
      llTrace[i] = currVal
      lpTrace[i] = currPrior
      acceptanceTrace[i] = acceptanceRate
      lpllTrace[i] = currVal+currPrior
    end
  end

  if storeTrace
    return bestθ, acceptanceTrace, llTrace, lpTrace, lpllTrace
  end
  bestθ
end

function particleFilter(
    f::Function, # likelihood
    θ::Array{Float64,1},
    nParticles::Int64,
    iterations::Int64,
    σ::Float64,
    priorCov::Array{Float64,2},
    jumpCov::Array{Float64,2},
    plotWeights::Bool = false,
    storeTrace::Bool = false,
  )
  dim::Int64 = size(θ,1)
  invPriorCov::Array{Float64,2} = inv(priorCov)
  priorDist::Distributions.MvNormal = MvNormal(zeros(dim), priorCov)
  jumpDist::Distributions.MvNormal = MvNormal(zeros(dim), jumpCov)
  prevSamples::Array{Float64, 2} = rand(priorDist, nParticles)'
  currSamples::Array{Float64, 2} = Array(Float64, nParticles, dim)
  newθ::Array{Float64, 1} = Array(Float64, dim)
  weights::Array{Float64,1} = Array(Float64, nParticles)
  cumSumWeights::Array{Float64,1} = Array(Float64, nParticles)
  perturbation::Array{Float64,1} = Array(Float64, dim)
  sumWeights::Float64 = 0.0
  maxVal::Float64 = -Inf
  bestθ::Array{Float64,1} = θ

  if storeTrace
    llTrace::Array{Float64,1} = zeros(iterations)
    lpTrace::Array{Float64,1} = zeros(iterations)
    lpllTrace::Array{Float64,1} = zeros(iterations)
  end

  for i = 1:iterations
    if i%10 == 0
      println("Running iteration $i out of $iterations")
    end
    # update weights
    for j = 1:nParticles
      for k = 1:dim
        newθ[k] = prevSamples[j,k] # avoiding squeeze
      end
      logLikelihood::Float64 = -0.5*f(newθ)/(σ^2)
      logPrior::Float64 = calculateLogProbGaussian(newθ, zeros(dim), invPriorCov)[1]
      weights[j] = logLikelihood + logPrior

      if weights[j] > maxVal
        maxVal = weights[j]
        bestθ = newθ
        if storeTrace
          llTrace[i] = logLikelihood
          lpTrace[i] = logPrior
          lpllTrace[i] = logLikelihood + logPrior
        end
      end
    end

    if storeTrace
      if lpllTrace[i] == 0 && i > 1
        llTrace[i] = llTrace[i-1]
        lpTrace[i] = lpTrace[i-1]
        lpllTrace[i] = lpllTrace[i-1]
      end
    end

    maxWeight = weights[1]
    for j = 1:nParticles
      if weights[j] != Inf && weights[j] > maxWeight
        maxWeight = weights[j]
      end
    end

    weights -= maxWeight
    sumWeights = 0.0
    for j = 1:nParticles
      weights[j] = e^weights[j]
      sumWeights += weights[j]
    end

    weights /= sumWeights
    if plotWeights
      PyPlot.clf()
      PyPlot.plot(1:nParticles, cumsum(sort(weights)),label="dist", color="blue")
      savefig(string("plot", i, ".svg"))
    end

    # resample
    cumSumWeights = cumsum(weights)
    for j = 1:nParticles
      cumWeightSample::Float64 = rand()
      for k = 1:nParticles
        if cumSumWeights[k] >= cumWeightSample
          currSamples[j,:] = prevSamples[k,:]
          break
        end
      end
    end

    # perturb
    for j = 1:nParticles
      perturbation = rand(jumpDist)
      for k = 1:dim
        prevSamples[j,k] = perturbation[k] + currSamples[j,k]
      end
    end
  end

  if storeTrace
    return bestθ, llTrace, lpTrace, lpllTrace
  end
  bestθ
end


function particleFilterParallel(
    f::Function, # likelihood
    θ::Array{Float64,1},
    nParticles::Int64,
    iterations::Int64,
    σ::Float64,
    priorCov::Array{Float64,2},
    jumpCov::Array{Float64,2},
    plotWeights::Bool = false,
    storeTrace::Bool = false,
  )
  dim::Int64 = size(θ,1)
  invPriorCov::Array{Float64,2} = inv(priorCov)
  priorDist::Distributions.MvNormal = MvNormal(zeros(dim), priorCov)
  jumpDist::Distributions.MvNormal = MvNormal(zeros(dim), jumpCov)
  prevSamples::SharedArray{Float64, 2} = convert(SharedArray, rand(priorDist, nParticles)')
  currSamples::SharedArray{Float64, 2} = SharedArray(Float64, nParticles, dim)
  weights::SharedArray{Float64,1} = SharedArray(Float64, nParticles)
  cumSumWeights::Array{Float64,1} = Array(Float64, nParticles)
  perturbation::Array{Float64,1} = Array(Float64, dim)
  sumWeights::Float64 = 0.0
  maxVal::SharedArray{Float64,1} = convert(SharedArray, [-Inf])
  bestθ::SharedArray{Float64,1} = convert(SharedArray, θ)
  llTrace::SharedArray{Float64,1} = SharedArray(Float64, iterations)
  lpTrace::SharedArray{Float64,1} = SharedArray(Float64, iterations)
  lpllTrace::SharedArray{Float64,1} = SharedArray(Float64, iterations)

  threads::Array{Int64,1} = workers()
  nThreads::Int64 = length(threads)
  refs::Array{RemoteRef,1} = Array(RemoteRef, nThreads)

  for i = 1:iterations
    if i%10 == 0
      println("Running iteration $i out of $iterations")
    end
    # update weights in parallel
    for index::Int64=1:nThreads
      p = threads[index]
      refs[index] = @spawnat(p, (
        newθ::Array{Float64, 1} = Array(Float64, dim);
        logPrior::Float64 = 0.0;
        logLikelihood::Float64 = 0.0;
        for j::Int64 = myrange(prevSamples);
          for k = 1:dim;
            newθ[k] = prevSamples[j,k]; # avoiding squeeze
          end;
          logLikelihood = -0.5*f(newθ)/(σ^2);
          logPrior = calculateLogProbGaussian(newθ, zeros(dim), invPriorCov)[1];
          weights[j] = logLikelihood + logPrior;

          if weights[j] > maxVal[1];
            maxVal[1] = weights[j];
            for k = 1:dim;
              bestθ[k] = newθ[k];
            end;
            if storeTrace
              llTrace[i] = logLikelihood
              lpTrace[i] = logPrior
              lpllTrace[i] = logLikelihood + logPrior
            end
          end;
        end;
      ))
    end

    # wait for the workers to finish
    map(wait, refs)

    if storeTrace
      if llTrace[i] == 0 && i > 1
        llTrace[i] = llTrace[i-1]
        lpTrace[i] = lpTrace[i-1]
        lpllTrace[i] = lpllTrace[i-1]
      end
    end

    maxWeight = weights[1]
    for j = 1:nParticles
      if weights[j] != Inf && weights[j] > maxWeight
        maxWeight = weights[j]
      end
    end
    weights -= maxWeight

    sumWeights = 0.0
    for j = 1:nParticles
      weights[j] = e^weights[j]
      sumWeights += weights[j]
    end
    weights /= sumWeights
    if plotWeights
      PyPlot.clf()
      PyPlot.plot(1:nParticles, cumsum(sort(weights)),label="dist", color="blue")
      savefig(string("plot", i, ".svg"))
    end

    # resample
    cumSumWeights = cumsum(weights)
    for j = 1:nParticles
      cumWeightSample::Float64 = rand()
      for k = 1:nParticles
        if cumSumWeights[k] >= cumWeightSample
          currSamples[j,:] = prevSamples[k,:]
          break
        end
      end
    end

    # perturb
    for j = 1:nParticles
      perturbation = rand(jumpDist)
      for k = 1:dim
        prevSamples[j,k] = perturbation[k] + currSamples[j,k]
      end
    end
  end
  if storeTrace
    return bestθ, llTrace, lpTrace, lpllTrace
  end
  bestθ
end

function particleFilterDistributed(
    f::Function, # likelihood
    θ::Array{Float64,1},
    nParticles::Int64,
    iterations::Int64,
    σ::Float64,
    priorCov::Array{Float64,2},
    jumpCov::Array{Float64,2},
    plotWeights::Bool = false,
  )
  dim::Int64 = size(θ,1)
  invPriorCov::Array{Float64,2} = inv(priorCov)
  priorDist::Distributions.MvNormal = MvNormal(zeros(dim), priorCov)
  jumpDist::Distributions.MvNormal = MvNormal(zeros(dim), jumpCov)
  @broadcast :jumpDist jumpDist
  prevSamples::Array{Float64, 2} = Array(Float64, nParticles, dim)
  currSamples::Array{Float64, 2} = rand(priorDist, nParticles)'
  distributedWeights::DArray{Float64,1} = dzeros((nParticles,), workers(), [length(workers()),])
  weights::Array{Float64,1} = Array(Float64, nParticles)
  cumSumWeights::Array{Float64,1} = Array(Float64, nParticles)
  perturbation::Array{Float64,1} = Array(Float64, dim)
  sumWeights::Float64 = 0.0
  maxVal::Float64 = -Inf
  bestθ::Array{Float64,1} = θ
  threads::Array{Int64,1} = workers()
  nThreads::Int64 = length(threads)
  refs::Array{RemoteRef,1} = Array(RemoteRef, nThreads)
  prevSamplesDistributed::DArray{Float64, 2} = dzeros((nParticles, dim), workers(), [length(workers()), 1])
  for index::Int64=1:nThreads # initialize all the threads
    p = threads[index]
    refs[index] = @spawnat(p, (
      indicesOnThread::Array{Int64,1} = localindexes(prevSamplesDistributed)[1];
      localPrevSamples = localpart(prevSamplesDistributed);
      localCurrSamples = copy(currSamples);
      for j::Int64 = 1:length(indicesOnThread);
        for k = 1:dim;
          localPrevSamples[j,k] = currSamples[j,k];
        end;
      end;
    ))
  end
  map(wait, refs)

  for i = 1:iterations
    if i%10 == 0
      println("Running iteration $i out of $iterations")
    end
    # update weights in parallel
    for index::Int64=1:nThreads
      p = threads[index]
      refs[index] = @spawnat(p, (
        indicesOnThread::Array{Int64,1} = localindexes(prevSamplesDistributed)[1];
        localPrevSamples = localpart(prevSamplesDistributed);
        localWeights = localpart(distributedWeights);
        newθ::Array{Float64, 1} = Array(Float64, dim);
        logPrior::Float64 = 0.0;
        maxWeightThread::Float64 = -Inf;
        maxValThread::Float64 = copy(maxVal);
        bestθThread::Array{Float64,1} = Array(Float64,dim);
        logLikelihood::Float64 = 0.0;
        for j::Int64 = 1:length(indicesOnThread);
          for k = 1:dim;
            newθ[k] = localPrevSamples[j,k]; # avoiding squeeze
          end;
          logLikelihood = -0.5*f(newθ)/(σ^2);
          logPrior = calculateLogProbGaussian(newθ, zeros(dim), invPriorCov)[1];
          localWeights[j] = logLikelihood + logPrior;
          if localWeights[j] > maxWeightThread;
            maxWeightThread = localWeights[j];
          end
          if localWeights[j] > maxValThread;
            maxValThread = localWeights[j];
            for k = 1:dim;
              bestθThread[k] = newθ[k];
            end;
          end;
        end;
        return (maxValThread, bestθThread, maxWeightThread);
      ))
    end

    #wait for the workers to finish
    #find the max weight as well as update best θ
    maxWeight::Float64 = -Inf
    for index::Int64=1:nThreads
      maxValThread1, bestθThread1, maxWeightThread1 = fetch(refs[index])
      if maxWeightThread1 > maxWeight
        maxWeight = maxWeightThread1
      end
      if maxValThread1 > maxVal
        maxVal = maxValThread1
        for k = 1:dim
          bestθ[k] = bestθThread1[k]
        end        
      end
    end

    #wait exponentiate and find sum of the weights
    for index::Int64=1:nThreads
      p = threads[index]
      refs[index] = @spawnat(p, (
        indicesOnThread::Array{Int64,1} = localindexes(distributedWeights)[1];
        localWeights = localpart(distributedWeights);
        sumWeightsThread::Float64 = 0.0;
        for j::Int64 = 1:length(indicesOnThread);
          localWeights[j] -= maxWeight;
          localWeights[j] = e^localWeights[j];
          sumWeightsThread += localWeights[j];
        end;
        return sumWeightsThread;
      ))
    end
    sumWeights = 0.0
    for index::Int64=1:nThreads
      sumWeights += fetch(refs[index])
    end

    #plot weights
    weights = convert(Array, distributedWeights)
    if plotWeights
      PyPlot.clf()
      PyPlot.plot(1:nParticles, cumsum(sort(weights)),label="dist", color="blue")
      savefig(string("plot", i, ".svg"))
    end

    # resample
    cumSumWeights = cumsum(weights)
    prevSamples = convert(Array, prevSamplesDistributed)
    for j = 1:nParticles
      cumWeightSample::Float64 = sumWeights*rand()
      for k = 1:nParticles
        if cumSumWeights[k] >= cumWeightSample
          currSamples[j,:] = prevSamples[k,:]
          break
        end
      end
    end

    # perturb
    for index::Int64=1:nThreads
      p = threads[index]
      refs[index] = @spawnat(p, (
        indicesOnThread::Array{Int64,1} = localindexes(prevSamplesDistributed)[1];
        localPrevSamples = localpart(prevSamplesDistributed);
        localCurrSamples = copy(currSamples);
        for j::Int64 = 1:length(indicesOnThread);
          perturbation = rand(jumpDist);
          for k = 1:dim;
            localPrevSamples[j,k] = perturbation[k] + currSamples[j,k];
          end;
        end;
      ))
    end
    map(wait, refs)
  end
  bestθ
end