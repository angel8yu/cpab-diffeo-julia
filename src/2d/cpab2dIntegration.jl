@everywhere using DistributedArrays
include("../utils.jl")

# Get the cell id for a given x. Note cells are labeled 1,2,3,...
@everywhere function getCellID2d(
    p::Array{Float64,1},
    nCols::Int64,
    nRows::Int64,
    incX::Float64,
    incY::Float64,
    bdryConstraints::Bool = false,
  )
  #clamp
  p1::Float64 = min(nCols-1e-12, max(0.0, p[1]))
  p2::Float64 = min(nRows-1e-12, max(0.0, p[2]))

  cellId::Int64 = round(Int, floor(Int64, p1/incX)+ nRows/incY*floor(Int64, p2/incY))
  cellId *= 4

  #out of bounds case
  if !bdryConstraints
    #left
    if p[1] ≤ 0.0
      if (p[2]≤0.0 && p[2]/incY<p[1]/incX)
        cellId += 1
      elseif (p[2]≥nRows && (p[2]-nRows)/incY>-p[1]/incX)
        cellId += 3
      else
        cellId += 4
      end
      return cellId
    end

    #right
    if p[1] ≥ nCols
      if (p[2]≤0.0 && -p[2]/incY>(p[1]-nCols)/incX)
        cellId += 1
      elseif (p[2]≥nRows && (p[2]-nRows)/incY>(p[1]-nCols)/incX)
        cellId += 3
      else
        cellId += 2
      end
      return cellId
    end

    #up
    if p[2]≤0.0
      cellId += 1
      return cellId
    end

    #down
    if p[2]≥nRows
      cellId += 3
      return cellId
    end
  end

  #inbounds
  x::Float64 = mod(p1, incX)/incX
  y::Float64 = mod(p2, incY)/incY
  if (x<y)
    if (1-x<y)
      cellId += 3 #down
    else
      cellId += 4 #left
    end
  elseif (1-x<y)
    cellId += 2 #right
  else
    cellId += 1 #top
  end
  cellId
end

# does Ap+B
@everywhere function getVelocity2d(
    p::Array{Float64,1},
    As::Array{Float64,3},
    cellId::Int64,
    v::Array{Float64,1} = Array(Float64,2),
  )
  v[1] = As[cellId,1,1]*p[1] + As[cellId,1,2]*p[2] + As[cellId,1,3]
  v[2] = As[cellId,2,1]*p[1] + As[cellId,2,2]*p[2] + As[cellId,2,3]
  v
end

@everywhere function calculateNumericalPhi(
    p::Array{Float64,1},
    t::Float64,
    As::Array{Float64,3},
    nCols::Int64,
    nRows::Int64,
    incX::Float64,
    incY::Float64,
    nSteps::Int64,
    bdryConstraints::Bool = false,
    midpoint::Array{Float64,1} = Array(Float64,2),
    velocity::Array{Float64,1} = Array(Float64,2),
  )

  yn::Array{Float64,1} = p
  Δt::Float64 = t/nSteps
  cellId::Int64 = 1
  for i::Int64 = 1:nSteps
    cellId = getCellID2d(yn, nCols, nRows, incX, incY, bdryConstraints)
    getVelocity2d(yn, As, cellId, velocity)
    midpoint[1] = yn[1]+Δt/2*velocity[1]
    midpoint[2] = yn[2]+Δt/2*velocity[2]
    yn[1] += Δt*velocity[1]
    yn[2] += Δt*velocity[2]
  end
  yn
end

# Result is computed in pts_fwd
@everywhere function solverODE(
    pts_src::Array{Float64,2},
    pts_fwd::Array{Float64,2},
    t::Float64,
    As::Array{Float64,3},
    nCols::Int64,
    nRows::Int64,
    incX::Float64,
    incY::Float64,
    nSteps::Int64,
    bdryConstraints::Bool = false
  )

  nPts::Int64 = size(pts_src,1)
  for i::Int64=1:nPts
    pts_fwd[i,:] = calculateNumericalPhi(squeeze(pts_src[i,:],1), t, As, nCols, nRows, incX, incY, nSteps, bdryConstraints)
  end
  pts_fwd
end

# Result is computed in pts_fwd
@everywhere function solverODEParallel(
    pts_src::Array{Float64,2},
    pts_fwd::DArray{Float64,2},
    t::Float64,
    As::Array{Float64,3},
    nCols::Int64,
    nRows::Int64,
    incX::Float64,
    incY::Float64,
    nSteps::Int64,
    bdryConstraints::Bool = false
  )

  nPts::Int64 = size(pts_src,1)
  threads::Array{Int64,1} = workers()
  nThreads::Int64 = length(threads)
  refs::Array{RemoteRef,1} = Array(RemoteRef, nThreads)

  # spawns the tasks at each worker
  for index::Int64=1:nThreads
    p = threads[index]
    refs[index] = @spawnat(p, (
      indicesOnThread::Array{Int64,1} = localindexes(pts_fwd)[1];
      localArray = localpart(pts_fwd);
      tic();
      for i::Int64 = 1:length(indicesOnThread);
        localArray[i,:] = calculateNumericalPhi(squeeze(pts_src[indicesOnThread[i],:],1), t, As, nCols, nRows, incX, incY, nSteps, bdryConstraints);
      end;
      toc();
      ))
  end

  # wait for the workers to finish
  pmap(wait, refs)
  pts_fwd
end

# expm seems faster than this, Assume 2d matrix is A[i,1:2,1:2], Stores result in expT[i,1:2,1:2]
@everywhere function expm2x2(A::Array{Float64,3}, expT::Array{Float64,3}, i::Int64)
  a::Float64=A[i,1,1]
  b::Float64=A[i,1,2]
  c::Float64=A[i,2,1]
  d::Float64=A[i,2,2]

  Δtmp::Float64 = (a-d)^2 + 4*b*c
  avgADexp::Float64 = exp((a+d)/2)
  Δ::Float64
  sinhΔoverΔ::Float64

  if Δtmp == 0
    expT[i,1,1] = (1 + (a-d)/2) * avgADexp
    expT[i,1,2] = b * avgADexp
    expT[i,2,1] = c * avgADexp
    expT[i,2,2] = (1 - (a-d)/2) * avgADexp

  elseif Δtmp > 0
    Δ = sqrt(Δtmp) / 2
    coshΔ::Float64 = cosh(Δ)
    sinhΔ::Float64 = sinh(Δ)
    sinhΔoverΔ = sinhΔ / Δ

    expT[i,1,1] = (coshΔ + (a-d)/2 * sinhΔoverΔ) * avgADexp
    expT[i,1,2] = b * sinhΔoverΔ  * avgADexp
    expT[i,2,1] = c * sinhΔoverΔ  * avgADexp
    expT[i,2,2] = (coshΔ - (a-d)/2 * sinhΔoverΔ) * avgADexp
  else
    Δ = sqrt(-Δtmp) / 2
    cosΔ::Float64 = cos(Δ)
    sinΔ::Float64 = sin(Δ)
    sinΔoverΔ = sinΔ / Δ

    expT[i,1,1] = (cosΔ + (a-d)/2 * sinΔoverΔ) * avgADexp
    expT[i,1,2] = b * sinΔoverΔ * avgADexp
    expT[i,2,1] = c * sinΔoverΔ * avgADexp
    expT[i,2,2] = (cosΔ - (a-d)/2 * sinΔoverΔ) * avgADexp
  end
end

# Stores result in expT[i,:,:], Assume 2d matrix is A[i,:,:]
@everywhere function expm3x3(A::Array{Float64,3}, expT::Array{Float64,3}, i::Int64)
  det2x2::Float64 = A[i,1,1]*A[i,2,2] - A[i,1,2]*A[i,2,1]
  if det2x2 != 0.0
    expm2x2(A, expT, i)

    a = A[i,2,2]/det2x2*A[i,1,3] - A[i,1,2]/det2x2*A[i,2,3]
    b = -A[i,2,1]/det2x2*A[i,1,3] + A[i,1,1]/det2x2*A[i,2,3]
    expT[i,1,3] = (expT[i,1,1]-1)*a + expT[i,1,2]*b
    expT[i,2,3] = expT[i,2,1]*a + (expT[i,2,2]-1)*b

    #last row 0,0,1
    expT[i,3,1] = expT[i,3,2] = 0
    expT[i,3,3] = 1
  else
    Ã::Array{Float64,2} = Array(Float64,3,3)
    Ã[1:2,:] = squeeze(A[i,:,:],1)
    Ã[3,1] = Ã[3,2] = Ã[3,3] = 0
    expT[i,:,:] = expm(Ã)
  end
end

@everywhere function expm3x3(A::Array{Float64,2})
  det2x2::Float64 = A[1,1]*A[2,2] - A[1,2]*A[2,1]
  expT::Array{Float64,2} = Array(Float64,3,3)
  if det2x2 != 0.0
    expT[1:2,1:2] = expm2x2(A[1:2,1:2])

    a = A[2,2]/det2x2*A[1,3] - A[1,2]/det2x2*A[2,3]
    b = -A[2,1]/det2x2*A[1,3] + A[1,1]/det2x2*A[2,3]
    expT[1,3] = (expT[1,1]-1)*a + expT[1,2]*b
    expT[2,3] = expT[2,1]*a + (expT[2,2]-1)*b

    #last row 0,0,1
    expT[3,1] = expT[3,2] = 0
    expT[3,3] = 1
  else
    Ã::Array{Float64,2} = Array(Float64,3,3)
    Ã[1:2,:] = A
    Ã[3,1] = Ã[3,2] = Ã[3,3] = 0
    expT[:,:] = expm(Ã)
  end
  expT
end

@everywhere function solver2dPhi(
    p::Array{Float64,1},
    t::Float64,
    As::Array{Float64,3},
    expT::Array{Float64,3},
    expTHalf::Array{Float64,3},
    nCols::Int64,
    nRows::Int64,
    incX::Float64,
    incY::Float64,
    Nsteps::Int64,
    Δt::Float64,
    nSteps::Int64,
    bdryConstraints::Bool = false,
    midpoint::Array{Float64,1} = Array(Float64,2),
    velocity::Array{Float64,1} = Array(Float64,2),
    pTemp::Array{Float64,1} = Array(Float64,2),
  )
  pPrev::Array{Float64,1} = p
  cellId::Int64 = getCellID2d(pPrev, nCols, nRows, incX, incY, bdryConstraints)
  for j::Int64 = 1:Nsteps
    getVelocity2d(pPrev, expT, cellId, pTemp) # expT[cellId] * p
    if cellId == getCellID2d(pTemp, nCols, nRows, incX, incY, bdryConstraints)
      pPrev[1] = pTemp[1]
      pPrev[2] = pTemp[2]
    else
      getVelocity2d(pPrev, expTHalf, cellId, pTemp)
      if cellId == getCellID2d(pTemp, nCols, nRows, incX, incY, bdryConstraints)
        pPrev[1] = pTemp[1]
        pPrev[2] = pTemp[2]
        pPrev = calculateNumericalPhi(pPrev, Δt/2, As, nCols, nRows, incX, incY, round(Int64, nSteps/2), bdryConstraints, midpoint, velocity)
        cellId = getCellID2d(pPrev, nCols, nRows, incX, incY, bdryConstraints)

      else
        pPrev = calculateNumericalPhi(pPrev, Δt, As, nCols, nRows, incX, incY, nSteps, bdryConstraints, midpoint, velocity)
        cellId = getCellID2d(pPrev, nCols, nRows, incX, incY, bdryConstraints)
      end
    end
  end
  pPrev
end

# Stores result in pts_fwd
@everywhere function solver2d(
    pts_src::Array{Float64,2},
    pts_fwd::Array{Float64,2},
    t::Float64,
    As::Array{Float64,3},
    nCols::Int64,
    nRows::Int64,
    incX::Float64,
    incY::Float64,
    Nsteps::Int64,
    Δt::Float64,
    nSteps::Int64,
    ΔtAs::Array{Float64,3},
    bdryConstraints::Bool = false,
    expT::Array{Float64,3} = Array(Float64, size(As,1), 3, 3),
  )
  for i::Int64 = 1:size(As, 1)
    expm3x3(ΔtAs, expT, i)
  end

  expTHalf = Array(Float64, size(As,1), 3, 3)
  for i::Int64 = 1:size(As, 1)
    expm3x3(ΔtAs/2, expTHalf, i)
  end

  nPts::Int64 = size(pts_src,1)
  midpoint::Array{Float64,1} = Array(Float64,2)
  velocity::Array{Float64,1} = Array(Float64,2)
  currPoint::Array{Float64,1} = Array(Float64,2)
  pTemp::Array{Float64,1} = Array(Float64,2)
  for i::Int64=1:nPts
    currPoint[1] = pts_src[i,1]
    currPoint[2] = pts_src[i,2]
    pts_fwd[i,:] = solver2dPhi(currPoint, t, As, expT, expTHalf, nCols, nRows, incX, incY, Nsteps, Δt, nSteps, bdryConstraints, midpoint, velocity, pTemp)
  end
  pts_fwd
end

# Stores result in pts_fwd (note: Distributed Array)
@everywhere function solver2dParallel(
    pts_src::Array{Float64,2},
    pts_fwd::DArray{Float64,2},
    t::Float64,
    As::Array{Float64,3},
    nCols::Int64,
    nRows::Int64,
    incX::Float64,
    incY::Float64,
    Nsteps::Int64,
    Δt::Float64,
    nSteps::Int64,
    ΔtAs::Array{Float64,3},
    bdryConstraints::Bool = false,
    expT::Array{Float64,3} = Array(Float64, size(As,1), 3, 3),
  )
  
  for i::Int64 = 1:size(As, 1)
    expm3x3(ΔtAs, expT, i)
  end

  expTHalf = Array(Float64, size(As,1), 3, 3)
  for i::Int64 = 1:size(As, 1)
    expm3x3(ΔtAs/2, expTHalf, i)
  end

  nPts::Int64 = size(pts_src,1)
  threads::Array{Int64,1} = workers()
  nThreads::Int64 = length(threads)
  refs::Array{RemoteRef,1} = Array(RemoteRef, nThreads)

  #spawns the tasks at each worker
  for index::Int64=1:nThreads
    p = threads[index]
    refs[index] = @spawnat(p, (
      indicesOnThread::Array{Int64,1} = localindexes(pts_fwd)[1];
      localArray = localpart(pts_fwd);
      midpoint::Array{Float64,1} = Array(Float64,2);
      velocity::Array{Float64,1} = Array(Float64,2);
      currPoint::Array{Float64,1} = Array(Float64,2);
      pTemp::Array{Float64,1} = Array(Float64,2);
      for i::Int64 = 1:length(indicesOnThread);
        currPoint[1] = pts_src[indicesOnThread[i],1]
        currPoint[2] = pts_src[indicesOnThread[i],2]
        localArray[i,:] = solver2dPhi(currPoint, t, As, expT, expTHalf, nCols, nRows, incX, incY, Nsteps, Δt, nSteps, bdryConstraints, midpoint, velocity, pTemp);
      end;
      ))
  end

  # wait for the workers to finish
  map(wait, refs)
  pts_fwd
end

# This function retuns the range indexes assigned to this worker
@everywhere function myrange(q::SharedArray)
    idx = indexpids(q)
    if idx == 0
        # This worker is not assigned a piece
        return 1:0
    end
    nchunks = length(procs(q))
    splits = [round(Int, s) for s in linspace(0,size(q,1),nchunks+1)]
    splits[idx]+1:splits[idx+1]
end

# Stores result in pts_fwd (note: Shared Array)
@everywhere function solver2dParallelShared(
    pts_src::Array{Float64,2},
    pts_fwd::SharedArray{Float64,2},
    t::Float64,
    As::Array{Float64,3},
    nCols::Int64,
    nRows::Int64,
    incX::Float64,
    incY::Float64,
    Nsteps::Int64,
    Δt::Float64,
    nSteps::Int64,
    ΔtAs::Array{Float64,3},
    bdryConstraints::Bool = false,
    expT::Array{Float64,3} = Array(Float64, size(As,1), 3, 3),
  )
  
  for i::Int64 = 1:size(As, 1)
    expm3x3(ΔtAs, expT, i)
  end

  expTHalf = Array(Float64, size(As,1), 3, 3)
  for i::Int64 = 1:size(As, 1)
    expm3x3(ΔtAs/2, expTHalf, i)
  end

  nPts::Int64 = size(pts_src,1)
  threads::Array{Int64,1} = workers()
  nThreads::Int64 = length(threads)
  refs::Array{RemoteRef,1} = Array(RemoteRef, nThreads)

  # spawns the tasks at each worker
  for index::Int64=1:nThreads
    p = threads[index]
    refs[index] = @spawnat(p, (
      midpoint::Array{Float64,1} = Array(Float64,2);
      velocity::Array{Float64,1} = Array(Float64,2);
      currPoint::Array{Float64,1} = Array(Float64,2);
      pTemp::Array{Float64,1} = Array(Float64,2);
      for i::Int64 = myrange(pts_fwd);
        currPoint[1] = pts_src[i,1];
        currPoint[2] = pts_src[i,2];
        pts_fwd[i,:] = solver2dPhi(currPoint, t, As, expT, expTHalf, nCols, nRows, incX, incY, Nsteps, Δt, nSteps, bdryConstraints, midpoint, velocity, pTemp);
      end;
      ))
  end

  # wait for the workers to finish
  map(wait, refs)
  pts_fwd
end