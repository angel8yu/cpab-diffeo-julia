@everywhere using DistributedArrays
include("../utils.jl")

# Get the cell id for a given x. Note cells are labeled 1,2,3,...
@everywhere function getCellID1d(x::Float64, nCells::Int64, nCols::Int64)
  id::Int64 = floor(Int64, x * nCells / nCols)+1
  id<0 ? 0 : (id>nCells ? nCells : id) # make it between 0 and nCells
end

# Returns (xTess, velTess)
@everywhere function getTess(As::Array{Float64,3}, nCells::Int64, nCols::Int64)
  colPerCell::Float64 = nCols/nCells
  velTess::Array{Float64,1} = zeros(nCells+1)
  xTess::Array{Float64,1} = zeros(nCells+1)
  for i::Int64=1:nCells
    xTess[i+1]::Float64 = i*colPerCell
    velTess[i+1]::Float64 = getVelocity1d(xTess[i], As, i)
  end
  xTess, velTess
end

# Assuming 0 boundary
@everywhere function getVelocity1d(x::Float64, As::Array{Float64,3}, cellId::Int64)
  As[cellId,1,1]*x + As[cellId,1,2]
end

# loop though all points to get velocity field
@everywhere function getVelocityField1d(pts_src::Array{Float64,1}, As::Array{Float64,3}, nCells::Int64, nCols::Int64)
  nPts::Int64 = length(pts_src)
  v::Array{Float64,1} = zeros(nPts)
  for i::Int64=1:nPts
    x::Float64 = pts_src[i]
    cellId::Int64 = getCellID1d(x, nCells, nCols)
    v[i]::Float64 = getVelocity1d(x, As, cellId)
  end
  v
end

@everywhere function calculateAffinePhi(x::Float64, t::Float64, cellId::Int64, As::Array{Float64,3})
  a::Float64, b::Float64 = As[cellId,1,1], As[cellId,1,2]

  xNext::Float64
  if a == 0
    xNext = x + t * b
  else
    xNext = exp(t*a)*x + b*(exp(t*a)-1)/a
  end
  xNext
end

@everywhere function findCrossingTime(x::Float64, cellId::Int64, As::Array{Float64,3}, xTess::Array{Float64,1}, nCols::Int64)
  v::Float64 = getVelocity1d(x, As, cellId)
  a::Float64, b::Float64 = As[cellId,1,1], As[cellId,1,2]

  # Calculate next cell, velocity
  cellIdNext::Int64 = cellId + round(Int64, sign(v))
  if !(1 <= cellIdNext <= nCells)
    return Inf, Union{}
  end

  xNext::Float64
  if v > 0
    xNext = xTess[cellIdNext] + 1e-16 # pick the right endpoint
  else
    xNext = xTess[cellId] - 1e-16 # pick the left endpoint
  end
  vNext::Float64 = getVelocity1d(xNext, As, cellIdNext)

  # Cases where we will never make it to the next cell
  if allclose(vNext, 0) || sign(v) != sign(vNext)
    #println("b")
    return Inf, Union{}
  end

  t::Float64
  if allclose(xNext, 0) || allclose(xNext, nCols)
    t = Inf
  elseif allclose(a, 0)
    t = (xNext - x)/b
  else
    t = 1/a * log( (xNext + b/a)/(x + b/a) )
  end
  t, cellIdNext
end

@everywhere function calculatePhi(x::Float64, t::Float64, tess::Tuple{Array{Float64,1},Array{Float64,1}}, As::Array{Float64,3}, nCells::Int64, nCols::Int64)
  xTess::Array{Float64,1} = tess[1]
  velTess::Array{Float64,1} = tess[2]

  xPrev::Float64 = x
  xNext::Float64 = 0.0
  #counter::Int64 = 0
  tRemainder::Float64 = t
  v::Float64

  #
  if 0.0<xPrev<nCols && in(xPrev, xTess)
    v = velTess[findfirst(xTess, xPrev)]
    if v>0.0
        pass
    else
      #println("line 105")
      xPrev = xPrev-1e-16
    end
  end

  # Do the function compositions
  while true
    cellId::Int64 = getCellID1d(xPrev, nCells, nCols)
    tCell::Float64, cellIdNext::Any = findCrossingTime(xPrev, cellId, As, xTess, nCols)

    # Whether or not we stay in the same cell
    crossed::Bool
    if tCell >= tRemainder
      tCell = tRemainder
      crossed = false
    else
      crossed = true
    end

    if crossed
      if cellIdNext > cellId
        # Moving to next cell
        xNext = xTess[cellIdNext] + 1e-12
      else
        # going to previous cell
        xNext = xTess[cellId] - 1e-12
      end
    else
      xNext = calculateAffinePhi(xPrev, tCell, cellId, As)
    end

    # Making it stay within 0 and nCols
    xNext = min(nCols, max(0, xNext))

    tRemainder::Float64 -= tCell

    if !crossed
      break
    end

    xPrev = xNext
  end
  xNext
end

@everywhere function calculateNumericalPhi(
    x::Float64,
    t::Float64,
    tess::Tuple{Array{Float64,1},Array{Float64,1}},
    As::Array{Float64,3},
    nCells::Int64,
    nCols::Int64,
    nSteps::Int64
  )

  yn::Float64 = x
  Δt::Float64 = t/nSteps
  for i::Int64 = 1:nSteps
    midpoint::Float64 = yn+Δt/2*getVelocity1d(yn, As, getCellID1d(yn, nCells, nCols))
    cellId::Int64 = getCellID1d(midpoint, nCells, nCols)
    yn = yn + Δt*getVelocity1d(midpoint, As, cellId)
  end
  yn
end

# loop though all points to get integration
function calculateNumericalIntegration(
    pts_src::Array{Float64,1},
    t::Float64,
    tess::Tuple{Array{Float64,1},Array{Float64,1}},
    As::Array{Float64,3},
    nCells::Int64,
    nCols::Int64,
    nSteps::Int64
  )

  nPts::Int64 = length(pts_src)
  pts_fwd::Array{Float64,1} = zeros(nPts)
  x::Float64 = 0.
  for i::Int64=1:nPts
    x = pts_src[i]
    pts_fwd[i] = calculateNumericalPhi(x, t, tess, As, nCells, nCols, nSteps)
  end
  pts_fwd
end

# loop though all points to get integration
function calculateNumericalIntegrationParallel(
    pts_src::Array{Float64,1},
    t::Float64,
    tess::Tuple{Array{Float64,1},Array{Float64,1}},
    As::Array{Float64,3},
    nCells::Int64,
    nCols::Int64,
    nSteps::Int64
  )

  nPts::Int64 = length(pts_src)
  pts_fwd::DArray{Float64,1} = dzeros(nPts)
  threads::Array{Int64,1} = workers()
  nThreads::Int64 = length(threads)
  refs::Array{RemoteRef,1} = Array(RemoteRef, nThreads)

  #spawns the tasks at each worker
  for index::Int64=1:nThreads
    p = threads[index]
    refs[index] = @spawnat(p, (
      indicesOnThread::Array{Int64,1} = localindexes(pts_fwd)[1];
      localArray = localpart(pts_fwd);
      for j::Int64 = 1:length(indicesOnThread);
        localArray[j] = calculateNumericalPhi(pts_src[indicesOnThread[j]], t, tess, As, nCells, nCols, nSteps);
      end;
      ))
  end

  # wait for the workers to finish
  map(wait, refs)

  pts_fwd
end

# loop though all points to get integration
function calculateIntegration(
    pts_src::Array{Float64,1},
    t::Float64,
    tess::Tuple{Array{Float64,1},Array{Float64,1}},
    As::Array{Float64,3},
    nCells::Int64,
    nCols::Int64
  )

  nPts::Int64 = length(pts_src)
  pts_fwd::Array{Float64,1} = zeros(nPts)
  for i::Int64=1:nPts
    x::Float64 = pts_src[i]
    pts_fwd[i] = calculatePhi(x, t, tess, As, nCells, nCols)
  end
  pts_fwd
end

function calculateIntegrationParallel(
    pts_src::Array{Float64,1},
    t::Float64,
    tess::Tuple{Array{Float64,1},Array{Float64,1}},
    As::Array{Float64,3},
    nCells::Int64,
    nCols::Int64
  )

  nPts::Int64 = length(pts_src)
  pts_fwd::DArray{Float64,1} = dzeros(nPts)
  threads::Array{Int64,1} = workers()
  nThreads::Int64 = length(threads)
  refs::Array{RemoteRef,1} = Array(RemoteRef, nThreads)

  # spawns the tasks at each worker
  for index::Int64=1:nThreads
    p = threads[index]
    refs[p] = @spawnat(p, (
      indicesOnThread::Array{Int64,1} = localindexes(pts_fwd)[1];
      localArray = localpart(pts_fwd);
      for j::Int64 = 1:length(indicesOnThread);
        localArray[j] = calculatePhi(pts_src[indicesOnThread[j]], t, tess, As, nCells, nCols);
      end;
      ))
  end

  # wait for the workers to finish
  map(wait, refs)

  pts_fwd
end

@everywhere function solver1d(
    x::Float64,
    t::Float64,
    tess::Tuple{Array{Float64,1},Array{Float64,1}},
    As::Array{Float64,3},
    nCells::Int64,
    nCols::Int64,
    NSteps::Int64,
    Δt::Float64,
    nSteps::Int64
  )

  xPrev::Float64 = x
  cellId::Int64 = getCellID1d(xPrev, nCells, nCols)
  for j::Int64 = 1:NSteps
    xTemp::Float64 = calculateAffinePhi(xPrev, Δt, cellId, As)
    if cellId == getCellID1d(xTemp, nCells, nCols)
      xPrev = xTemp
    else
      xPrev = calculateNumericalPhi(xPrev, Δt, tess, As, nCells, nCols, nSteps)
      cellId == getCellID1d(xPrev, nCells, nCols)
    end
  end
  xPrev
end

function solver1dIntegration(
    pts_src::Array{Float64,1},
    t::Float64,
    tess::Tuple{Array{Float64,1},Array{Float64,1}},
    As::Array{Float64,3},
    nCells::Int64,
    nCols::Int64,
    NSteps::Int64,
    nSteps::Int64
  )

  nPts::Int64 = length(pts_src)
  pts_fwd::Array{Float64,1} = zeros(nPts)
  Δt::Float64 = t/NSteps
  for i::Int64=1:nPts
    pts_fwd[i] = solver1d(pts_src[i], t, tess, As, nCells, nCols, NSteps, Δt, nSteps)
  end
  pts_fwd
end

function solver1dParallelIntegration(
    pts_src::Array{Float64,1},
    t::Float64,
    tess::Tuple{Array{Float64,1},Array{Float64,1}},
    As::Array{Float64,3},
    nCells::Int64,
    nCols::Int64,
    NSteps::Int64,
    nSteps::Int64
  )

  nPts::Int64 = length(pts_src)
  pts_fwd::DArray{Float64,1} = dzeros(nPts)
  Δt::Float64 = t/NSteps
  δt::Float64 = Δt/nSteps
  threads::Array{Int64,1} = workers()
  nThreads::Int64 = length(threads)
  refs::Array{RemoteRef,1} = Array(RemoteRef, nThreads)

  # spawns the tasks at each worker
  for index::Int64=1:nThreads
    p = threads[index]
    refs[index] = @spawnat(p, (
      indicesOnThread::Array{Int64,1} = localindexes(pts_fwd)[1];
      localArray = localpart(pts_fwd);
      for i::Int64 = 1:length(indicesOnThread);
        localArray[i] = solver1d(pts_src[indicesOnThread[i]], t, tess, As, nCells, nCols, NSteps, Δt, nSteps);
      end;
      ))
  end

  # wait for the workers to finish
  map(wait, refs)

  pts_fwd
end