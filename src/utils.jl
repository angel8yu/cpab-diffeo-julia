function sendtosimple(p::Int, nm, val)
  ref = @spawnat(p, eval(Main, Expr(:(=), nm, val)))
end 

macro sendto(p, nm, val)
  return :( sendtosimple($p, $nm, $val) )
end

macro broadcast(nm, val)
  quote
  @sync for p in workers()
    @async sendtosimple(p, $nm, $val)
  end
  end
end

# same as np.allclose
@everywhere function allclose(a::Any, b::Any; rtol::Float64 = 1e-5, atol::Float64 = 1e-8)
  all(abs(a - b) .<= (atol + rtol * abs(b)))
end

function separateAxis2D(A::Array{Float64,2})
  x = Array(Float64, size(A,1))
	y = Array(Float64, size(A,1))
	for j=1:size(A,1)
	  x[j] = A[j,1]
	  y[j] = A[j,2]
	end
	x,y
end 