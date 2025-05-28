include("../utilitarios.jl")

function Rand_DRM_rsets(A::AbstractMatrix, x0::Vector, b::Vector, r::Int, x̄; j = .5,  l = .4 , itmax = 10000, ϵ = 1e-12, pesos = peso_Vershynin(A), t = 1)
	m,n = size(A)
	xk = copy(x0)
	iter = 0
	tol = 1.
	numproj = 0 
    count_error = 0
    index = 1:m
    
	while iter < itmax && tol > ϵ 
		xko = xk
		index_sample = sample(index, Weights(pesos), r; replace = false)
        
		for k in index_sample
            @views Ak = A[k, :]
            @views bk = b[k]
            xk, _ = reflec(Ak,xk,bk)
			numproj += 1
		end

		xk = j*(xko + xk) + l*(xk - xko)

        iter += 1

       

    end
  
	return xk, iter, tol, numproj, count_error
end