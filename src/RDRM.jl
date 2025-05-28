function Rand_DRM_rsets(A::AbstractMatrix, x0::Vector, b::Vector, r::Int, x̄; itmax = 10000, ϵ = 1e-12, pesos = peso_Vershynin(A), true_error = false, vec_error = false)
	m,n = size(A)
	xk = copy(x0)
	iter = 0
	tol = 1. 
    count_error = 0
    index = 1:m
    if vec_error
        vec_erro = Float64[]
    end
	while iter < itmax && tol > ϵ 
		xko = copy(xk)
		index_sample = sample(index, Weights(pesos), r; replace = false)
		@inbounds for k in index_sample
                        @views Ak = A[k, :]
                        @views bk = b[k]
                        xk, = reflec(Ak,xk,bk)
            		end
       
         xk = 0.5*(xko + xk)
       
        if true_error
            tol = norm(xk - x̄)^2 / norm(x0 - x̄)^2 
        else 
            tol = norm(A*xk - b)
    	end
        if vec_error
            push!(vec_erro, tol)
        end 
        
		iter += 1
         
	end
	if !vec_error
        return xk, iter, tol, r*iter, count_error
    else
        return xk, iter, tol, r*iter, count_error, vec_erro
    end
end
