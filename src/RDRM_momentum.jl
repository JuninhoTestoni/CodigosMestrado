function Rand_mDRM_rsets(A::AbstractMatrix, x0::Vector, b::Vector, r::Int, x̄; j = .5,  l = .4 , itmax = 10000, ϵ = 1e-12, pesos = peso_Vershynin(A), true_error = false, vec_error = false)
	m,n = size(A)
	xk = copy(x0)
	iter = 0
	tol = 1.
    count_error = 0
    index = 1:m
    xko = copy(x0)
   if vec_error
        vec_erro = Float64[]
    end
	while iter < itmax && tol > ϵ
        xkoo = copy(xko)
		xko = xk
		index_sample = sample(index, Weights(pesos), r; replace = false)
        
		@inbounds for k in index_sample
                      @views Ak = A[k, :]
                      @views bk = b[k]
                      xk, = reflec(Ak,xk,bk)
                   end
       
        xk = j*(xko + xk) + l*(xko - xkoo)

       if true_error
            tol = norm(xk - x̄)^2 / norm(x0 - x̄)^2 
        else 
            tol = norm(A*xk - vec(b))
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

