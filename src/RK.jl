function Rand_Kaczmarz(A, x0, b, r, x̄; itmax = 1000000, ϵ = 1e-12,pesos = peso_Vershynin(A), true_error = false, vec_error = false)
	m,n = size(A)
	xk = copy(x0)
	iter = 0
	tol = 1
	numproj = 0 
    count_error = 0
    if vec_error
        vec_erro = Float64[]
    end
	while iter < itmax && tol > ϵ 
        k = sample(1:m, Weights(pesos))
		@views	Ak = A[k,:]
        @views bk = b[k]
		xk = proj(Ak,xk,bk)
        numproj += 1
        
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
        return xk, iter, tol, numproj, count_error
    else
        return xk, iter, tol, numproj, count_error, vec_erro
    end
end