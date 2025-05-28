
function Rand_CRM_rsets(A, x0, b, r, x̄; itmax = 20000, ϵ = 1e-12, pesos = peso_Vershynin(A), true_error = false, vec_error = false)
	m,n = size(A)
	xk = copy(x0)
	iter = 0
	tol = 1
	count_error = 0
    index = collect(1:m)
    if vec_error
        vec_erro = Float64[]
    end
	while iter < itmax && tol > ϵ 
           index_sample = sample(index, Weights(pesos), r; replace = false)
            X = []
            push!(X, xk)
            @inbounds for k in index_sample
                            @views Ak = A[k, :]
                            @views bk = b[k]
                            xk, skip = reflec(Ak,xk,bk)
                            if !skip
                            push!(X, xk)
                            end
                      end
    		
    		xk, circ_error =  FindCircumcenter(X)
            
    		 if circ_error
                count_error += 1
            end
            
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
