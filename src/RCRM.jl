include("../utilitarios.jl")




function Rand_CRM_rsets(A, x0, b, r, x̄; itmax = 20000, ϵ = 1e-12, pesos = peso_Vershynin(A))
	m,n = size(A)
	xk = copy(x0)
	iter = 0
	tol = 1
	numproj = 0
	count_error = 0
	index = 1:m
    
	while iter < itmax && tol > ϵ 
        index_sample = sample(index, Weights(pesos), r; replace = false)
        X = []
        push!(X, xk)
        for k in index_sample
            @views Ak = A[k, :]
            @views bk = b[k]
            xk, skip = reflec(Ak,xk,bk)
            if !skip
            push!(X, xk)
            end
            numproj += 1
        end
		
		try
			xk =  FindCircumcentermSet(X)
		catch 
            count_error += 1
			xk = .5*(X[1] + X[end])
		end
        
        tol = norm(A*xk - b)
        
		iter += 1
	end
    numproj = r*iter
	return xk, iter, tol, numproj, count_error
end

