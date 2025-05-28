function Rand_Kaczmarz(A, x0, b, r, x̄; itmax = 100000, ϵ = 1e-12, pesos = peso_Vershynin(A))
	m,n = size(A)
	xk = copy(x0)
	iter = 0
	tol = 1.
	count_error = 0
	index = 1:m
	numproj = 0 
	
	
	while iter < itmax && tol > ϵ
        index_sample = sample(index, Weights(pesos),r)

		for k in index_sample
		@views	Ak = A[k,:]
		xk = proj(Ak,xk,b[k])
		end

		#tol = norm(xk - xko)/(norm(xko))
		#tol = norm(A*xk - b)
		tol = norm(xk - x̄)^2 / norm(x0 - x̄)^2
		iter += 1
		
	end
	numproj = r*iter
	return xk, iter, tol, numproj, count_error
end


