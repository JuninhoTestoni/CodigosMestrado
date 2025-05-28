function Rand_mDRM_rsets(A::Matrix, x0::Vector, b::Vector, r::Int, x̄; α = .5, β = .4, itmax = 100000, ϵ = 1e-12)
	m,n = size(A)
	xk = copy(x0)
	iter = 0
	tol = 1.
	norml = Float64[]
	
	numproj = 0 
	xko = xk
	for i = 1:m
	push!(norml, norm(A[i,:]))
	end
	peso = norml / sum(norml)

	while iter <= itmax && tol > ϵ 
		xkoo = xko
		xko = xk
		for j = 1:r
			k = sample(1:m, Weights(peso))
			μ = (2 * ((dot(A[k,:], xk) - b[k]) / dot(A[k,:], A[k,:])))
			xk -= μ .* A[k,:]
		end
		
		xk = α.*xko + (1-α).*xk + β.*(xko - xkoo)
		#tol = norm(xk - x̄) / norm(x0 - x̄)
		#tol = norm(xk - xko)/(norm(xko))
		tol = norm(A*xk - b)
		iter += 1
        numproj = r*iter

	end
	return xk, iter, tol, numproj
end

