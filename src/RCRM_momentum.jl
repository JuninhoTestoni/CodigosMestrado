include("../utilitarios.jl")

function Rand_CRM_rsets(A, x0, b, r, x̄; itmax = 100000, ϵ = 1e-12)
	num_rows, num_cols = size(A)
	xk = copy(x0)
	iter = 0
	tol = 1
	numproj = 0
	count_error = 0
	peso = peso_Vershynin(A)
	
	while iter < itmax && tol > ϵ 
		X = Matrix{}(undef, num_cols, r+1)
		X[:,1] = xk
		
		weights = Weights(peso)
		for i = 1:r
			# Sample the k-th row of A with probability proportional to its norm
			k = sample(1:num_rows, weights)

			# Calculate the dot product of the k-th row of A and xk
			dot_product_Axk = dot(A[k,:], xk)
			
			# Calculate the residual for the k-th row
			residual = dot_product_Axk - b[k]
			
			# Calculate the norm of the k-th row of A
			norm_Ak = dot(A[k,:], A[k,:])
			
			# Calculate the step size μ
			μ = 2 * (residual / norm_Ak)
			
			# Update xk
			xk -= μ .* A[k,:]
			
			X[:,i+1] = xk
		end
		
		try
			xk = Circuncentro_Rn(X)
		catch 
			count_error += 1
			xk = .5*(X[:,1] + X[:,end])
		end

		tol = norm(A*xk - b)
		#tol = norm(xk - x̄)^2 / norm(x0 - x̄)^2
		#tol = norm(xk - xko) / norm(xko)
		iter += 1
		numproj = iter*r

	end
	return xk, iter, tol, numproj, count_error
end

##


Random.seed!(0)
m = 2000
n = 500
c = 0.6
A = (1-c)*randn(m,n) + c*ones(m,n)
w = randn(m)
x̄ = A'*w / norm(A'w)
b = A*x̄
x0 = zeros(n)

df = DataFrame(Método = String[], r =Int[], time = Float64[])
for r in [2,3,5,10,20,50]
	tempo = Float64[]
	for i = 1:10
		T = @belapsed $Rand_CRM_rsets($A,$x0,$b,$r,$x̄; itmax = 100000, ϵ = 1e-6)
		push!(tempo, T)
	end
	meantime = mean(tempo)
	push!(df, ["Rand_CRM_rsets", r, meantime])
end

df