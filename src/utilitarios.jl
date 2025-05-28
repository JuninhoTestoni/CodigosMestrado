using LinearAlgebra
using StatsBase
using Random
using Distributions
using DataFrames, CSV
using BenchmarkTools

teste_dir = "../tests/"
source_dir = "../src/"

##

function normalize_rows!(A, p::Real=2.0)
	for row in eachrow(A)
		normalize!(row, p)
	end
	return nothing
end

##

NF(A) = sqrt(sum(A .^ 2))

##

function peso_Vershynin(A)
    # Calcula as normas das linhas em um único passo
    norml = map(row -> norm(row), eachrow(A))
    # Divide cada norma pela soma total das normas
    peso = norml / sum(norml)
    return peso
end

##

function proj(A,x,b)
    μ =  ((dot(A, x) - b) / dot(A, A))
    return x - μ .* A
end

##

function reflec(A,x,b)
    T = eltype(x)
    μ = (2 * ((dot(A, x) - b) / dot(A, A)))
    if μ ≈ zero(T)
        return x, true
    end
	return x - μ .* A , false
end


##

function block_list(lista, r)
    return [lista[i:min(i + r - 1, end)] for i in 1:r:length(lista)]
end

##

function FindCircumcenter(X)
    circ_error = false
    # Finds the Circumcenter of  linearly independent points  X = [X1, X2, X3, ... Xn]
        # println(typeof(X))
        lengthX = length(X)
        if lengthX  == 1
            return X[1], circ_error
        elseif lengthX == 2
            return .5*(X[1] + X[2]), circ_error
        end
        V = []
        b = Float64[]
        # Forms V = [X[2] - X[1] ... X[n]-X[1]]
        # and b = [dot(V[1],V[1]) ... dot(V[n-1],V[n-1])]
        for ind in 2:lengthX
            difXnX1 = X[ind]-X[1]
            push!(V,difXnX1)
            push!(b,dot(difXnX1,difXnX1))
        end

       # Forms Gram Matrix
        dimG = lengthX-1
        G = diagm(b)

        for irow in 1:(dimG-1)
            for icol in  (irow+1):dimG
                G[irow,icol] = dot(V[irow],V[icol])
                G[icol,irow] = G[irow,icol]
            end
        end
       
        y = similar(X[1])
         try
            L = cholesky(G)
            y = L \ b
        catch
            # Se Cholesky falhar, usa outro tipo de solver
            #F = pinv(G)
            #y = F*b  # Usa a pseudoinversa para resolver diretamente
            #circ_error = true
	    # Outro modo é usando o ponto de Douglas-Rachford
            return .5*(X[1] + X[end]), true
        end
    
        CC = X[1]
        for ind in 1:dimG
            CC += .5*y[ind]*V[ind]
        end
        
        return CC, circ_error
    end

################