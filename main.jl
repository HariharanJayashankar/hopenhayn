# Basic Hopenhayn problem

using LinearAlgebra, Plots, QuantEcon, Parameters, Roots

markov_stationary = function(P; niter=1000, tol=1e-10)
    # get stationary distriubiton for markov
    # transition matrix P
    n = size(P, 1)
    dist0 = ones(n)   
    iter = 0
    error = 20

    while error > tol && iter < niter
        dist1 = P' * dist0
        error = maximum(abs.(dist1 - dist0))
        dist0 = dist1
        iter += 1
    end

    return dist0
end

entrant_dist = function(params)
    # whats the inital draw dist for entrants
    # I assue its teh stationary dist of z 
    
    dist = markov_stationary(params.zP)

    return dist
end

params = @with_kw (
    β=0.8,
    n_z = 101,
    ρ = 0.9,
    σ_ϵ = 0.2,
    logzbar = (1-ρ) *  1.4,
    zmc = tauchen(n_z, ρ, σ_ϵ, logzbar, 4),
    zP = zmc.p,
    zgrid = zmc.state_values,
    kₑ=1.0,
    α=0.66,
    n_l=500,
    ngrid = range(0.1, 500.0, n_l),
    k=20.0,
    Dbar=100
)

Π = function(logz, n, p, w=1; α=0.66, k=20)
    # static profit
    
    z = exp(logz)
    
    profit = p*z*(n^α) - w*n - k
    return profit
end

T! = function(V, pol, params, prices)

    # Bellman operator
    @unpack β, n_z, ρ, σ_ϵ, logzbar, zmc, zP, zgrid, kₑ, α, n_l, ngrid, k, Dbar = params
    p, w = prices

    # Bellman Operator
    
    for zidx in 1:n_z
        z = zgrid[zidx] 

        maxsofar = -Inf
        for nidx in 1:n_l
            n = ngrid[nidx]
            u = Π(z,n,p,w; α=α, k=k) + β * max(0, zP[zidx, :] ⋅ V)

            if u >= maxsofar
                V[zidx] = u
                pol[zidx] = nidx
            end
        end
    end
end

firmoutput = function(n, params)
    # meant as the output policy func
    # i.e. after you solve for vfi 
    α = params.α
    out = exp.(params.zgrid) .* n.^α
    return out
end

exitthresh = function(v, params)
    zP = params.zP 
    exitcond = zP * v 
    exitcond[exitcond .< 0] .= Inf
    zidx = argmin(exitcond)
    return zidx
end

vfi = function(prices, params; niter=1000, tol=1e-10)
    # value function iteration

    v0 = zeros(params.n_z)
    pol0 = zero(v0)
    error = 20
    iter=0

    while error > tol && iter < niter
        v1 = copy(v0)
        pol1 = copy(pol0)
        T!(v1, pol1, params, prices) 

        error = maximum(abs.(v1 - v0))
        iter += 1

        v0 = v1
        pol0 = pol1

        if iter % 100 == 0
            println("Iteration: $iter, Error: $error")
        end
    end

    zˢ = exitthresh(v0, params)
    return v0, pol0, zˢ, error, iter
end


v, pol, zˢ, e, i = vfi((5, 1), params())
condmarkov = function(params, zˢ)
    # what edmond calls ϕ
    ϕ = copy(params.zP)
    ϕ[1:zˢ, :] .= 0.0
    return ϕ

end

getge = function(params, niter=1000, tol=1e-10, learningparam=0.9)
    # We normalize w and need to guess p
    # leaningparam is how much p updates each cycle

    # First stage : price
    error = 20
    iter = 0
    V = zeros(params.n_z)
    pol = zero(V)
    g = entrant_dist(params)

    freeentry = function(p)

        prices = (p, 1)
        V, pol, zs, e1, i1 = vfi(prices, params)
        kₑ_test = params.β * V' * g 
        error = kₑ_test - params.kₑ
        return error

    end

    p = find_zero(freeentry, (0.1, 10.0))

    # second stage: guessing m
    prices = (p, 1)
    v, pol, zˢ, error, iter = vfi(prices,params)
    y = firmoutput(pol, params)

    ϕ = condmarkov(params, zˢ)
    μ_1 = 1.0 * (I - ϕ)^(-1) * g
    D = params.Dbar/p
    m = D/(y' * μ_1)
    μ = m * (I - ϕ)^(-1) * g
    
    return p, v, pol, zˢ, μ, m  
end

parameters = params()
p, v, pol, zˢ, μ, m = getge(parameters)

# testing if error is low
eqerror = parameters.β * V' * entrant_dist(parameters) - parameters.kₑ
println("Equlibrium error: $eqerror")
gooderror = firmoutput(pol, parameters) ⋅ μ - parameters.Dbar/p
println("Goods market eq error: $gooderror")

# plots
plot(exp.(parameters.zgrid), v)
plot(exp.(parameters.zgrid), μ )



