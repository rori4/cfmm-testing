include("./optim.jl")

export Router, route!
export netflows!, netflows, update_reserves!

struct Router{O,T}
    objective::O
    cfmms::Vector{CFMM{T}}
    Δs::Vector{AbstractVector{T}}
    Λs::Vector{AbstractVector{T}}
    v::Vector{T}
end

"""
    Router(objective, cfmms, n_tokens)

Constructs a router that finds a set of trades `(router.Δs, router.Λs)` through `cfmms` 
which maximizes `objective`. The number of tokens `n_tokens` must be specified.
"""
function Router(objective::O, cfmms::Vector{C}, n_tokens) where {T,O<:Objective,C<:CFMM{T}}
    VT = Vector{Vector{typeof(objective).parameters[1]}}
    Δs = VT()
    Λs = VT()

    for c in cfmms
        push!(Δs, zerotrade(c))
        push!(Λs, zerotrade(c))
    end

    return Router{O,T}(
        objective,
        cfmms,
        Δs,
        Λs,
        zeros(T, n_tokens)
    )
end
Router(objective, n_tokens) = Router(objective, Vector{CFMM{Float64}}(), n_tokens)

function find_arb!(r::Router, v)
    Threads.@threads for i in 1:length(r.Δs)
        find_arb!(r.Δs[i], r.Λs[i], r.cfmms[i], v[r.cfmms[i].Ai])
    end
end

@doc raw"""
    route!(r::Router)

Solves the routing problem,
```math
\begin{array}{ll}
\text{maximize}     & U(\Psi) \\
\text{subject to}   & \Psi = \sum_{i=1}^m A_i(\Lambda_i - \Delta_i) \\
& \phi_i(R_i + \gamma_i\Delta_i - \Lambda_i) \geq \phi_i(R_i), \quad i = 1, \dots, m \\
&\Delta_i \geq 0, \quad \Lambda_i \geq 0, \quad i = 1, \dots, m.
\end{array}
```
Overwrites `r.Δs` and `r.Λs`.
"""
function route!(r::R; v=nothing, verbose=false, m=5, factr=1e1, pgtol=1e-5, maxfun=15_000, maxiter=15_000, version=1) where {R<:Router}
    # Optimizer set up
    optimizer = L_BFGS_B(length(r.v), 17)
    if isnothing(v)
        r.v .= ones(length(r.v)) / length(r.v) # We should use the initial marginal price here
    else
        r.v .= v
    end

    bounds = zeros(3, length(r.v))
    bounds[1, :] .= 2
    bounds[2, :] .= lower_limit(r.objective)
    bounds[3, :] .= upper_limit(r.objective)

    function fn(v)
        if !all(v .== r.v)
            find_arb!(r, v)
            r.v .= v
        end

        acc = 0.0

        for (Δ, Λ, c) in zip(r.Δs, r.Λs, r.cfmms)
            acc += @views dot(Λ, v[c.Ai]) - dot(Δ, v[c.Ai])
        end

        # Return the sum of the objective function value and the accumulated net trade value
        return f(r.objective, v) + acc
    end
    # Derivative of objective function
    function g!(G, v)
        G .= 0
        if !all(v .== r.v)
            find_arb!(r, v)
            r.v .= v
        end
        grad!(G, r.objective, v)

        for (Δ, Λ, c) in zip(r.Δs, r.Λs, r.cfmms)
            @views G[c.Ai] .+= Λ .- Δ
        end
        return G
    end

    if version == 1
        find_arb!(r, r.v)
        _, v = optimizer(fn, g!, r.v, bounds, m=m, factr=factr, pgtol=pgtol, iprint=verbose ? 1 : -1, maxfun=maxfun, maxiter=maxiter)
        r.v .= v
        find_arb!(r, v)
    elseif version == 2
        find_arb!(r, r.v)

        lb = bounds[2, :]
        ub = bounds[3, :]

        x0 = r.v
        if any(x0 .< lb) || any(x0 .> ub)
            @warn "Initial guess outside bounds - clamping to bounds"
            x0 = clamp.(x0, lb, ub)
        end

        # Objective function needs to accept u and p
        function fV2(u, p)
            return fn(u)
        end

        # Gradient function needs to accept G, u, and p
        function gV2!(G, u, p)
            return g!(G, u)
        end

        optprob = OptimizationFunction(fV2, grad=gV2!)
        prob = Optimization.OptimizationProblem(optprob, x0, lb=lb, ub=ub)
        sol = solve(prob, Optim.LBFGS(m=m))

        r.v .= sol.u
        find_arb!(r, sol.u)
    else
        error("Invalid version")
    end
end

# ----- Convenience functions
function netflows!(ψ, r::Router)
    fill!(ψ, 0)

    for (Δ, Λ, c) in zip(r.Δs, r.Λs, r.cfmms)
        ψ[c.Ai] += Λ - Δ
    end

    return nothing
end

function netflows(r::Router)
    ψ = zero(r.v)
    netflows!(ψ, r)
    return ψ
end

function update_reserves!(r::Router)
    for (Δ, Λ, c) in zip(r.Δs, r.Λs, r.cfmms)
        update_reserves!(c, Δ, Λ, r.v[c.Ai])
    end
    return nothing
end
