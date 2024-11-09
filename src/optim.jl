using Optimization, OptimizationOptimJL
using Printf
using LineSearches

function optimize(f, g!, x0, lb, ub; p=nothing)
    # Check if initial guess is within bounds
    # the L_BFGS_B seems to be adjusting the guess to be within bounds
    if any(x0 .< lb) || any(x0 .> ub)
        @warn "Initial guess outside bounds - clamping to bounds"
        x0 = clamp.(x0, lb, ub)
    end

    optprob = OptimizationFunction(f, Optimization.AutoForwardDiff(), grad=g!)
    prob = Optimization.OptimizationProblem(optprob, x0, p, lb=lb, ub=ub)
    sol = solve(prob, Optim.LBFGS(m=10))

    return sol, x_history, f_history
end
