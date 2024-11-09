include("./src/optim.jl")

rosenbrock(x, p) = (BigFloat(1) - x[1])^2 + p[1] * (x[2] - x[1]^2)^2
function rosenbrock_gradient!(G, x, p)
    println("G", G)
    G[1] = -2 * (BigFloat(1) - x[1]) - 4 * p[1] * x[1] * (x[2] - x[1]^2)
    G[2] = 2 * p[1] * (x[2] - x[1]^2)
    println("G_END", G)
end

x0 = BigFloat[-2, 1.0]  # Standard starting point for Rosenbrock
p = [BigFloat(100)]  # Classic Rosenbrock parameter a=100
sol, x_history, f_history = optimize(rosenbrock, rosenbrock_gradient!, x0, BigFloat[0, 0], BigFloat[1, 1], p=p)

# Print optimization results
@printf("Optimization completed in %d iterations\n", length(x_history) - 1)
@printf("Final value: %.18f\n", Float64(f_history[end]))
@printf("Solution: x₁=%.18f, x₂=%.18f\n", Float64(sol.u[1]), Float64(sol.u[2]))
