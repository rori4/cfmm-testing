include("./src/CFMMRouter.jl")

# Create CFMMs
cfmms = [
    CFMMRouter.ProductTwoCoin([1e3, 1e4], 0.997, [1, 2]),
    CFMMRouter.ProductTwoCoin([1e3, 1e2], 0.997, [2, 3]),
    CFMMRouter.ProductTwoCoin([1e3, 2e4], 0.997, [1, 3])
]

# We want to liquidate a basket of tokens 2 & 3 into token 1
Δin = [0, 1e1, 1e2]

# Build a routing problem with liquidation objective
router = CFMMRouter.Router(
    CFMMRouter.BasketLiquidation(1, Δin),
    cfmms,
    maximum([maximum(cfmm.Ai) for cfmm in cfmms]),
)

# Optimize!
println("Starting optimization!")
CFMMRouter.route!(router, version=2)

# Print results
Ψ = CFMMRouter.netflows(router)
println("Input Basket: $(Δin)")
println("Net trade: $Ψ")
println("Amount recieved: $(Ψ[1])")

# Print individual trades
for (i, (Δ, Λ)) in enumerate(zip(router.Δs, router.Λs))
    tokens = router.cfmms[i].Ai
    println("CFMM $i:")
    println("\tTendered basket:")
    for (ind, δ) in enumerate(Δ)
        if δ > eps()
            print("\t  $(tokens[ind]): $(δ), ")
        end
    end
    println("\n\tRecieved basket:")
    for (ind, λ) in enumerate(Λ)
        if λ > eps()
            print("\t  $(tokens[ind]): $(λ), ")
        end
    end
    print("\n")
end