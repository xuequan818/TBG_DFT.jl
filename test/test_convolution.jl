using TBG_DFT
using JLD2, Plots

σ = 0.4
g(x) = exp(-x^2/(2σ^2))/sqrt(2pi*σ^2) # support on [-L2,L2], L2 = 3.0

# DoS(g;E)
dos1 = jldopen("test_results_dos/dos_-4_k_01.jld2", "r")["dos"]
dos3 = jldopen("test_results_dos/dos_-4_k_01_04.jld2", "r")["dos"]

# verify g3(E) = ∫Tr[g1(H-E-y)]g2(y)dy = ∫Tr[g2(H-E-y)]g1(y)dy
#   ∫Tr[g1(H-E-y)]g2(y)dy
# = ∫_{-L2}^L2 Tr[g1(H-E-y)]g2(y)dy 
# = hΣ_{y∈[-L2,L2]} DoS(g1;E+y)*g2(y)
L = 4.
xs = collect(-8:0.1:34)
ys = collect(-L:0.1:L)
Es = collect(0:0.1:20)
dos_conv = zero(Es)
El = length(Es)
for y in ys
    ys1 = round.(Es .+ y, digits=1)
	l1 = findfirst(isequal(ys1[1]), xs)
    dos_conv += g(y) .* dos1[l1:l1+El-1]
end
plot(Es, 0.1*dos_conv)
l2 = findfirst(isequal(Es[1]), xs)
plot!(Es, dos3[l2:l2+El-1])
