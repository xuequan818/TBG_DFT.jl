#using TBG_DFT
using LinearAlgebra
using Plots

gauss = [Gaussian(7, 0.05), Gaussian(5, 0.05)]
L = 1
ϵ = 0.01
model = TbgToy(L, ϵ, gauss)

EcL = 1000
EcW = 100
basis = Basis(EcL, EcW, model);

σ = 0.4
xs = collect(-8:0.1:34)
K = Int(round(EcW / 0.1)) 
#xx = collect(range(-EcW, EcW, length=2K))
M = 10
@time dos = compute_dos_shift_kpm(xs, Gauss(σ), basis, K, M)
plot!(xs, dos)
P = plot(title=L"\epsilon=%$ϵ", xs, dos)

using JLD2
jldsave("test_dos/dos_05.jld2";ϵ, xs, dos)