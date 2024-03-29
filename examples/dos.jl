using TBG_DFT
using LinearAlgebra
using Plots

gauss = [Gaussian(7, 0.05), Gaussian(5, 0.05)]
L = 1
ϵ = 0.0
model = TbgToy(L, ϵ, gauss)

EcL = 500
EcW = 40
basis = Basis(EcL, EcW, model);

σ = 0.4
xs = collect(-8:0.1:34)
K = Int(round(EcW / 0.1)) 
M = 4000
@time dos = compute_dos_shift_kpm(xs, Gauss(σ), basis, K, M)
P = plot(title=L"\epsilon=%$ϵ", xs, dos)