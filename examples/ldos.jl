using TBG_DFT
using LinearAlgebra
using Plots

gauss = [Gaussian(7, 0.05), Gaussian(5, 0.05)]
L = 1
ϵ = 0.01
model = TbgToy(L, ϵ, gauss)

EcL = 200
EcW = 40
Kgrid = collect(-EcW:0.2:EcW)
basis = Basis(EcL, EcW, model; kpts = Kgrid);

σ = 0.4
ϵ = collect(-10:0.1:34)
@time ldos = compute_ldos_kpm(ϵ, Gauss(σ), basis, 2000);
heatmap(Kgrid, ϵ, ldos, title=L"L=%$EcL, W = %$EcW", grid=:off, size=(740, 600),  xlims=(-10,10),levels=14, xlabel=L"\xi", ylabel="Energy", tickfontsize=20, legendfontsize=18, guidefontsize=26, titlefontsize=30, left_margin=2mm, right_margin=4mm)