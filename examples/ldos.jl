using TBG_DFT
using LinearAlgebra
using Plots, Plots.Measures, LaTeXStrings

gauss = [Gaussian(7, 0.05), Gaussian(5, 0.05)]
L = 1

ϵ = 0.0
model = TbgToy(L, ϵ, gauss)

EcL = 400
EcW = 50
Kgrid = collect(1:0.05:5)
basis = Basis(EcL, EcW, model; kpts = Kgrid);

σ = 0.15
E = collect(-17:0.01:-13.5)
@time ldos = compute_ldos_kpm(E, Gauss(σ), basis; tol=1e-4);

heatmap(Kgrid, E, ldos, title=L"\epsilon=%$ϵ", color=:viridis,grid=:off, size=(740, 600), levels=14, xlabel=L"\xi", ylabel="Energy", tickfontsize=20, legendfontsize=18, guidefontsize=26, titlefontsize=30, left_margin=2mm, right_margin=8mm, dpi=300)

savefig("ldos0_2.png")