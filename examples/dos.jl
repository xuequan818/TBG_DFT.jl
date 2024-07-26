using TBG_DFT
using LinearAlgebra
using Plots, LaTeXStrings, Plots.Measures

gauss = [Gaussian(7, 0.05), Gaussian(5, 0.05)]
L = 1
EcL = 800
EcW = 40
σ = 0.4
xs = collect(-8:0.1:34);
h = 0.04
ϵ = 0.0
model = TbgToy(L, ϵ, gauss);

basis = Basis(EcL, EcW, model);
@time dos = compute_dos_shift_kpm(xs, Gauss(σ), basis, h; Ktrunc=20, tol=1e-6)

plot(xs, dos, label="ϵ=$ϵ")