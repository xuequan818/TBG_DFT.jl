using MomentumDOS
using LinearAlgebra
using Plots, LaTeXStrings, Plots.Measures

gauss = [Gaussian(7, 0.05), Gaussian(5, 0.05)]
L = 1
EcL = 500
EcW = 30
σ = 0.4
xs = collect(-8:0.1:34);
h = 0.1
ϵ = 0.0 #* sqrt(2) - 0.00414
model = TbgToy(L, ϵ, gauss);

basis = Basis(EcL, EcW, model);
@time dos = compute_dos_shift_kpm(xs, Gauss(σ), basis, h; Ktrunc=20, tol=1e-3, M=2100);

plot(xs, dos, label="ϵ=$ϵ")
