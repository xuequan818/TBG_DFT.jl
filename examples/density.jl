using TBG_DFT
using LinearAlgebra
using Plots, Plots.Measures, LaTeXStrings


gauss = [Gaussian(7, 0.05), Gaussian(5, 0.05)]
L = 1
ϵ = 0.05
model = TbgToy(L, ϵ, gauss);

EcL = 100
EcW = 20
basis = Basis(EcL, EcW, model);

σ = 1.
xs = collect(-10.:1:34);
h = 0.1
@time rho_vec = compute_density_vec(xs, Gauss(σ), basis, h; Ktrunc=10, tol=1e-6);
xx = collect(-10:0.01:30)
rho = compute_density(xx, rho_vec, basis)
rho_eig_vec = compute_density_eig(xs, Gauss(σ), basis, h, 50)
rho_eig = compute_density(xx, rho_eig_vec, basis)

#plot!(xx, rho*1000)

heatmap(xx, xs, rho, title=L"\epsilon=%$ϵ", grid=:off, size=(740, 600), levels=14, xlabel=L"x", ylabel="Energy", tickfontsize=20, legendfontsize=18, guidefontsize=26, titlefontsize=30, left_margin=2mm, right_margin=8mm)

plot(xx, rho[:,200])

ϵ = 0.05
lat = 20
latR = 2pi / lat
L = 1
g(x, σ, A) = -(A / (σ * sqrt(2pi))) * exp(-x^2 / (2σ^2))
v1(x) = g(x, 0.05, 7)# support on (-0.5,0.5)
v2(x) = g(x, 0.05, 5)
R = collect(-10:60)
v(x) =
    sum(R) do r
        (v1(x - r) + v2((1 + ϵ) * x - r))
    end
xx = collect(-10:0.1:30)
plot(xx, v.(xx))