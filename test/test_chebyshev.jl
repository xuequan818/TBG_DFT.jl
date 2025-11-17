using MomentumDOS
using MomentumDOS: KPM, JacksonKPM
using LinearAlgebra, KrylovKit
using Plots, Plots.Measures, LaTeXStrings

function gCP(x; cf=nothing)
    t0 = 1.0
    t1 = x
    val = cf[1] * t0 + cf[2] * t1
    for i = 3:length(cf)
        t2 = 2 * x * t1 - t0
        val += cf[i] * t2

        t0 = t1
        t1 = t2
    end
    return val
end

gauss = [Gaussian(7, 0.05), Gaussian(5, 0.05)]
L = 1
ϵ = 0.005
model = TbgToy(L, ϵ, gauss)
EcL = 1400
EcW = 100
basis = Basis(EcL, EcW, model);
HV = ham_Potential(basis)
vmin, umin = eigsolve(HV, 1, :SR)
# Elb = λmin(HV),  Eup = 2 * W^2
E1 = 0.5*(basis.EcutW^2 + vmin[1])
E2 = 0.5*(basis.EcutW^2 - vmin[1])
σ = 0.1
σs = σ / E2
Ept = 0
Es = (Ept - E1) / E2
g(x) = exp(-(x - Es)^2 / (2σs^2)) / (sqrt(2pi) * σs)

M = 100000
Npt = Int(round(1.1M))
pt = cos.(range(0, 2pi - pi / Npt, length=2Npt))

@time M1, cf1 = MomentumDOS.genCheb(M, σ, pt, [Ept], E1, E2, KPM(); tol=1e-6)
#@time M2, cf2 = MomentumDOS.genCheb(M, σ, pt, [Ept], E1, E2, JacksonKPM(); tol=1e-6)
h1(x) = gCP(x; cf=cf1)
#h2(x) = gCP(x; cf=cf2)
xx = (collect(-8.:0.1 :34)  .- E1) ./ E2
val1 = h1.(xx)
#val2 = h2.(xx)
val = g.(xx)
@show norm(val1 - val) / E2
#@show norm(val2 - val) / E2
plot(xx, val, lw = 5)
plot!(xx, val1, lw = 4)
#plot!(xx, val2, lw = 2)

gauss = [Gaussian(7, 0.05), Gaussian(5, 0.05)]
L = 1
ϵ = 0.
model = TbgToy(L, ϵ, gauss)

EcL = 100
EcW = 40
Kgrid = collect(-20:0.5:20)
basis = Basis(EcL, EcW, model; kpts=Kgrid);

σ = 0.4
ϵ = collect(-10:0.1:34)
@time ldos_ref = MomentumDOS.compute_ldos(ϵ, Gauss(σ), basis;ERange=10, n_eigs=100)
@time ldos = compute_ldos_kpm(ϵ, Gauss(σ), basis; M=Int(1e5), tol = 1e-5);
@show norm(ldos - ldos_ref, Inf)
heatmap(Kgrid, ϵ, ldos, title=L"L=%$EcL, W = %$EcW", grid=:off, size=(740, 600), levels=14, xlabel=L"\xi", ylabel="Energy", tickfontsize=20, legendfontsize=18, guidefontsize=26, titlefontsize=30, left_margin=2mm, right_margin=4mm)

gauss = [Gaussian(7, 0.05), Gaussian(5, 0.05)]
L = 1
ϵ = 0.0
model = TbgToy(L, ϵ, gauss)

EcL = 100
EcW = 40
basis = Basis(EcL, EcW, model);

σ = 0.1
xs = collect(-10:0.1:34)
h = 0.08
K = Int(round(EcW / h))
@time dos_ref = MomentumDOS.compute_dos_shift(xs, Gauss(σ), basis, 100, K; ERange=10)
@time dos = compute_dos_shift_kpm(xs, Gauss(σ), basis, h; M=Int(1e5), tol=1e-5);
@show norm(dos - dos_ref, Inf)
plot(xs,dos_ref)
