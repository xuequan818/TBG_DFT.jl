using MomentumDOS
using LinearAlgebra

gauss = [Gaussian(7, 0.05), Gaussian(5, 0.05)]
L = 1
ϵ = 0.01
model = TbgToy(L, ϵ, gauss)

EcL = 500
EcW = 100
Kgrid = collect(-10:0.1:10)

σ = 0.4
xs = collect(-8:0.1:34)

basis = Basis(EcL, EcW, model; kpts=Kgrid);
@time ldos_ref = compute_ldos_kpm(xs, Gauss(σ), basis, 10000);


LTest = collect(160:40:320)
ldosTest = Vector{Any}(undef, length(LTest))
for (i, L) in enumerate(LTest)
    println(" $(i) of $(length(LTest)) : L = $(L)")
    basisL = Basis(L, EcW, model; kpts=Kgrid)
    @time ldosL = compute_ldos_kpm(xs, Gauss(σ), basisL, 10000)
    ldosTest[i] = ldosL
end
e = [norm(ldosi - ldos_ref) for ldosi in ldosTest]
P = plot(LTest, e, yscale=:log10, ylabel="Error", xlabel="W", guidefontsize=22, color=:black, label="", tickfontsize=20, legendfontsize=20, legend=:topright, grid=:off, box=:on, size=(740, 620), titlefontsize=30, right_margin=3mm, top_margin=3mm, lw=2, marker=:circle, markersize=8, markercolor=:white, markerstrokecolor=:black)

i = 5
heatmap(Kgrid, xs, ldosTest[i], title=L"L=%$(LTest[i])", grid=:off, size=(740, 600), xlims=(-10, 10), levels=14, xlabel=L"\xi", ylabel="Energy", tickfontsize=20, legendfontsize=18, guidefontsize=26, titlefontsize=30, left_margin=2mm, right_margin=4mm)

heatmap(Kgrid, xs, ldos_ref, title=L"L=%$EcL", grid=:off, size=(740, 600), xlims=(-10, 10), levels=14, xlabel=L"\xi", ylabel="Energy", tickfontsize=20, legendfontsize=18, guidefontsize=26, titlefontsize=30, left_margin=2mm, right_margin=4mm)
