using TBG_DFT
using LinearAlgebra
using Plots, LaTeXStrings, Plots.Measures
using JLD2

gauss = [Gaussian(7, 0.05), Gaussian(5, 0.05)]
L = 1
EcL = 800
EcW = 36
σ = 0.2
xs = collect(-20:0.02:-10);
h = 0.04
E = collect(0.01:0.002:0.03)

dos = Vector{Float64}[]
for ϵ in E
	model = TbgToy(L, ϵ, gauss);

	basis = Basis(EcL, EcW, model);
	@time dosi = compute_dos_shift_kpm(xs, Gauss(σ), basis, h; Ktrunc = 9,tol=1e-6);
	push!(dos, dosi)
end


f = jldopen("pluto/data_dos_os/dos_os_02.jld2")
dos = f["dos"]
xs = f["xs"]
E = f["E"]

P = plot()
for i in 11:13
    plot!(P, xs, dos[i], label="$(E[i])", lw=2)
end
P
plot!(xlims=(5,8))


lens!([5, 10], [0.0, 0.2], inset = (1, bbox(0.06, 0.2, 0.76, 0.76)), subplot=2, ticks=nothing, box=:on)

dos_mat = hcat(dos...)'
heatmap(xlims=[-10,0],xs, E, dos_mat,color=:viridis, grid=:off, size=(740, 600), levels=14, xlabel="E", ylabel="ϵ", tickfontsize=20, legendfontsize=18, guidefontsize=26, titlefontsize=30, left_margin=2mm, right_margin=8mm)

#jldsave("dos_ho_02.jld2"; σ, xs, E, dos)

xp = Vector{Vector{Float64}}(undef, length(dos))
ind = collect(900:1100)
for (iy,y) in enumerate(dos)
	y = y[ind]
    ym = y[1:end-1] - y[2:end]
    ind1 = findall(x -> x > 0, ym)
    ind2 = findall(x -> x < 0, ym) .+ 1
    mp1 = intersect(ind1, ind2)
    mp2 = findall(x -> abs(x) < 1e-6, y[mp1])
    mp = mp1[setdiff(1:length(mp1), mp2)]
    xp[iy] = xs[ind[mp]]
end

#jldsave("position.jld2"; E, xp)

dxp = Vector{Vector{Float64}}(undef, length(xp))
for (ix, xx) in enumerate(xp)
    dxx = xx[2:end] - xx[1:end-1]
	dxp[ix] = dxx#dxx[findall(x->x<1.5,dxx)]
end

P = plot(st=:scatter,label="")
for (i, x) in enumerate(dxp)
	scatter!(P, E[i]*ones(length(x)),x ./ E[i],label="")
end
P

A = 101.5583951840892 / (2pi)^2
B = -0.1482558604282147 / (2pi)
C = 210.192060625358
c = sqrt(A*C-B^2)