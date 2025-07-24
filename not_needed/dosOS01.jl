using TBG_DFT
using LinearAlgebra
using Plots, LaTeXStrings, Plots.Measures

gauss = [Gaussian(7, 0.05), Gaussian(5, 0.05)]
L = 1
EcL = 900
EcW = 34
σ = 0.1
xs = collect(5:0.02:20);
h = 0.04
E = collect(0.01:0.002:0.03)


dos = Vector{Float64}[]
for ϵ in E
	model = TbgToy(L, ϵ, gauss);

	basis = Basis(EcL, EcW, model);
	@time dosi = compute_dos_shift_kpm(xs, Gauss(σ), basis, h; Ktrunc = 9,tol=1e-6);
	push!(dos, dosi)
end
jldsave("dos_ho_01.jld2"; σ, xs, E, dos)


P = plot(xs, dos[1], label= "", lw = 2)
for i = 2:3
    plot!(P, xs, dos[i], label="", lw = 2)
end
P


lens!([5, 10], [0.0, 0.2], inset = (1, bbox(0.06, 0.2, 0.76, 0.76)), subplot=2, ticks=nothing, box=:on)

dos_mat = hcat(dos...)'
heatmap(xlims=(5,10),xs, E, dos_mat,color=:viridis, grid=:off, size=(740, 600), levels=14, xlabel="E", ylabel="ϵ", tickfontsize=20, legendfontsize=18, guidefontsize=26, titlefontsize=30, left_margin=2mm, right_margin=8mm)

#jldsave("dos_ho_02.jld2"; σ, xs, E, dos)

xp = Vector{Vector{Float64}}(undef, length(dos))
ind = collect(1:250)
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
	dxp[ix] = dxx[findall(x->x<1.5,dxx)]
end

P = plot(st=:scatter,label="")
for i = 5:length(dos)
	scatter!(P, E[i]*ones(length(dxp[i])),dxp[i] ./ E[i],label="")
end
P

A = −56.592761096094776/ (2π)^2
B =0.0005892406562679753/(2π)
C = −132.00755479836346
c = sqrt(A*C-B^2)/2