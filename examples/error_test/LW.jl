 using JLD2, Plots, Plots.Measures
 using LinearAlgebra

ϵ = 0
ref = jldopen("pluto/data_LW/ref_008_-4.jld2", "r")
Wtest = jldopen("pluto/data_LW/Wtest_008_-4.jld2", "r")
Ltest = jldopen("pluto/data_LW/Ltest_008_-4.jld2", "r")

#e2W = [norm(x - ref["ldos_ref"]) for x in Wtest["ldosTest"]]
#eInfW = [norm(x - ref["ldos_ref"],Inf) for x in Wtest["ldosTest"]]\
ldosW = Wtest["ldosTest"]
e2W = [norm(ldosW[i+1] - ldosW[i]) for i = 1:length(ldosW)-1]
eInfW = [norm(x - ref["ldos_ref"],Inf) for x in Wtest["ldosTest"]]
eW = hcat([abs.(x - ref["ldos_ref"]) for x in Wtest["ldosTest"]]...)

e2L = [norm(x - ref["ldos_ref"]) for x in Ltest["ldosTest"]]
eInfL = [norm(x - ref["ldos_ref"], Inf) for x in Ltest["ldosTest"]]
eL = hcat([abs.(x - ref["ldos_ref"]) for x in Ltest["ldosTest"]]...)


P1 = plot(Wtest["WTest"][1:end-1], e2W, yscale=:log10, ylabel="", xlabel="W", guidefontsize=22, color=:black, title="ϵ = $(ϵ)", label="2", tickfontsize=20, legendfontsize=20, legend=:topright, grid=:off, box=:on, size=(740, 600), titlefontsize=30, left_margin=2mm, right_margin=2mm, top_margin=3mm, lw=2, marker=:circle, markersize=8, markercolor=:white, markerstrokecolor=:black)
plot!(P1, Wtest["WTest"], eInfW, label="∞", color=:red, lw=2, marker=:circle, markersize=8, markercolor=:white, markerstrokecolor=:red)


P2 = plot(Ltest["LTest"], e2L, yscale=:log10, ylabel="", xlabel="L", guidefontsize=22, color=:black, title="", label="2", tickfontsize=20, legendfontsize=20, legend=:topright, grid=:off, box=:on, size=(740, 600), titlefontsize=30, left_margin=2mm, right_margin=4mm, top_margin=3mm, lw=2, marker=:circle, markersize=8, markercolor=:white, markerstrokecolor=:black)
plot!(P2, Ltest["LTest"], eInfL, label="∞", color=:red, lw=2, marker=:circle, markersize=8, markercolor=:white, markerstrokecolor=:red)

P = plot([P1, P2]..., size=(740, 900), ylabel="Error",left_margin=6mm,layout = grid(2, 1, heights=[0.5, 0.5]))
#savefig("error_lods_-4.png")
#"W = 120, L = 1600 for ϵ = 0"

#heatmap(ref["kpts"], ref["ER"], ref["ldos_ref"])