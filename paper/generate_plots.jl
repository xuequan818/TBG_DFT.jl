### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ b1aa93b4-831b-4440-a71f-f6648082708b
begin
	using PythonCall
	const mticker = PythonCall.pyimport("matplotlib.ticker")
end

# ╔═╡ 61f6b454-deb5-45ca-bce7-6749fddd0816
md"# Plots generation for article"

# ╔═╡ 62f2a254-1ec9-4f68-ad57-8d58df909670
md"## Preparation"

# ╔═╡ b34451a8-38d9-4267-8114-46fdf5ee69b2
function is_running_in_pluto()
    return isdefined(Main, :PlutoRunner)
end;

# ╔═╡ 570be094-6ea9-11f0-1c4e-55c239b84740
begin
    using PythonPlot
    const plt = PythonPlot
    using Interpolations
	using ClassicalOrthogonalPolynomials
    using LaTeXStrings
    using JLD2
    if is_running_in_pluto()
        using PlutoUI
    end
    using Printf
end

# ╔═╡ 903eff0e-6642-4f71-9d15-0f3b239c0594
md"### Parameters"

# ╔═╡ 3b18d124-92ca-4b6c-8bf0-a9151c1977e4
begin
    const DATA_DIR = (dir = joinpath(@__DIR__, "..",  "data"); isdir(dir) ? dir : joinpath(@__DIR__, "data"))
    const IMAGE_DIR = (dir = joinpath(@__DIR__, "..", "images"); isdir(dir) ? dir : joinpath(@__DIR__, "images"))

    const SIGMA_VALS = ["04", "008"]
    const ORDERS = [0, 1, 2]
    const EPSILON_VALS = [0.001, 0.01]

    const ENERGY_RANGE = [-20, 20]
end

# ╔═╡ 4312571a-39c8-41e4-a0bb-114831849dd1
begin
    const Es = [-14.046, -7.662, 10.133, 11.287]
    const ωs = [-13.762, 23.461, -235.907, 263.204]
end

# ╔═╡ 9dc73e37-50bc-45f8-9d97-9848950ff689
begin
    width = 426/72.27
    golden = (1 + 5^0.50) / 2
    figsize=(width,width/golden)
end

# ╔═╡ 21f4cd71-4b29-497a-a66d-4482cd2b163d
begin
    plt.rc("font", family="serif")
    plt.rc("axes", titlesize=25, labelsize=20, grid=false)
    plt.rc("axes.spines", top=false, right=false)
    plt.rc("xtick", labelsize=18)
    plt.rc("ytick", labelsize=18)
    plt.rc("legend", fontsize=16, frameon=false)
    plt.rc("lines", linewidth=1.5)
    plt.rc("savefig", dpi=500)
    plt.rc("text",usetex="True")

    # Coloring scheme. Defaults to tab10
    momentum = (; label="momentum-space", color="C0", linestyle="-")
    semiclassical = (; label="semiclassical", color="C1", linestyle="--",
                     dashes=(5, 5))
    theory = (; label="harmonic model", color="C2", linestyle="-.")
    band_colors = ["C0", "C1", "C2"]
    band_contour_colors = ["Blues", "Oranges"]
end

# ╔═╡ ecfba73d-a1f2-4867-8237-fb2719bfd7a6
md"### Loading functions"

# ╔═╡ b4d1900c-9d84-4459-925c-f23c4b6e4b7c
function load_planewave_derivatives(σs)
    data_0 = load(joinpath(DATA_DIR, "dos_0_$(σs).jld2"))
    data_p = load(joinpath(DATA_DIR, "dos_-5_$(σs).jld2"))
    data_m = load(joinpath(DATA_DIR, "dos_-5_$(σs)m.jld2"))

    tr0 = data_0["dos"]
    tr1 = (data_p["dos"] - data_m["dos"]) ./ (data_p["ϵ"] - data_m["ϵ"])
    tr2 = (data_p["dos"] + data_m["dos"] - 2 .* data_0["dos"]) ./ (2 * data_p["ϵ"]^2)

    return [tr0, tr1, tr2]
end;

# ╔═╡ f222ba3d-8d59-475d-b387-35777fdbacfe
function load_semiclassical_derivatives(σs)
    data = jldopen(joinpath(DATA_DIR, "long_$(σs).jld2"), "r")
    trs = [data["tr0"], data["tr1"], data["tr2"]]
    xs = data["xs"]
    return trs, xs
end;

# ╔═╡ 2dd3a3f1-94a9-4659-84f9-8d30877a493d
md"## Plots"

# ╔═╡ 3bcaa492-2517-4d63-8240-529c98c2f0aa
md"### Comparison between the two models"

# ╔═╡ 099e92bd-31dd-4590-b9eb-36e6fcb0a1ee
begin
    all_plots_comparison_merged = Dict{String, Any}()

    for order in ORDERS
        fig, axs = plt.subplots(2, length(SIGMA_VALS);
                                figsize=(figsize[1] * length(SIGMA_VALS), figsize[2]),
                                sharex=true,
                                sharey=false,
                                gridspec_kw=Dict("height_ratios" => [0.60, 0.40]))

        flat_axs = axs.flatten()
        num_sigmas = length(SIGMA_VALS)

        local handles, labels

        for (iσ, σs) in enumerate(SIGMA_VALS)
            planewave_trs = load_planewave_derivatives(σs)
            semiclassical_trs, xs = load_semiclassical_derivatives(σs)
            ind = findall(x -> ENERGY_RANGE[1] <= x <= ENERGY_RANGE[2], xs)

            py_index_deriv = iσ - 1
            py_index_diff = num_sigmas + iσ - 1

            ax_deriv = flat_axs.__getitem__(py_index_deriv)
            ax_diff  = flat_axs.__getitem__(py_index_diff)

            ax_deriv.plot(xs[ind], planewave_trs[order+1][ind]; momentum...)

            ax_deriv.plot(xs[ind], semiclassical_trs[order+1][ind]; semiclassical...)

            ax_deriv.set_title("\$\\sigma = $(Symbol(σs[1], ".", σs[2:end]))\$")

            if iσ == 1
                handles, labels = ax_deriv.get_legend_handles_labels()
            end

            ax_diff.plot(xs[ind],
                         abs.(planewave_trs[order+1][ind] - semiclassical_trs[order+1][ind]);
                         color="black",
                        )

            ax_diff.set_yscale("log")
            ax_diff.set_xlabel(L"Energy $E$")

            # Force two ticks, otherwise may only be one...
            ax_diff.yaxis.set_major_locator(mticker.LogLocator(numticks=3))

            if iσ == 1
                ax_deriv.set_ylabel(L"L_{%$(order),\sigma}(E)")
                ax_diff.set_ylabel(L"|\Delta L_{%$(order),\sigma}|")
            end
        end

        fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.1))

        #fig.suptitle("Order $order Derivative Comparison for Different \$\\sigma\$", y=1.02, fontsize=16)
        #fig.tight_layout(rect=[0, 0.03, 1, 0.98])

        all_plots_comparison_merged["Order = $order"] = fig

        filename = joinpath(IMAGE_DIR, "combined_order_$(order).pdf")
        fig.savefig(filename, bbox_inches="tight")  # tight to prevent title of head
        println("Saved plot: $filename")
    end
end

# ╔═╡ 54a94ad1-bee1-4331-98df-e5e07ab7d8f5
if is_running_in_pluto()
    @bind selected_comparison_plot_merged Select(collect(keys(all_plots_comparison_merged)))
end

# ╔═╡ 6759d423-a369-44f8-838d-3fd1b8af241f
if is_running_in_pluto()
    all_plots_comparison_merged[selected_comparison_plot_merged]
end

# ╔═╡ adcd347e-5aa7-409e-aecd-069a11f4485c
md"### Plots for all expansion"

# ╔═╡ 4e674238-1bf1-4984-979c-15e9f71d4d6c
md"### Combined plot"

# ╔═╡ df8290e5-000a-4803-a94e-26e1dd541a8b
begin
    all_plots_reconstructed_merged = Dict()

    for ε in EPSILON_VALS
        num_sigmas = length(SIGMA_VALS)


        fig, axs = plt.subplots(1, num_sigmas;
                                figsize=(figsize[1] * num_sigmas, figsize[2]),
                                sharey=false,
                               )

        local handles, labels

        for (iσ, σs) in enumerate(SIGMA_VALS)
            py_index = iσ - 1
            ax = axs.__getitem__(py_index)

            semiclassical_trs, xs = load_semiclassical_derivatives(σs)

            ε_str = string(ε)[3:end]
            filepath = joinpath(DATA_DIR, "dos_$(ε_str)_$σs.jld2")

            dos_planewave = jldopen(filepath, "r")["dos"]

            tr0, tr1, tr2 = semiclassical_trs
            dos_semiclassical = tr0 .+ tr1 .* ε .+ tr2 .* ε^2
            ind = findall(x -> ENERGY_RANGE[1] <= x <= ENERGY_RANGE[2], xs)

            ax.plot(xs[ind], dos_planewave[ind]; momentum...)

            ax.plot(xs[ind], dos_semiclassical[ind]; semiclassical...)

            title_σ = Symbol(σs[1], ".", σs[2:end])
            ax.set_title(L"$\sigma = %$(title_σ)$")

            ax.set_xlabel(L"Energy $E$")

            if iσ == 1
                handles, labels = ax.get_legend_handles_labels()
            end
        end

        axs.__getitem__(0).set_ylabel("\$\\nu_{\\epsilon,\\sigma}\$ (\$\\epsilon = $(ε)\$)")
        fig.legend(handles,
                   labels,
                   loc="upper center",
                   ncol=2,
                   bbox_to_anchor=(0.5, 1.1),
                  )

        all_plots_reconstructed_merged["ε = $(ε)"] = fig

        ε_str = string(ε)[3:end]
        filename = joinpath(IMAGE_DIR, "combined_dos_$(ε_str).pdf")
        fig.savefig(filename, bbox_inches="tight")
        println("Saved merged plot: $filename")
    end
end

# ╔═╡ abaeab70-4dc4-4262-857a-59df89a443f6
if is_running_in_pluto()
    @bind selected_reconstructed_merged_plot Select(collect(keys(all_plots_reconstructed_merged)))
end

# ╔═╡ 0ecd3570-b18d-4265-9139-cdab8fa9f441
if is_running_in_pluto()
    all_plots_reconstructed_merged[selected_reconstructed_merged_plot]
end

# ╔═╡ 888b97ee-7dac-4369-bc81-e9de57b7a4b8
md"### Oscillations plot"

# ╔═╡ 5064021b-787e-4464-b449-a551d20f3646
begin
    function plot_full_dos_oscillations()
        filepath = joinpath(DATA_DIR, "dos_01_004.jld2")
        data_os = jldopen(filepath, "r")

        fig, ax = plt.subplots(figsize=(figsize[1], figsize[2]))

        ax.plot(data_os["xs"], data_os["dos"]; momentum...)

        ax.legend()

        ax.set_xlabel("Energy \$E\$")
        ax.set_ylabel("\$\\nu_{\\epsilon,\\sigma}\$ (\$\\epsilon = 0.01, \\sigma = 0.04\$)")

        fig.tight_layout()

        filename = joinpath(IMAGE_DIR, "dos_os.pdf")
        fig.savefig(filename, bbox_inches="tight")
        println("Saved plot: $filename")

        return fig
    end

    oscillations_plot = plot_full_dos_oscillations()
end

# ╔═╡ a70e031d-e8d8-41fe-b315-5c5e450187e5
md"### Plot band structure"

# ╔═╡ 3d3d9b35-a444-41e1-8acc-481bc4f1aa33
let
    function plot_band_structure()
        filepath = joinpath(DATA_DIR, "band_structure.jld2")

        data = load(filepath)
        λs_matrix = data["λs_matrix"]
        Xgrid_number = data["Xgrid_number"]
        kgrid_number = data["kgrid_number"]
        n_bands = data["n_bands"]

        fig, ax = plt.subplots(figsize=(figsize[1], figsize[2]))
        ax.set_xlabel(L"$X$-disregistry")
        ax.set_ylabel("Energy")

        x_axis = [iX / Xgrid_number for iX in 1:Xgrid_number]

        for ik in 1:kgrid_number
            for iband in 1:n_bands
                band_data = @view λs_matrix[iband, ik, :]

                ax.plot(x_axis,
                        band_data,
                        color=band_colors[iband],
                       )
            end
        end

        extrema_points = []
        ymin, ymax = ax.get_ylim()
        y_offset = (ymax - ymin) * 0.08

        # 1. Maximum of the first band
        val, idx = findmax(@view λs_matrix[1, :, :])
        push!(extrema_points, (x_axis[idx[2]], val, L"1", -y_offset))

        # 2. Minimum of the second band
        val, idx = findmin(@view λs_matrix[2, :, :])
        push!(extrema_points, (x_axis[idx[2]], val, L"2", y_offset))

        # 3 & 4. The two maxima of the second band
        temp_band2 = copy(@view λs_matrix[2, :, :])
        val1, idx1 = findmax(temp_band2)
        push!(extrema_points, (x_axis[idx1[2]], val1, L"4", -y_offset))
        temp_band2[idx1] = -Inf
        val2, idx2 = findmax(temp_band2)
        push!(extrema_points, (x_axis[idx2[2]], val2, L"3", -y_offset))

        # 5 & 6. The two minima of the third band
        temp_band3 = copy(@view λs_matrix[3, :, :])
        val1, idx1 = findmin(temp_band3)
        push!(extrema_points, (x_axis[idx1[2]], val1, L"5", y_offset))
        temp_band3[idx1] = Inf
        val2, idx2 = findmin(temp_band3)
        push!(extrema_points, (x_axis[idx2[2]], val2, L"6", y_offset))

        for (x, y, label, y_shift) in extrema_points
            ax.scatter(x, y;
                       color="white",
                       edgecolor="black",
                       s=70,
                       zorder=5,
                      )
            ax.text(x, y + y_shift, label;
                    color="black",
                    ha="center",
                    va="center",
                    fontsize=18,
                    zorder=5,
                    bbox=Dict("facecolor"=>"white",
                              "edgecolor"=>"white",
                              "boxstyle"=>"round",
                              "pad"=>0.1,
                              "alpha"=>0.9),
                   )
        end

        fig.tight_layout()
        filename = joinpath(IMAGE_DIR, "band_structure_with_extrema.pdf")
        fig.savefig(filename, bbox_inches="tight")
        println("Saved plot: $filename")

        return fig
    end

    band_structure_plot = plot_band_structure()
end

# ╔═╡ 7220c410-be31-450c-811b-dd544d1c62c0
md"### Contour plots"

# ╔═╡ 8da4e3f6-f85f-4115-83c6-e852c4963b43
contour_combined = let
    function plot_extrema_contours_combined()
        fig, axs = plt.subplots(1, 2;
            figsize=(figsize[1] * 1.8, 0.84 * figsize[1]),
            sharey=false,
        )

        handles, labels = axs[1].get_legend_handles_labels()

        filepath = joinpath(DATA_DIR, "band_structure_finer.jld2")

        data = load(filepath)
        λs_matrix = data["λs_matrix"]
        kgrid_number = data["kgrid_number"]

        k_axis = (1:kgrid_number) ./ (kgrid_number + 1) .* 2pi

        λs_matrix_sorted = λs_matrix

        x_axis = data["x_axis"]
        #x_axis = (1:x_axis.len) ./ (x_axis.len + 1)

        # First band
        band1_data = @view λs_matrix_sorted[1, :, :]

        ax1 = axs.__getitem__(0)

        #ax1.set_title(L"Energy near $p_\ell = 1$"; y=1.05)
        ax1.set_xlabel(L"$X$-disregistry")

        ax1.set_xticks([x_axis[1], 0.5, x_axis[end]])
        ax1.set_xticklabels([L"0", L"0.5", L"1.0"])

        ax1.set_yticks([x_axis[1], π, k_axis[end]])
        ax1.set_yticklabels([L"$0$", L"$\pi$", L"$2\pi$"])


        levels1 = collect(-29.5:0.5:-14.5)
        CS1 = ax1.contour(x_axis, k_axis, band1_data;
            levels=levels1,
            cmap=band_contour_colors[1],
        )

        levels_to_label = collect(-14.5:-2:-29.5)
        ax1.clabel(CS1;
            inline=true,
            fontsize=10,
            colors="black",
            levels=levels_to_label,
        )
        ax1.set_title(L"$E_1(k,X)$"; y=1.05)


        # Second band
        band2_data = @view λs_matrix_sorted[2, :, :]

        ax2 = axs.__getitem__(1)

        #ax2.set_title(L"Energy near $p_\ell = 2$"; y=1.05)
        ax2.set_xlabel(L"$X$-disregistry")

        ax2.set_xticks([x_axis[1], 0.5, x_axis[end]])
        ax2.set_xticklabels([L"0", L"0.5", L"1.0"])

        ax2.set_yticks([x_axis[1], π, k_axis[end]])
        ax2.set_yticklabels([L"$0$", L"$\pi$", L"$2\pi$"])

        CS2 = ax2.contour(x_axis, k_axis, band2_data;
            levels=collect(-7.5:0.5:7.5),
            cmap="Oranges_r",
        )

        ax2.clabel(CS2;
            inline=true,
            fontsize=10,
            colors="black",
            levels=collect(-7.5:2:7.5),
        )
        ax2.set_title(L"$E_2(k,X)$"; y=1.05)


        axs.__getitem__(0).set_ylabel(L"$k$-point")

        filename = joinpath(IMAGE_DIR, "contour_band_combined.pdf")
        fig.savefig(filename, bbox_inches="tight")
        println("Saved combined plot: $filename")

        return fig
    end

    plot_extrema_contours_combined()
end;

# ╔═╡ a726c9f8-d12e-4134-b7b7-52d334fb30e6
md"### Wigner plots"

# ╔═╡ 5fe4285d-a238-4bec-8f7a-1d9fbb53300e
md"
```math
W(x,y)=\frac{1}{\pi}\int_{-\infty}^\infty\psi(x-y)^*\psi(x+y)
```
"

# ╔═╡ a8eb020a-1783-4478-81c3-532a81e9868a
function wavefun(x, n, mω)
    (1 / sqrt(2^n * prod(1:n))) * (mω / pi)^(1 / 4) * exp(-mω * x^2 / 2) * hermiteh(n, sqrt(mω) * x)
end

# ╔═╡ 542e7613-3233-4b47-ac35-12ed5ad2b2d1
function wigner(ψ, x, k; L=20, h=0.1)
    ygrid = collect(-L:h:L)
    val = @. ψ(x + ygrid) * ψ(x - ygrid) * exp(2 * im * k * ygrid)
    sum(val) * h / pi
end

# ╔═╡ 3cb58227-5ba3-4943-a59c-765673ee8784
wigner_combined = let
    ϵ = 0.01
    cw = "C2"
    fig, axs = plt.subplots(3, 2;
        figsize=(figsize[1] * 1.8, figsize[1] * 2.8),
        sharex=true,
        sharey=false,
        gridspec_kw=Dict("hspace" => 0.1))

    flat_axs = axs.flatten()

    local handles, labels

    function plot_wigner_combined()
        plots_dict = Dict()

        filepath = joinpath(DATA_DIR, "band_structure_finer.jld2")

        data = load(filepath)
        λs_matrix = data["λs_matrix"]
        kgrid_number = data["kgrid_number"]

        k_axis = (1:kgrid_number) ./ (kgrid_number + 1) .* 2pi
        lnk = length(k_axis) ÷ 4
        k0 = pi
        k_axis = k_axis[lnk:3lnk+1]
        λs_matrix_sorted = λs_matrix

        x_axis = data["x_axis"]
        #x_axis = data["x_axis"]
        lnx = length(x_axis) ÷ 4
        X0 = 0.5
        x_axis = x_axis[lnx:3lnx+1]

        xs = x_axis
        ks = k_axis

        # First band
        band1_data = @view λs_matrix_sorted[1, :, :]
        band1_data = band1_data[lnx:3lnx+1, lnk:3lnk+1]
        # Second band
        band2_data = @view λs_matrix_sorted[2, :, :]
        band2_data = band2_data[lnx:3lnx+1, lnk:3lnk+1]


        modes = [0, 2, 4]
        for (imode, mode) in enumerate(modes)
            ax1 = flat_axs.__getitem__(2 * (imode - 1))
            if imode == 3
                ax1.set_xlabel(L"$X$-disregistry")
            end

            ax1.set_ylabel(L"$k$-point ($n=%$(mode)$)")

            ax1.set_xticks([0.4, 0.6])
            ax1.set_xticklabels([L"0.4", L"$0.6$"])

            ax1.set_yticks([π / 2, π, 3π / 2])
            ax1.set_yticklabels([L"$\pi/2$", L"$\pi$", L"$3\pi/2$"])

            levels1 = vcat(collect(-14.1:-0.2:-15.5), collect(-16:-0.5:-29.5))
            sort!(levels1)
            CS1 = ax1.contour(x_axis, k_axis, band1_data;
                levels=levels1,
                cmap=band_contour_colors[1],
            )

            levels_to_label = collect(-14.5:-2:-29.5)

            ψn(x) = wavefun(x, mode, ϵ * 9.592)
            wigner_mat = abs.([wigner(ψn, (x - X0) / ϵ, k - k0) for k in ks, x in xs])

            mw = maximum(wigner_mat)
            levelsw1 = vcat(1.2e-3, collect(mw/5:0.15:mw))

            ax1.contour(xs, ks, wigner_mat;
                colors=cw, levels=levelsw1, linestyles="-.", linewidths=2)

            if imode == 1
                handles, labels = ax1.get_legend_handles_labels()
            end

            ax2 = flat_axs.__getitem__(2 * (imode - 1) + 1)
            if imode == 3
                ax2.set_xlabel(L"$X$-disregistry")
            end

            ax2.set_xticks([0.4, 0.6])
            ax2.set_xticklabels([L"0.4", L"$0.6$"])

            ax2.set_yticks([π / 2, π, 3π / 2])
            ax2.set_yticklabels([L"$\pi/2$", L"$\pi$", L"$3\pi/2$"])

            levels2 = vcat(collect(-7.5:0.2:-6.5), collect(-6:0.5:7.5))
            CS2 = ax2.contour(x_axis, k_axis, band2_data;
                levels=levels2,
                cmap="Oranges_r",
            )

            ψn(x) = wavefun(x, mode, ϵ * 8.962)
            wigner_mat = abs.([wigner(ψn, (x - X0) / ϵ, k - k0) for k in ks, x in xs])

            mw = maximum(wigner_mat)
            levelsw2 = vcat(1.2e-3, collect(mw/5:0.15:mw))

            ax2.contour(xs, ks, wigner_mat;
                colors=cw, levels=levelsw2, linestyles="-.", linewidths=2)

            if imode == 1
                ax1.set_title(L"$E_1(k,X)$ near $p=1$"; y=1.05)
                ax2.set_title(L"$E_2(k,X)$ near $p=2$"; y=1.05)
            end
        end

        filename = joinpath(IMAGE_DIR, "wigner_combined.pdf")
        fig.savefig(filename, bbox_inches="tight")
        println("Saved combined plot: $filename")

        return fig
    end

    plot_wigner_combined()
end;

# ╔═╡ 48dac3f6-bd91-44b7-af6f-ead473f4279e
md"### Plot oscillations near local extrema"

# ╔═╡ 668e2a3d-0972-4e1d-a0ce-67683ae3d670
function Interp2D(x, y, data, factor)
    IC = CubicSplineInterpolation((axes(data, 1), axes(data, 2)), data)

    xx = LinRange(x[1], x[end], length(x) * factor)
    yy = LinRange(y[1], y[end], length(y) * factor)
    finerx = LinRange(firstindex(data, 1), lastindex(data, 1), size(data, 1) * factor)
    finery = LinRange(firstindex(data, 2), lastindex(data, 2), size(data, 2) * factor)

    data_interp = IC.(finerx, finery')

    return xx, yy, data_interp
end

# ╔═╡ 17dbf805-f582-4bee-8d6c-34d2e22f356a
begin
    oscillations_data = let
        filepath = joinpath(DATA_DIR, "dos_os_004.jld2")
        data_os = jldopen(filepath, "r")
        (; ys=data_os["ER"], xs=data_os["ϵs"], dos=data_os["dos"])
    end

    function plot_oscillations_data(ax, k, data)
        (; ys, xs, dos) = data
        yl = [-14.7, -7.75, 5.5, 11.]
        yr = [-14, -6.5, 10.5, 16.5]
        ind = findall(x -> yl[k] < x < yr[k], ys)
        dos_mat = hcat(dos...)'

        xx, yy, val = Interp2D(xs, ys[ind], dos_mat[:, ind]', 10)

        ax.imshow(val;
            cmap="plasma",
            extent=[xx[1], xx[end], yy[1], yy[end]],
            origin="lower",
            aspect="auto",
            interpolation="bilinear",
        )

        ax.set_xticks([0.002, 0.006, 0.010])
        ax.set_xlabel(L"\epsilon")

        y_formatter_py(value, pos) = @sprintf "%.1f" value
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(y_formatter_py))

        if k in [1, 3]
            ax.set_ylabel("Energy \$E\$ (\$\\sigma = 0.04\$)")
        end

        N = [3, 3, 1, 1]
        f(x; n) = Es[k] + ωs[k] * x * (n + 0.5)

        for n in 0:N[k]
            ax.plot(xx, f.(xx; n);
                color="white",
                ls="--",
            )

            x_text = 0.008
            y_text = f(x_text; n=n)

            ax.text(x_text, y_text, L"%$(n)";
                ha="center",
                va="center",
                fontsize=18,
                color="black",
                bbox=Dict("facecolor" => "white",
                    "edgecolor" => "white",
                    "pad" => 0.1,
                    "boxstyle" => "round",
                    "alpha" => 0.9,
                ))
        end

        ell = [1, 2, 2, 3][k]
        n_param = [1, 2, 3, 5][k]
        info_text = "\$\\nu_{\\epsilon,\\sigma}\$ near \$p\$ = $(n_param)"
        ax.text(xs[end] - 0.0002, yr[k], info_text;
            ha="right",
            va="top",
            fontsize=12,
            color="black",
            bbox=Dict("facecolor" => "white"),
        )
    end

    function oscillations_combined(iband)
        kp = isone(iband) ? 0 : 2
        fig, axs = subplots(1, 2;
            figsize=(figsize[1], figsize[1] * 0.7),
            sharey=false,
            gridspec_kw=Dict("wspace" => 0.5))

        for k in 1:2
            plot_oscillations_data(axs.__getitem__(k - 1), k + kp, oscillations_data)
        end

        handles, labels = axs[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))

        filename = joinpath(IMAGE_DIR, "oscillations_combined_$(iband).pdf")
        fig.savefig(filename, bbox_inches="tight")
        println("Saved combined plot: $filename")

        fig
    end

    for iband in 1:2
        oscillations_combined(iband)
    end
end

# ╔═╡ 71a4462a-785c-4b0b-a29c-aa50c91b386f
md"### Plot dos near local extrema"

# ╔═╡ 2033a8e1-c74c-4671-a72b-707befacf6c4
function Interp1D(x, data, factor)
    IC = CubicSplineInterpolation(axes(data), data)

    xx = LinRange(x[1], x[end], length(x) * factor)
    finerx = LinRange(firstindex(data), lastindex(data), length(data) * factor)

    data_interp = IC.(finerx)

    return xx, data_interp
end

# ╔═╡ 083e42e4-e1d2-4472-9a92-20744f289b56
begin
    dos_slice_data = let
        filepath = joinpath(DATA_DIR, "dos_os_004.jld2")
        data_os = jldopen(filepath, "r")

        (; σ=data_os["σ"], xs=data_os["ER"], εs=data_os["ϵs"], dos_all=data_os["dos"])
    end

    function plot_dos_slice(ax, k, data)
        (; σ, xs, εs, dos_all) = data

        ε = 0.01
        dos = dos_all[findmin(x -> abs(x - ε), εs)[2]]

        xl = [-14.45, -7.78, 6.8, 10.9]
        xr = [-13.95, -7.02, 10.2, 15]
        Xticks = [[-14.4, -14.2, -14.0], [-7.7, -7.4, -7.1], [7, 8.5, 10], [11, 13, 15]]

        ind = findall(x -> xl[k] < x < xr[k], xs)
        xx, val = Interp1D(xs[ind], dos[ind], 5)

        ax.plot(xx, val; momentum...)

        Ej(n) = Es[k] + ωs[k] * ε * (n + 0.5)
        f(x, μ) = exp(-(x - μ)^2 / (2σ^2)) / (sqrt(2pi) * σ)
        g(x) = sum(f.(x, Ej.(0:30))) * ε
        wk = k in [1, 2] ? 1 : 2

        ax.plot(xx, wk * g.(xx); theory...)

        ax.set_xlabel("Energy \$E\$")
        ax.set_xticks(Xticks[k])

        if k in [1, 3]
            ax.set_ylabel("\$\\nu_{\\epsilon,\\sigma}\$ (\$\\epsilon = 0.01, \\sigma = $(σ)\$)")
        end
    end

    function dos_comparison_combined(iband)
        kp = isone(iband) ? 0 : 2
        fig, axs = subplots(1, 2;
            figsize=(figsize[1], figsize[1] * 0.7),
            sharey=false,
            gridspec_kw=Dict("wspace" => 0.5))

        for k in 1:2
            plot_dos_slice(axs.__getitem__(k - 1), k + kp, dos_slice_data)
        end

        handles, labels = axs[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.48, 1.0))

        filename = joinpath(IMAGE_DIR, "dos_comparison_combined_$(iband).pdf")
        fig.savefig(filename, bbox_inches="tight")
        println("Saved combined plot: $filename")

        fig
    end

    for iband in 1:2
        dos_comparison_combined(iband)
    end
end

# ╔═╡ 193ece9c-6c93-4ea4-a091-d39ba8887901
md"### Zoom up"

# ╔═╡ 2f15acac-67a5-4974-a3f6-0296c117a142
begin
    zoom_data = let
        filepath = joinpath(DATA_DIR, "dos_os_004.jld2")
        data_os = jldopen(filepath, "r")
        # Store all data in a single named tuple for convenience
        (; σ=data_os["σ"], ER=data_os["ER"], ϵs=data_os["ϵs"], dos=data_os["dos"])
    end

    function plot_heatmap_zoom(ax, k, data)
        (; ER, ϵs, dos) = data # Unpack data
        dos_mat = hcat(dos...)'

        yl = [-14.12, -7.7, 9.1, 11.1]
        yr = [-14.02, -7.55, 10.1, 12.2]

        yind = findall(x -> yl[k] < x < yr[k], ER)
        xind = findall(x -> 0.001 <= x <= 0.0022, ϵs)
        xx, yy, val = Interp2D(ϵs[xind], ER[yind], dos_mat[xind, yind]', 20)

        ax.imshow(val;
            cmap="plasma",
            extent=[xx[1], xx[end], yy[1], yy[end]],
            origin="lower",
            aspect="auto",
        )

        ax.set_xlabel(L"\epsilon")
        ax.set_xlim(0.001, 0.002)
        ax.set_xticks([0.001, 0.002])

		if k <= 2
			formatter = mticker.FormatStrFormatter("%.2f")
		else
			formatter = mticker.FormatStrFormatter("%.1f")
		end
	    ax.yaxis.set_major_formatter(formatter)

        f(x; n) = Es[k] + ωs[k] * x * (n + 0.5)
        for n in 0:1
            ax.plot(xx, f.(xx; n); color="white", ls="--")
            x_text = 0.0018
            y_text = f(x_text; n=n)
            ax.text(x_text, y_text, L"%$(n)";
                ha="center", va="center", fontsize=18, color="black",
                bbox=Dict("facecolor" => "white", "edgecolor" => "white", "pad" => 0.1,
                    "boxstyle" => "round", "alpha" => 0.9))
        end

        ell = [1, 2, 2, 3][k]
        n_param = [1, 2, 3, 5][k]
        info_text = "\$\\nu_{\\epsilon,\\sigma}\$ near \$p\$ = $(n_param)"
        ax.text(0.98, 1.02, info_text;
            ha="right", va="top", fontsize=12, color="black",
			transform=ax.transAxes,
            bbox=Dict("facecolor" => "white"))
    end

    function plot_dos_slice_zoom(ax, k, data)
        (; σ, ER, ϵs, dos) = data

        ε = 0.001
        dos = dos[findmin(x -> abs(x - ε), ϵs)[2]]

        xl = [-14.45, -7.78, 9.6, 11]
        xr = [-13.95, -7.02, 10.2, 12.]
        Xticks = [[-14.4, -14.2, -14], [-7.1, -7.4, -7.7], [9.6, 10], [11, 11.5, 12]]

        ind = findall(x -> xl[k] < x < xr[k], ER)
        xx, val = Interp1D(ER[ind], dos[ind], 5)

        ax.plot(xx, val; momentum...)

        Ej(n) = Es[k] + ωs[k] * ε * (n + 0.5)
        f(x, μ) = exp(-(x - μ)^2 / (2σ^2)) / (sqrt(2pi) * σ)
        g(x) = sum(f.(x, Ej.(0:30))) * ε

        ax.plot(xx, 2 * g.(xx); theory...)

        ax.set_xlabel("Energy \$E\$")
        ax.set_xticks(Xticks[k])
    end

    figs = Dict()
    for iband in [2]
        figs["heatmap_$iband"] = let
            fig, axs = subplots(1, 2;
                figsize=(figsize[1], figsize[1] * 0.7),
                sharey=false,
                gridspec_kw=Dict("wspace" => 0.5),
			)

            kp = isone(iband) ? 0 : 2
            plot_heatmap_zoom(axs[0], 1 + kp, zoom_data)
            plot_heatmap_zoom(axs[1], 2 + kp, zoom_data)

            axs[0].set_ylabel("Energy \$E\$ (\$\\sigma = 0.04\$)")
            axs[1].set_ylabel("")

            filename = joinpath(IMAGE_DIR, "combined_zoom_os_$iband.pdf")
            savefig(filename, bbox_inches="tight")
            println("Saved combined plot: $filename")

            fig
        end

        figs["oscillations_$iband"] = let
            fig, axs = subplots(1, 2;
                figsize=(figsize[1], figsize[1] * 0.7),
                sharey=false,
                gridspec_kw=Dict("wspace" => 0.5))

            kp = isone(iband) ? 0 : 2
            plot_dos_slice_zoom(axs[0], 1 + kp, zoom_data)
            plot_dos_slice_zoom(axs[1], 2 + kp, zoom_data)


            axs[0].set_ylabel("\$\\nu_{\\epsilon,\\sigma}\$ (\$\\epsilon = 0.001, \\sigma = $(zoom_data.σ)\$)")
            axs[1].set_ylabel("")

            handles, labels = axs[1].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.48, 1.0))

            filename = joinpath(IMAGE_DIR, "combined_zoom_dos_$iband.pdf")
            savefig(filename, bbox_inches="tight")
            println("Saved combined plot: $filename")

            fig
        end
    end


    figs
end;

# ╔═╡ 2991bb15-229f-4fff-9162-0805b01e679d
if is_running_in_pluto()
    @bind select_fig Select(collect(keys(figs)))
end

# ╔═╡ 8a4072af-e549-4ed6-9a76-16613851e8ab
if is_running_in_pluto()
    figs[select_fig]
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ClassicalOrthogonalPolynomials = "b30e2e7b-c4ee-47da-9d5f-2c5c27239acd"
Interpolations = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
PythonPlot = "274fc56d-3b97-40fa-a1cd-1b4a50311bf9"

[compat]
ClassicalOrthogonalPolynomials = "~0.15.11"
Interpolations = "~0.16.2"
JLD2 = "~0.6.2"
LaTeXStrings = "~1.4.0"
PlutoUI = "~0.7.71"
PythonCall = "~0.9.29"
PythonPlot = "~1.0.6"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.5"
manifest_format = "2.0"
project_hash = "8a349c62e095b8748d00955e53a3ab39fbd4e8ca"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "f7817e2e585aa6d924fd714df1e2a84be7896c60"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.3.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArrayLayouts]]
deps = ["FillArrays", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "355ab2d61069927d4247cd69ad0e1f140b31e30d"
uuid = "4c555306-a7a7-4459-81d9-ec55ddd5c99a"
version = "1.12.0"
weakdeps = ["SparseArrays"]

    [deps.ArrayLayouts.extensions]
    ArrayLayoutsSparseArraysExt = "SparseArrays"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.BandedMatrices]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra", "PrecompileTools"]
git-tree-sha1 = "4826c9fe6023a87029e54870ad1a9800c7ea6623"
uuid = "aae01518-5342-5314-be14-df237901396f"
version = "1.10.1"

    [deps.BandedMatrices.extensions]
    BandedMatricesSparseArraysExt = "SparseArrays"
    CliqueTreesExt = "CliqueTrees"

    [deps.BandedMatrices.weakdeps]
    CliqueTrees = "60701a23-6482-424a-84db-faee86b9b1f8"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Bessels]]
git-tree-sha1 = "4435559dc39793d53a9e3d278e185e920b4619ef"
uuid = "0e736298-9ec6-45e8-9647-e4fc86a2fe38"
version = "0.2.8"

[[deps.BlockArrays]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra"]
git-tree-sha1 = "d06da0a776248b2cb4d8d3c3dd37222183d303eb"
uuid = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
version = "1.9.2"
weakdeps = ["Adapt", "BandedMatrices"]

    [deps.BlockArrays.extensions]
    BlockArraysAdaptExt = "Adapt"
    BlockArraysBandedMatricesExt = "BandedMatrices"

[[deps.BlockBandedMatrices]]
deps = ["ArrayLayouts", "BandedMatrices", "BlockArrays", "FillArrays", "LinearAlgebra", "MatrixFactorizations"]
git-tree-sha1 = "4eef2d2793002ef8221fe561cc822eb252afa72f"
uuid = "ffab5731-97b5-5995-9138-79e8c1846df0"
version = "0.13.4"
weakdeps = ["SparseArrays"]

    [deps.BlockBandedMatrices.extensions]
    BlockBandedMatricesSparseArraysExt = "SparseArrays"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "e4c6a16e77171a5f5e25e9646617ab1c276c5607"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.ChunkCodecCore]]
git-tree-sha1 = "51f4c10ee01bda57371e977931de39ee0f0cdb3e"
uuid = "0b6fb165-00bc-4d37-ab8b-79f91016dbe1"
version = "1.0.0"

[[deps.ChunkCodecLibZlib]]
deps = ["ChunkCodecCore", "Zlib_jll"]
git-tree-sha1 = "cee8104904c53d39eb94fd06cbe60cb5acde7177"
uuid = "4c0bbee4-addc-4d73-81a0-b6caacae83c8"
version = "1.0.0"

[[deps.ChunkCodecLibZstd]]
deps = ["ChunkCodecCore", "Zstd_jll"]
git-tree-sha1 = "34d9873079e4cb3d0c62926a225136824677073f"
uuid = "55437552-ac27-4d47-9aa3-63184e8fd398"
version = "1.0.0"

[[deps.ClassicalOrthogonalPolynomials]]
deps = ["ArrayLayouts", "BandedMatrices", "BlockArrays", "BlockBandedMatrices", "ContinuumArrays", "DomainSets", "FFTW", "FastGaussQuadrature", "FastTransforms", "FillArrays", "HypergeometricFunctions", "InfiniteArrays", "InfiniteLinearAlgebra", "IntervalSets", "LazyArrays", "LazyBandedMatrices", "LinearAlgebra", "QuasiArrays", "RecurrenceRelationshipArrays", "RecurrenceRelationships", "SpecialFunctions"]
git-tree-sha1 = "aa7f992993fffc2ecad10843b078b4aa3d084373"
uuid = "b30e2e7b-c4ee-47da-9d5f-2c5c27239acd"
version = "0.15.11"

    [deps.ClassicalOrthogonalPolynomials.extensions]
    ClassicalOrthogonalPolynomialsMutableArithmeticsExt = "MutableArithmetics"

    [deps.ClassicalOrthogonalPolynomials.weakdeps]
    MutableArithmetics = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

    [deps.ColorTypes.weakdeps]
    StyledStrings = "f489334b-da3d-4c2e-b8f0-e476e12c162b"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "0037835448781bb46feb39866934e243886d756a"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.CompositeTypes]]
git-tree-sha1 = "bce26c3dab336582805503bed209faab1c279768"
uuid = "b152e2b5-7a66-4b01-a709-34e65c35f657"
version = "0.1.4"

[[deps.CondaPkg]]
deps = ["JSON3", "Markdown", "MicroMamba", "Pidfile", "Pkg", "Preferences", "Scratch", "TOML", "pixi_jll"]
git-tree-sha1 = "bd491d55b97a036caae1d78729bdb70bf7dababc"
uuid = "992eb4ea-22a4-4c89-a5bb-47a3300528ab"
version = "0.2.33"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.ContinuumArrays]]
deps = ["AbstractFFTs", "ArrayLayouts", "BandedMatrices", "BlockArrays", "DomainSets", "FillArrays", "InfiniteArrays", "Infinities", "IntervalSets", "LazyArrays", "LinearAlgebra", "QuasiArrays", "StaticArrays"]
git-tree-sha1 = "6e9b87acb2deeb34d0edc96ddb24a38e1f859d90"
uuid = "7ae1f121-cc2c-504b-ac30-9b923412ae5c"
version = "0.20.1"

    [deps.ContinuumArrays.extensions]
    ContinuumArraysMakieExt = "Makie"
    ContinuumArraysRecipesBaseExt = "RecipesBase"

    [deps.ContinuumArrays.weakdeps]
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"

[[deps.DSP]]
deps = ["Bessels", "FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "5989debfc3b38f736e69724818210c67ffee4352"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.8.4"
weakdeps = ["OffsetArrays"]

    [deps.DSP.extensions]
    OffsetArraysExt = "OffsetArrays"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.DomainSets]]
deps = ["CompositeTypes", "IntervalSets", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "c249d86e97a7e8398ce2068dce4c078a1c3464de"
uuid = "5b8099bc-c8ec-5219-889f-1d9e522a28bf"
version = "0.7.16"

    [deps.DomainSets.extensions]
    DomainSetsMakieExt = "Makie"
    DomainSetsRandomExt = "Random"

    [deps.DomainSets.weakdeps]
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "Libdl", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "97f08406df914023af55ade2f843c39e99c5d969"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.10.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d6219a004b8cf1e0b4dbe27a2860b8e04eba0be"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.11+0"

[[deps.FastGaussQuadrature]]
deps = ["LinearAlgebra", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "0044e9f5e49a57e88205e8f30ab73928b05fe5b6"
uuid = "442a2c76-b920-505d-bb47-c5924d526838"
version = "1.1.0"

[[deps.FastTransforms]]
deps = ["AbstractFFTs", "ArrayLayouts", "BandedMatrices", "FFTW", "FastGaussQuadrature", "FastTransforms_jll", "FillArrays", "GenericFFT", "LazyArrays", "Libdl", "LinearAlgebra", "RecurrenceRelationships", "SpecialFunctions", "ToeplitzMatrices"]
git-tree-sha1 = "b41969ccec1379b33967c9b720a250d4687cfc2d"
uuid = "057dd010-8810-581a-b7be-e3fc3b93f78c"
version = "0.17.0"

[[deps.FastTransforms_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "FFTW_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl", "MPFR_jll", "OpenBLAS_jll"]
git-tree-sha1 = "efb41482692019ed03e0de67b9e48e88c0504e7d"
uuid = "34b6f7d7-08f9-5794-9e10-3819e4c7e49a"
version = "0.6.3+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "b66970a70db13f45b7e57fbda1736e1cf72174ea"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.17.0"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "5bfcd42851cf2f1b303f51525a54dc5e98d408a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.15.0"

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

    [deps.FillArrays.weakdeps]
    PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GMP_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "781609d7-10c4-51f6-84f2-b8444358ff6d"
version = "6.3.0+0"

[[deps.GenericFFT]]
deps = ["AbstractFFTs", "FFTW", "LinearAlgebra", "Reexport"]
git-tree-sha1 = "1bc01f2ea9a0226a60723794ff86b8017739f5d9"
uuid = "a8297547-1b15-4a5a-a998-a2ac5f1cef28"
version = "0.1.6"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "68c173f4f449de5b438ee67ed0c9c748dc31a2ec"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.28"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.InfiniteArrays]]
deps = ["ArrayLayouts", "FillArrays", "Infinities", "LazyArrays", "LinearAlgebra"]
git-tree-sha1 = "f9fb453287ef06d182939f18b02b2ea5773954d3"
uuid = "4858937d-0d70-526a-a4dd-2d5cb5dd786c"
version = "0.15.9"
weakdeps = ["BandedMatrices", "BlockArrays", "BlockBandedMatrices", "DSP", "Statistics"]

    [deps.InfiniteArrays.extensions]
    InfiniteArraysBandedMatricesExt = "BandedMatrices"
    InfiniteArraysBlockArraysExt = "BlockArrays"
    InfiniteArraysBlockBandedMatricesExt = "BlockBandedMatrices"
    InfiniteArraysDSPExt = "DSP"
    InfiniteArraysStatisticsExt = "Statistics"

[[deps.InfiniteLinearAlgebra]]
deps = ["ArrayLayouts", "BandedMatrices", "BlockArrays", "BlockBandedMatrices", "FillArrays", "InfiniteArrays", "Infinities", "LazyArrays", "LazyBandedMatrices", "LinearAlgebra", "MatrixFactorizations", "SemiseparableMatrices"]
git-tree-sha1 = "9aa4a229916c06f2f9296382b8c852cd086b1992"
uuid = "cde9dba0-b1de-11e9-2c62-0bab9446c55c"
version = "0.10.2"

[[deps.Infinities]]
git-tree-sha1 = "4495006c20b2fd27b8c453a1dd31d423654f3772"
uuid = "e1ba4f0e-776d-440f-acd9-e1d2e9742647"
version = "0.1.12"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "ec1debd61c300961f98064cfb21287613ad7f303"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "65d505fa4c0d7072990d659ef3fc086eb6da8208"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.16.2"

    [deps.Interpolations.extensions]
    InterpolationsForwardDiffExt = "ForwardDiff"
    InterpolationsUnitfulExt = "Unitful"

    [deps.Interpolations.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.IntervalSets]]
git-tree-sha1 = "03b4f40b4987baa6a653a21f6f33f902af6255f3"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.12"
weakdeps = ["Printf", "Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsPrintfExt = "Printf"
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["ChunkCodecLibZlib", "ChunkCodecLibZstd", "FileIO", "MacroTools", "Mmap", "OrderedCollections", "PrecompileTools", "ScopedValues"]
git-tree-sha1 = "da2e9b4d1abbebdcca0aa68afa0aa272102baad7"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.6.2"

    [deps.JLD2.extensions]
    UnPackExt = "UnPack"

    [deps.JLD2.weakdeps]
    UnPack = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "PrecompileTools", "StructTypes", "UUIDs"]
git-tree-sha1 = "411eccfe8aba0814ffa0fdf4860913ed09c34975"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.14.3"

    [deps.JSON3.extensions]
    JSON3ArrowExt = ["ArrowTypes"]

    [deps.JSON3.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.LazyArrays]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra", "MacroTools", "SparseArrays"]
git-tree-sha1 = "85829152db633948b418181ec33b8badaece9c3e"
uuid = "5078a376-72f3-5289-bfd5-ec5146d43c02"
version = "2.9.0"
weakdeps = ["BandedMatrices", "BlockArrays", "BlockBandedMatrices", "StaticArrays"]

    [deps.LazyArrays.extensions]
    LazyArraysBandedMatricesExt = "BandedMatrices"
    LazyArraysBlockArraysExt = "BlockArrays"
    LazyArraysBlockBandedMatricesExt = "BlockBandedMatrices"
    LazyArraysStaticArraysExt = "StaticArrays"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LazyBandedMatrices]]
deps = ["ArrayLayouts", "BandedMatrices", "BlockArrays", "BlockBandedMatrices", "FillArrays", "LazyArrays", "LinearAlgebra", "MatrixFactorizations", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "6cf770e6443c9f8d91d5dd78de0336921df6976b"
uuid = "d7e5e226-e90b-4449-9968-0f923699bf6f"
version = "0.11.7"
weakdeps = ["InfiniteArrays"]

    [deps.LazyBandedMatrices.extensions]
    LazyBandedMatricesInfiniteArraysExt = "InfiniteArrays"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "282cadc186e7b2ae0eeadbd7a4dffed4196ae2aa"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.2.0+0"

[[deps.MPFR_jll]]
deps = ["Artifacts", "GMP_jll", "Libdl"]
uuid = "3a97d323-0669-5f0c-9066-3539efd106a3"
version = "4.2.1+0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MatrixFactorizations]]
deps = ["ArrayLayouts", "LinearAlgebra", "Printf", "Random"]
git-tree-sha1 = "3bb3cf4685f1c90f22883f4c4bb6d203fa882b79"
uuid = "a3b82374-2e81-5b9e-98ce-41277c0e4c87"
version = "3.1.3"
weakdeps = ["BandedMatrices"]

    [deps.MatrixFactorizations.extensions]
    MatrixFactorizationsBandedMatricesExt = "BandedMatrices"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.MicroMamba]]
deps = ["Pkg", "Scratch", "micromamba_jll"]
git-tree-sha1 = "011cab361eae7bcd7d278f0a7a00ff9c69000c51"
uuid = "0b3b1443-0f03-428d-bdfb-f27f9c1191ea"
version = "0.1.14"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.5+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pidfile]]
deps = ["FileWatching", "Test"]
git-tree-sha1 = "2d8aaf8ee10df53d0dfb9b8ee44ae7c04ced2b03"
uuid = "fa939f87-e72e-5be4-a000-7fc836dbe307"
version = "1.3.0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"

    [deps.Pkg.extensions]
    REPLExt = "REPL"

    [deps.Pkg.weakdeps]
    REPL = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "8329a3a4f75e178c11c1ce2342778bcbbbfa7e3c"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.71"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "OrderedCollections", "RecipesBase", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "972089912ba299fba87671b025cd0da74f5f54f7"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "4.1.0"

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsFFTWExt = "FFTW"
    PolynomialsMakieExt = "Makie"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"

    [deps.Polynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    MutableArithmetics = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "0f27480397253da18fe2c12a4ba4eb9eb208bf3d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PythonCall]]
deps = ["CondaPkg", "Dates", "Libdl", "MacroTools", "Markdown", "Pkg", "Serialization", "Tables", "UnsafePointers"]
git-tree-sha1 = "e6bba29a71959561b55e342e295c1f5c01af7027"
uuid = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
version = "0.9.29"

    [deps.PythonCall.extensions]
    CategoricalArraysExt = "CategoricalArrays"
    PyCallExt = "PyCall"

    [deps.PythonCall.weakdeps]
    CategoricalArrays = "324d7699-5711-5eae-9e2f-1d82baa6b597"
    PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"

[[deps.PythonPlot]]
deps = ["Colors", "CondaPkg", "LaTeXStrings", "PythonCall", "Sockets", "Test", "VersionParsing"]
git-tree-sha1 = "409884283434a04092ddf1d9594c22bc097d5d9a"
uuid = "274fc56d-3b97-40fa-a1cd-1b4a50311bf9"
version = "1.0.6"

[[deps.QuasiArrays]]
deps = ["ArrayLayouts", "DomainSets", "FillArrays", "LazyArrays", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "0f16de5c9d9984dd2f12fa1f7bc4ed7ca931c6e5"
uuid = "c4ea9172-b204-11e9-377d-29865faadc5c"
version = "0.13.1"

    [deps.QuasiArrays.extensions]
    QuasiArraysSparseArraysExt = "SparseArrays"
    QuasiArraysStatsBaseExt = "StatsBase"

    [deps.QuasiArrays.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecurrenceRelationshipArrays]]
deps = ["ArrayLayouts", "BandedMatrices", "FillArrays", "InfiniteArrays", "LazyArrays", "LinearAlgebra", "RecurrenceRelationships"]
git-tree-sha1 = "e73ec1eec60deea31b7282cd0e09cf19d07b56bf"
uuid = "b889d2dc-af3c-4820-88a8-238fa91d3518"
version = "0.1.3"

[[deps.RecurrenceRelationships]]
git-tree-sha1 = "aa0b5958764e974a6e8d52f5b2daf51b26ede1a2"
uuid = "807425ed-42ea-44d6-a357-6771516d7b2c"
version = "0.2.0"
weakdeps = ["FillArrays", "LazyArrays", "LinearAlgebra"]

    [deps.RecurrenceRelationships.extensions]
    RecurrenceRelationshipsFillArraysExt = "FillArrays"
    RecurrenceRelationshipsLazyArraysExt = "LazyArrays"
    RecurrenceRelationshipsLinearAlgebraExt = "LinearAlgebra"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "c3b2323466378a2ba15bea4b2f73b081e022f473"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.5.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.SemiseparableMatrices]]
deps = ["ArrayLayouts", "BandedMatrices", "BlockBandedMatrices", "LazyArrays", "LinearAlgebra", "MatrixFactorizations"]
git-tree-sha1 = "8e0d84d11d183c550084f6da61a4d613ff99da1f"
uuid = "f8ebbe35-cbfb-4060-bf7f-b10e4670cf57"
version = "0.4.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "f2685b435df2613e25fc10ad8c26dddb8640f547"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.6.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "b8693004b385c842357406e3af647701fe783f98"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.15"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "159331b30e94d7b11379037feeb9b690950cace8"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.ToeplitzMatrices]]
deps = ["AbstractFFTs", "DSP", "FillArrays", "LinearAlgebra"]
git-tree-sha1 = "338d725bd62115be4ba7ffa891d85654e0bfb1a1"
uuid = "c751599d-da0a-543b-9d20-d0a503d91d24"
version = "0.8.5"

    [deps.ToeplitzMatrices.extensions]
    ToeplitzMatricesStatsBaseExt = "StatsBase"

    [deps.ToeplitzMatrices.weakdeps]
    StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[[deps.Tricks]]
git-tree-sha1 = "372b90fe551c019541fafc6ff034199dc19c8436"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.12"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnsafePointers]]
git-tree-sha1 = "c81331b3b2e60a982be57c046ec91f599ede674a"
uuid = "e17b2a0c-0bdf-430a-bd0c-3a23cae4ff39"
version = "1.0.0"

[[deps.VersionParsing]]
git-tree-sha1 = "58d6e80b4ee071f5efd07fda82cb9fbe17200868"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.3.0"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.micromamba_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "2ca2ac0b23a8e6b76752453e08428b3b4de28095"
uuid = "f8abcde7-e9b7-5caa-b8af-a437887ae8e4"
version = "1.5.12+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "1350188a69a6e46f799d3945beef36435ed7262f"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.0.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.pixi_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "f349584316617063160a947a82638f7611a8ef0f"
uuid = "4d7b5844-a134-5dcd-ac86-c8f19cd51bed"
version = "0.41.3+0"
"""

# ╔═╡ Cell order:
# ╟─61f6b454-deb5-45ca-bce7-6749fddd0816
# ╟─62f2a254-1ec9-4f68-ad57-8d58df909670
# ╠═b34451a8-38d9-4267-8114-46fdf5ee69b2
# ╠═b1aa93b4-831b-4440-a71f-f6648082708b
# ╠═570be094-6ea9-11f0-1c4e-55c239b84740
# ╟─903eff0e-6642-4f71-9d15-0f3b239c0594
# ╠═3b18d124-92ca-4b6c-8bf0-a9151c1977e4
# ╠═4312571a-39c8-41e4-a0bb-114831849dd1
# ╠═9dc73e37-50bc-45f8-9d97-9848950ff689
# ╠═21f4cd71-4b29-497a-a66d-4482cd2b163d
# ╟─ecfba73d-a1f2-4867-8237-fb2719bfd7a6
# ╠═b4d1900c-9d84-4459-925c-f23c4b6e4b7c
# ╠═f222ba3d-8d59-475d-b387-35777fdbacfe
# ╟─2dd3a3f1-94a9-4659-84f9-8d30877a493d
# ╟─3bcaa492-2517-4d63-8240-529c98c2f0aa
# ╠═099e92bd-31dd-4590-b9eb-36e6fcb0a1ee
# ╠═54a94ad1-bee1-4331-98df-e5e07ab7d8f5
# ╠═6759d423-a369-44f8-838d-3fd1b8af241f
# ╟─adcd347e-5aa7-409e-aecd-069a11f4485c
# ╠═4e674238-1bf1-4984-979c-15e9f71d4d6c
# ╠═df8290e5-000a-4803-a94e-26e1dd541a8b
# ╟─abaeab70-4dc4-4262-857a-59df89a443f6
# ╟─0ecd3570-b18d-4265-9139-cdab8fa9f441
# ╟─888b97ee-7dac-4369-bc81-e9de57b7a4b8
# ╠═5064021b-787e-4464-b449-a551d20f3646
# ╟─a70e031d-e8d8-41fe-b315-5c5e450187e5
# ╠═3d3d9b35-a444-41e1-8acc-481bc4f1aa33
# ╠═7220c410-be31-450c-811b-dd544d1c62c0
# ╠═8da4e3f6-f85f-4115-83c6-e852c4963b43
# ╠═a726c9f8-d12e-4134-b7b7-52d334fb30e6
# ╟─5fe4285d-a238-4bec-8f7a-1d9fbb53300e
# ╠═a8eb020a-1783-4478-81c3-532a81e9868a
# ╠═542e7613-3233-4b47-ac35-12ed5ad2b2d1
# ╠═3cb58227-5ba3-4943-a59c-765673ee8784
# ╟─48dac3f6-bd91-44b7-af6f-ead473f4279e
# ╠═668e2a3d-0972-4e1d-a0ce-67683ae3d670
# ╠═17dbf805-f582-4bee-8d6c-34d2e22f356a
# ╟─71a4462a-785c-4b0b-a29c-aa50c91b386f
# ╠═2033a8e1-c74c-4671-a72b-707befacf6c4
# ╠═083e42e4-e1d2-4472-9a92-20744f289b56
# ╠═193ece9c-6c93-4ea4-a091-d39ba8887901
# ╠═2f15acac-67a5-4974-a3f6-0296c117a142
# ╠═2991bb15-229f-4fff-9162-0805b01e679d
# ╠═8a4072af-e549-4ed6-9a76-16613851e8ab
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
