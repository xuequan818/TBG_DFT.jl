export compute_density_vec, compute_density, compute_density_eig

# T_n(H(q))_{:,G} = T_n(H(q)) * e_0
# T_{n+1}(H)= 2H*T_n(H) - T_{n-1}(H)
function compute_TH0_col(M::Int64, H::SparseMatrixCSC, ind0::Int64)

    TH0 = zeros(M + 1, H.m)

    u0 = zeros(H.m)
    u0[ind0] = 1.0
    TH0[1, :] = u0
    u1 = H * u0
    TH0[2, :] = u1
    u2 = similar(u1)
    for k = 3:M+1
        mul!(u2, H, u1)
        @. u2 = 2.0 * u2 - u0
        TH0[k, :] = u2

        copy!(u0, u1)
        copy!(u1, u2)
    end

    TH0
end

function compute_density_vec(ϵ, smearf::DosFunction,
    					 model::TBG1D, EcutL::T,
    					 EcutW::T, h::Float64;
    					 M=Int(1e5), Npt=Int(round(1.1M)),
    					 tol=1e-6, Ktrunc=EcutW,
    					 kwidth=5.0, method=KPM(),
                         lb_fac=0.2, ub_fac=0.2) where {T<:Real}

    # Note that the LDoS of this toy model has symmetry about 0
    # we just compute ξ ∈ [0,Ktrunc]

    xx = collect(0:h:Ktrunc)
    basis = Basis(EcutL, EcutW, model, xx)

    HV = ham_Potential(basis, 1)
    vmin = real(eigsolve(HV, 1, :SR)[1][1])
    vmin = vmin - lb_fac * abs(vmin)
    vmax = real(eigsolve(HV, 1, :LR)[1][1])
    vmax = vmax + ub_fac * abs(vmax)
    E1 = (0.5 * basis.EcutW^2 + vmax + vmin) / 2
    E2 = (0.5 * basis.EcutW^2 + vmax - vmin) / 2

    pt = cos.(range(0, 2pi - pi / Npt, length=2Npt))
    newM, coef = genCheb(M, smearf.σ, pt, ϵ, E1, E2, method; tol)
    println(" M = $(newM)")

    nk = basis.nk
    rho_vec = Vector{AbstractArray}(undef, nk)
    Folds.foreach(1:nk, WorkStealingEx()) do ik
        G0ind = basis.kpoints[ik].G0_index
        Hk = ham(basis, ik)
        Hks = (Hk - E1 * I) / E2   
        THk0_col = compute_TH0_col(newM, Hks, G0ind)

        # g(H^{W,L}(q))_{0,G} = \sum_m cm*Tm(H(q))_{0,G}
        ck = coef * THk0_col
        if 1 < ik < nk
            ck = 2 .* ck ./ E2
        else
            ck = ck ./ E2
        end
        rho_vec[ik] = transpose(ck)
    end

    rho_vec .* h ./ 2pi
end
compute_density_vec(ϵ, smearf::DosFunction, basis::Basis, h::Float64; kwargs...) = compute_density_vec(ϵ, smearf, basis.model, basis.EcutL, basis.EcutW, h; kwargs...)

function compute_density_eig(ϵ, smearf::DosFunction, model::TBG1D, EcutL::T, EcutW::T, h::Float64, n_eigs::Int64; Ktrunc=EcutW, ERange=0.5 * (minimum(ϵ) + maximum(ϵ))) where {T<:Real}

    xx = collect(0:h:Ktrunc)
    basis = Basis(EcutL, EcutW, model, xx)
    g(x, μ) = evalf(x, μ, smearf)

    nk = basis.nk
    rho_vec = Vector{AbstractArray}(undef, nk)
    for k = 1:nk
        G0ind = basis.kpoints[k].G0_index
        rho_veck = zeros(basis.kpoints[k].npw, length(ϵ))

        Hk = ham(basis, k)
        Ek, ψk = eigsolve(Hk, n_eigs, EigSorter(x -> abs(ERange - x); rev=false); krylovdim=n_eigs + 20)

        # g(H^{W,L}(q))_{0,0} = \s um_j g(λ_j)|ψ_j|_0 ^2
        pw0k = ([ψkj[G0ind] * conj.(ψkj) for ψkj in ψk])
        if 1 < k < nk
			for (i,ϵi) in enumerate(ϵ)
                rho_veck[:, i] += 2 * sum(g.(Ek, ϵi) .* pw0k)
			end
        else
            for (i, ϵi) in enumerate(ϵ)
                rho_veck[:, i] += sum(g.(Ek, ϵi) .* pw0k)
            end
        end
        rho_vec[k] = rho_veck
    end

    rho_vec .* h ./ 2pi
end
compute_density_eig(ϵ, smearf::DosFunction, basis::Basis, h::Float64, n_eigs::Int64; kwargs...) = compute_density_eig(ϵ, smearf, basis.model, basis.EcutL, basis.EcutW, h, n_eigs; kwargs...)


function compute_density(x, rho_vec, basis::Basis) 
    val = zeros(length(x), size(rho_vec[1],2))
    for (ik, kpt) in enumerate(basis.kpoints)
        val += real.(exp.(-im .* x * kpt.G_cart_sum') * rho_vec[ik])
    end
    val
end