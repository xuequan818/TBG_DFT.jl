export DosFunction, Gauss, compute_dos_no_shift, compute_dos_shift, compute_ldos_kpm, compute_dos_shift_kpm

abstract type DosFunction end

# g(x) = e^{-(x)^2/2σ^2} / σ√2π 
struct Gauss <: DosFunction
    σ::T where {T<:Real}
end

evalf(x, μ, GF::Gauss) = exp(-(x - μ)^2 / (2GF.σ^2)) / (sqrt(2pi)*GF.σ)
#------------------------------------------------
# computed directly by eigen pairs
#------------------------------------------------
# no k-point shifiting
# g : semearing function
# \hat{Tr}_{W,L}(g(H^{W,L})) = pre_fac * Tr(g(H^{W,L}))
# pre_fac = |Γ_1^*||Γ_2^*|/SdL
function compute_dos_no_shift(ϵ, smearf::DosFunction,
							basis::BasisLW, ERange::Float64, n_eigs::Int64)

    H = hamiltonian(basis)
    E, U = eigsolve(H, n_eigs, EigSorter(x -> abs(x-ERange); rev=false); krylovdim= n_eigs+50)

	pre_fac = prod(basis.model.latR_unit_vol) / basis.SdL
    g(x,μ) = evalf(x,μ,smearf)
    dos = pre_fac .* [sum(g.(E,ϵi)) for ϵi in ϵ]
end

# \tilde{Tr}^h_{W,L}(g(H^{W,L})) = pre_fac * \sum_{q\in K^W_h} g(H^{W,L}(q))_{0,0}
# K^W_h : a uniform quadrature mesh on [-W,W]^dim  
# pre_fac = h^dim  with  h = W/K
function compute_dos_shift(ϵ, smearf::DosFunction, model::TBG1D, EcutL::T, EcutW::T, n_eigs::Int64, K::Int64; ERange=0.) where {T<:Real}

	h = EcutW / K
    xx = collect(range(-EcutW, EcutW, length=2K))

    basis = basisGen(EcutL, EcutW, model, xx)
	H = hamiltonian(basis)

	npw = basis.npw
	G0ind = basis.Gmap12[basis.G1max+1,basis.G2max+1]
	ldos = zeros(length(ϵ))
	for k = 1:2K
		krange = 1 + (k-1)*npw : k*npw
		Hk = H[krange,krange]
		Ek, ψk = eigsolve(Hk, n_eigs, EigSorter(x -> abs(ERange - x); rev=false); krylovdim=n_eigs + 50)

		# g(H^{W,L}(q))_{0,0} = \sum_j g(λ_j)|ψ_j|_0 ^2
		pw0k = abs2.([ψkj[G0ind] for ψkj in ψk])
        g(x, μ) = evalf(x, μ, smearf)
        ldos += [dot(g.(Ek, ϵi), pw0k) for ϵi in ϵ]
	end

	ldos .* h
end

compute_dos_shift(ϵ, smearf::DosFunction, basis::BasisLW, n_eigs::Int64, K::Int64; ERange=0.0) = compute_dos_shift(ϵ, smearf, basis.model, basis.EcutL, basis.EcutW, n_eigs, K)

#------------------------------------------------
# computed by Chebyshev Polynomials (KPM)
#------------------------------------------------
"""
KPM
M : the M-th Chebyshev polynomial
coef : coefficients for the Chebyshev expansion of function f
"""
struct KPM
    M::Int64
    coef::Array{Float64,1}
end

function genKPM(M::Int64, f::Function, pt)
    # f(x) = \sum_{n = 0}^{M}a_n*T_n(x) 

    fv = fft(f.(pt))
    coef = real.(fv[1:M+1]) * 2 / length(pt)
	coef[1] /= 2
    
	coef
end

function genKPM_Jackson(M::Int64, f::Function, pt)
    # f(x) = \sum_{n = 0}^{M}a_n*T_n(x) 

    fv = fft(f.(pt))
    coef = real.(fv[1:M+1]) * 2 / length(pt)
    coef[1] /= 2

    aM = pi / (M + 2)
    g(m) = ((1 - m / (M + 2))sin(aM)cos(m * aM) + (1 / (M + 2))cos(aM)sin(m * aM)) / sin(aM)
    @. coef = coef * g(0:M)

    return coef
end

# T_n(H(q))_{0,0} = e_0^T * T_n(H(q)) * e_0
# T_{n+1}(H)= 2H*T_n(H) - T_{n-1}(H)
function compute_TH0(M::Int64, H::SparseMatrixCSC, ind0::Int64)

	TH0 = zeros(M+1)
	TH0[1] = 1.

    u0 = zeros(H.m)
	u0[ind0] = 1.
    u1 = H * u0
	TH0[2] = u1[ind0]
	u2 = similar(u1)
    for k = 3:M+1
        mul!(u2, H, u1)
        @. u2 = 2.0 * u2 - u0
        TH0[k] = u2[ind0]

        u0 = copy(u1)
        u1 = copy(u2)
    end

	TH0
end

function compute_ldos_kpm(ϵ, smearf::DosFunction, basis::BasisLW, M::Int64; Npt=Int(round(1.1M)))

    H = hamiltonian(basis)

    npw = basis.npw
    nk = basis.nk
    G0ind = basis.Gmap12[basis.G1max+1, basis.G2max+1]
    ldos = zeros(length(ϵ),nk)

    pt = cos.(range(0, 2pi - pi / Npt, length=2Npt))
    for k = 1:nk
        krange = 1+(k-1)*npw:k*npw
        Hk = H[krange, krange]

        Emin, Umin = eigsolve(Hk, 1, :SR)
        Emax, Umax = eigsolve(Hk, 1, :LR)
        Elb = real.(Emin[1]) - 0.1
        Eub = real.(Emax[1]) + 0.1
        E1 = (Elb + Eub) / 2
        E2 = (Eub - Elb) / 2
        Hks = (Hk - E1 * I) / E2
        THk0 = compute_TH0(M, Hks, G0ind)
        GFs = Gauss(smearf.σ / E2)

        # g(H^{W,L}(q))_{0,0} = \sum_m cm*Tm(H(q))_{0,0} = (c, T(H(q))_{0,0})
        for (i, ϵi) in enumerate(ϵ)
            Esi = (ϵi - E1) / E2
            gsi(x) = evalf(x, Esi, GFs)
            coefi = genKPM(M, gsi, pt)
            ldos[i, k] = dot(coefi, THk0) / E2
        end
    end

    ldos
end


function compute_dos_shift_kpm(ϵ, smearf::DosFunction, model::TBG1D, EcutL::T, EcutW::T, K::Int64, M::Int64; Npt = Int(round(1.1M))) where {T<:Real}

	
    h = EcutW / K
    xx = collect(range(-EcutW, EcutW, length=2K))

    basis = basisGen(EcutL, EcutW, model, xx)
    H = hamiltonian(basis)

    npw = basis.npw
    G0ind = basis.Gmap12[basis.G1max+1, basis.G2max+1]
    ldos = zeros(length(ϵ))

    pt = cos.(range(0, 2pi - pi / Npt, length=2Npt))
    for k = 1:2K
        krange = 1+(k-1)*npw:k*npw
        Hk = H[krange, krange]

        Emin, Umin = eigsolve(Hk, 1, :SR)
        Emax, Umax = eigsolve(Hk, 1, :LR)
        Elb = real.(Emin[1]) - 0.1
        Eub = real.(Emax[1]) + 0.1
        E1 = (Elb + Eub) / 2
        E2 = (Eub - Elb) / 2
        Hks = (Hk - E1 * I) / E2
		THk0 = compute_TH0(M,Hks,G0ind)
		GFs = Gauss(smearf.σ/E2)

        # g(H^{W,L}(q))_{0,0} = \sum_m cm*Tm(H(q))_{0,0}
		for (i, ϵi) in enumerate(ϵ)
			Esi = (ϵi-E1)/E2
            gsi(x) = evalf(x, Esi, GFs) 
			coefi = genKPM(M,gsi,pt)
			ldos[i] += dot(coefi,THk0) / E2
		end
	end

    ldos .* h
end

compute_dos_shift_kpm(ϵ, smearf::DosFunction, basis::BasisLW, K::Int64, M::Int64; Npt = Int(round(1.1M))) = compute_dos_shift_kpm(ϵ, smearf, basis.model, basis.EcutL, basis.EcutW, K, M; Npt = Npt)