export DosFunction, Gauss, compute_dos_no_shift, compute_dos_shift, compute_ldos_kpm, compute_dos_shift_kpm

abstract type DosFunction end

# g(x) = e^{-(x)^2/2σ^2} / σ√2π 
struct Gauss <: DosFunction
    σ::T where {T<:Real}
end

evalf(x, μ, GF::Gauss) = exp(-(x - μ)^2 / (2GF.σ^2)) / (sqrt(2pi) * GF.σ)
#------------------------------------------------
# computed directly by eigen pairs
#------------------------------------------------
# no k-point shifiting
# g : semearing function
# \hat{Tr}_{W,L}(g(H^{W,L})) = pre_fac * Tr(g(H^{W,L}))
# pre_fac = |Γ_1^*||Γ_2^*|/SdL
function compute_dos_no_shift(ϵ, smearf::DosFunction,
    basis::BasisLW, ERange::Float64, n_eigs::Int64)

    @assert basis.nk == 1 && norm(basis.kpts[1], 1) == 0.0
    H = hamFull(basis)[1]
    E, U = eigsolve(H, n_eigs, EigSorter(x -> abs(x - ERange); rev=false); krylovdim=n_eigs + 20)

    pre_fac = prod(basis.model.latR_unit_vol) / (basis.SdL * 2pi)
    g(x, μ) = evalf(x, μ, smearf)
    dos = pre_fac .* [sum(g.(E, ϵi)) for ϵi in ϵ]
end

function compute_ldos(ϵ, smearf::DosFunction, basis::BasisLW; ERange=0, n_eigs=20)

    npw = basis.npw
    nk = basis.nk
    G0ind = basis.Gmap12[basis.G1max+1, basis.G2max+1]
    ldos = zeros(length(ϵ), nk)
    g(x, μ) = evalf(x, μ, smearf)
    HV = ham_Potential(basis)

    for k = 1:nk
        Hk = hamK(basis, k, HV)
        Ek, ψk = eigsolve(Hk, n_eigs, EigSorter(x -> abs(x - ERange); rev=false); krylovdim=n_eigs + 20)

        # g(H^{W,L}(q))_{0,0} = \s um_j g(λ_j)|ψ_j|_0 ^2
        pw0k = abs2.([ψkj[G0ind] for ψkj in ψk])
        for (i, ϵi) in enumerate(ϵ)
            ldos[i, k] = dot(g.(Ek, ϵi), pw0k)
        end
    end

    ldos ./ 2pi
end
# \tilde{Tr}^h_{W,L}(g(H^{W,L})) = pre_fac * \sum_{q\in K^W_h} g(H^{W,L}(q))_{0,0}
# K^W_h : a uniform quadrature mesh on [-W,W]^dim  
# pre_fac = h^dim  with  h = W/K
function compute_dos_shift(ϵ, smearf::DosFunction, model::TBG1D, EcutL::T, EcutW::T, n_eigs::Int64, K::Int64; ERange=0.0) where {T<:Real}

    h = EcutW / K
    xx = collect(range(-EcutW, EcutW, length=2K))
    basis = basisGen(EcutL, EcutW, model, xx)
    HV = ham_Potential(basis)

    npw = basis.npw
    G0ind = basis.Gmap12[basis.G1max+1, basis.G2max+1]
    ldos = zeros(length(ϵ))
    for k = 1:2K
        Hk = hamK(basis, k, HV)
        Ek, ψk = eigsolve(Hk, n_eigs, EigSorter(x -> abs(ERange - x); rev=false); krylovdim=n_eigs + 50)

        # g(H^{W,L}(q))_{0,0} = \s um_j g(λ_j)|ψ_j|_0 ^2
        pw0k = abs2.([ψkj[G0ind] for ψkj in ψk])
        g(x, μ) = evalf(x, μ, smearf)
        ldos += [dot(g.(Ek, ϵi), pw0k) for ϵi in ϵ]
    end

    ldos .* h ./ 2pi
end

compute_dos_shift(ϵ, smearf::DosFunction, basis::BasisLW, n_eigs::Int64, K::Int64; kwargs...) = compute_dos_shift(ϵ, smearf, basis.model, basis.EcutL, basis.EcutW, n_eigs, K; kwargs...)

#------------------------------------------------
# computed by Chebyshev Polynomials (KPM)
#------------------------------------------------
abstract type ChebyshevMethod end

struct KPM <: ChebyshevMethod end
struct JacksonKPM <: ChebyshevMethod end

function genCheb(M::Int64, f::Function, pt, ::KPM)
    # f(x) = \sum_{n = 0}^{M}a_n*T_n(x) 

    fv = fft(f.(pt))
    coef = real.(fv[1:M+1]) * 2 / length(pt)
    coef[1] /= 2

    coef
end

function genCheb(M::Int64, f::Function, pt, ::JacksonKPM)
    # f(x) = \sum_{n = 0}^{M}a_n*T_n(x) 

    fv = fft(f.(pt))
    coef = real.(fv[1:M+1]) * 2 / length(pt)
    coef[1] /= 2

    aM = pi / (M + 2)
    g(m) = ((1 - m / (M + 2))sin(aM)cos(m * aM) + (1 / (M + 2))cos(aM)sin(m * aM)) / sin(aM)
    @. coef = coef * g(0:M)

    return coef
end

function genCheb(M::Int64, σ::T, pt, Ept, E1::Float64, E2::Float64, ::KPM; tol=1e-6) where {T<:Real}
    # f(x) = \sum_{n = 0}^{M}a_n*T_n(x) 
    σ = σ / E2
    g(x, E) = exp(-(x - E)^2 / (2σ^2)) / (sqrt(2pi) * σ)
    Es = (Ept .- E1) ./ E2
    coef = zeros(length(Ept), M + 1)
    cM = (M + 1) * ones(Int, length(Ept))
    for (i, Esi) in enumerate(Es)
        @views coef[i, :] = real.(fft(g.(pt, Esi))[1:M+1]) .* 2 ./ length(pt)
        coef[i, 1] /= 2
        @views ciM = findlast(x -> abs(x) > tol, coef[i, :])
        if ciM != nothing
            cM[i] = ciM
        end
    end
    Mmax = maximum(cM)

    return Mmax - 1, coef[:, 1:Mmax]
end

function genCheb(M::Int64, σ::T, pt, Ept, E1::Float64, E2::Float64, ::JacksonKPM; tol=1e-6) where {T<:Real}
    # f(x) = \sum_{n = 0}^{M}a_n*T_n(x) 
    function JacksonDamping(M)
        aM = pi / (M + 2)
        g(m) = ((1 - m / (M + 2))sin(aM)cos(m * aM) + (1 / (M + 2))cos(aM)sin(m * aM)) / sin(aM)
        g.(0:M)
    end

    σ = σ / E2
    g(x, E) = exp(-(x - E)^2 / (2σ^2)) / (sqrt(2pi) * σ)
    Es = (Ept .- E1) ./ E2
    coef = zeros(M + 1, length(Ept))
    for (i, Esi) in enumerate(Es)
        coef[:, i] = real.(fft(g.(pt, Esi))[1:M+1]) .* 2 ./ length(pt)
        coef[1, i] /= 2
    end
    coefJD = coef .* JacksonDamping(M)

    cM = (M + 1) * ones(Int, length(Ept))
    for i = 1:length(Ept)
        ciM = findlast(x -> abs(x) > tol, coefJD[:, i])
        if ciM != nothing
            cM[i] = ciM
        end
    end

    Mmax = maximum(cM)
    newcoefJD = (coef[1:Mmax, :] .* JacksonDamping(Mmax - 1))'

    return Mmax - 1, newcoefJD
end

# T_n(H(q))_{0,0} = e_0^T * T_n(H(q)) * e_0
# T_{n+1}(H)= 2H*T_n(H) - T_{n-1}(H)
function compute_TH0(M::Int64, H::SparseMatrixCSC, ind0::Int64)

    TH0 = zeros(M + 1)
    TH0[1] = 1.0

    u0 = zeros(H.m)
    u0[ind0] = 1.0
    u1 = H * u0
    TH0[2] = u1[ind0]
    u2 = similar(u1)
    for k = 3:M+1
        mul!(u2, H, u1)
        @. u2 = 2.0 * u2 - u0
        TH0[k] = u2[ind0]

        copy!(u0, u1)
        copy!(u1, u2)
    end

    TH0
end

function compute_ldos_kpm(ϵ, smearf::DosFunction, basis::BasisLW; M=Int(1e5), Npt=Int(round(1.1M)), tol=1e-6, method=KPM())

    HV = ham_Potential(basis)
    vmin = eigsolve(HV, 1, :SR)[1][1]
    vmax = eigsolve(HV, 1, :LR)[1][1]

    npw = basis.npw
    nk = basis.nk
    G0ind = basis.Gmap12[basis.G1max+1, basis.G2max+1]
    ldos = zeros(length(ϵ), nk)

    pt = cos.(range(0, 2pi - pi / Npt, length=2Npt))
    ck = zeros(size(ldos, 1))
    E1 = 0
    E2 = 0
    Mk = 0
    coef = Matrix{Float64}[]
    for (k, kpt) in enumerate(basis.kpts)
        (k % 50 == 0 || k == nk) && println(" $(k) / $(nk) ")

        if (kpt - basis.kpts[1]) % 5.0 == 0.0
            # Elb = λmin(HV),  Eup = 0.5 * (|k| + W)^2 + λmax(HV)
            E1 = (0.5 * (abs(kpt) + basis.EcutW)^2 + vmax + vmin) / 2
            E2 = (0.5 * (abs(kpt) + basis.EcutW)^2 + vmax - vmin) / 2
            Mk, coef = genCheb(M, smearf.σ, pt, ϵ, E1, E2, method; tol=tol)
            @show Mk
        end

        Hk = hamK(basis, k, HV)
        Hks = (Hk - E1 * I) / E2
        THk0 = compute_TH0(Mk, Hks, G0ind)

        mul!(ck, coef, THk0)
        @views ldos[:, k] = ck ./ E2
    end

    ldos ./ 2pi
end


function compute_dos_shift_kpm(ϵ, smearf::DosFunction, model::TBG1D, EcutL::T, EcutW::T, h::Float64; M=Int(1e5), Npt=Int(round(1.1M)), tol=1e-6, method=KPM(), Ktrunc=EcutW) where {T<:Real}

    hgrid = Int(round(5 / h))
    xx = collect(0:h:Ktrunc)[2:end]
    basis = basisGen(EcutL, EcutW, model, xx)

    HV = ham_Potential(basis)
    vmin = eigsolve(HV, 1, :SR)[1][1]
    vmax = eigsolve(HV, 1, :LR)[1][1]

    npw = basis.npw
    nk = basis.nk
    G0ind = basis.Gmap12[basis.G1max+1, basis.G2max+1]
    ldos = zeros(length(ϵ))

    pt = cos.(range(0, 2pi - pi / Npt, length=2Npt))
    ck = zero(ldos)
    E1 = 0
    E2 = 0
    Mk = 0
    coef = Matrix{Float64}[]
    for (k, kpt) in enumerate(basis.kpts)
        (k % 50 == 0 || k == nk) && println(" $(k) / $(nk) ")

        if k % hgrid == 1
            # Elb = λmin(HV),  Eup = 0.5 * (|k| + W)^2 + λmax(HV)
            E1 = (0.5 * (abs(kpt) + basis.EcutW)^2 + vmax + vmin) / 2
            E2 = (0.5 * (abs(kpt) + basis.EcutW)^2 + vmax - vmin) / 2
            Mk, coef = genCheb(M, smearf.σ, pt, ϵ, E1, E2, method; tol=tol)
            @show Mk
        end

        Hk = hamK(basis, k, HV)
        Hks = (Hk - E1 * I) / E2
        THk0 = compute_TH0(Mk, Hks, G0ind)

        # g(H^{W,L}(q))_{0,0} = \sum_m cm*Tm(H(q))_{0,0}
        mul!(ck, coef, THk0)
        ldos += ck ./ E2
    end

    2.0 * ldos .* h ./ 2pi
end
compute_dos_shift_kpm(ϵ, smearf::DosFunction, basis::BasisLW, h::Float64; kwargs...) = compute_dos_shift_kpm(ϵ, smearf, basis.model, basis.EcutL, basis.EcutW, h; kwargs...)