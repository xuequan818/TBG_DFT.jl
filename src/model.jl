# incommensurate structures: 
# incommStruc for toy model

export Gaussian, TBG1D, TbgToy
"""TBG1D
lat : lattice vectors for periodic cells 
latR : lattice vectors for reciprocal lattices
Lz : distance in z-direction
vft : fourier transform of external potentials
lat_unit_vol : unit cell volume for lattice
latR_unit_vol : unit cell volume for reciprocal lattice
"""
struct TBG1D{T1,T2} <: Real
    lat::Vector{T1}
    latR::Vector{T1}
    Lz::T1
    vft::Vector{<:Function}
    lat_unit_vol::Vector{T2}
    latR_unit_vol::Vector{T2}
end

function TBG1D(lat::Vector{T}, Lz::T, vft::Vector{<:Function}) where {T}
    @assert length(lat) == 2
    lat_unit_vol = [det(latj) for latj in lat]
    latR = [2π / latj for latj in lat]
    latR_unit_vol = [det(latRj) for latRj in latR]

    return TBG1D(lat, latR, Lz, vft, lat_unit_vol, latR_unit_vol)
end
TBG1D(lat::Vector{T}, vft::Vector{<:Function}; Lz=zero(T)) where {T} = TBG1D(lat, Lz, vft)

# g(x) = Ae^{-x^2/2σ^2} / σ√2π 
struct Gaussian{T1,T2} <: Real
    A::T1
    σ::T2
end 

function TbgToy(L::Real, ϵ::Real, g::Vector{<:Gaussian})
	lat = [L, L/(1+ϵ)]
	Lz = 0.

    F(σ, q) = exp(-(q * σ)^2 / 2)
    v1(q) = -(g[1].A / L) * F(g[1].σ, q[1])
    v2(q) = -(g[2].A / L) * F(g[2].σ, q[1]/(1+ϵ))

    TBG1D(lat, Lz, [v1,v2])
end
