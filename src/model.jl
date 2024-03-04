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
struct TBG1D{T <: Real}
    lat::Vector{T}
    latR::Vector{T}
    Lz::T
    vft::Vector{Function}
    lat_unit_vol::Vector{Float64}
    latR_unit_vol::Vector{Float64}
end

function TBG1DGen(lat::Vector{T}, Lz::T, vft::Vector) where {T <: Real}
    @assert length(lat) == 2
    lat_unit_vol = [det(latj) for latj in lat]
    latR = [2π / latj for latj in lat]
    latR_unit_vol = [det(latRj) for latRj in latR]


    return TBG1D(lat, latR, Lz, vft, lat_unit_vol, latR_unit_vol)
end

# g(x) = Ae^{-x^2/2σ^2} / σ√2π 
struct Gaussian
    A::T where {T<:Real}
    σ::T where {T<:Real}
end 

function TbgToy(L::T1, ϵ::T2, g::Vector{Gaussian}) where {T1<:Real, T2<:Real} 
	lat = [L, L/(1+ϵ)]
	Lz = 0.
    latR = 2pi/L

    F(σ,q) = exp(-q^2/(2σ^2))
    v1(q) = -(g[1].A / L) * F(g[1].σ, q[1])
    v2(q) = -(g[2].A / L) * F(g[2].σ, q[1]/(1+ϵ))

    TBG1DGen(lat, Lz, [v1,v2])
end
