export Basis

struct Kpoint{T<:Integer}
    coordinate::Real

    Gmap12::SparseMatrixCSC{T,T}
    Gmap21::SparseMatrixCSC{T,T}

    G_vec::Vector{SVector{2,Int64}}
    G_cart_sum::Vector{Float64}

    npw::T
    G0_index::T
end

"""
Modified planewave cutoff
EcutL : cutoff for ergodicity
EcutW : cutoff for frequency
npw : size of planewave basis
Gimax : max index of the reciprocal lattices of sheet i
Gi : full reciprocal lattices of sheet i
G : reciprocal index of two sheets, i.e., G = (G1,G2)
Gmn : sum of reciprocal lattice i.e., G1 + G2
Gmapij : sorting index of (Gi,Gj)
nk : number of k-points
kpts : k-points
SdL : the volume of a d-simensional ball with diameter L
model : TBG1D
"""
struct Basis{T<:Integer}
    EcutL::Real
    EcutW::Real

    G1::Vector{Float64}
    G2::Vector{Float64}

    nk::T
    kpoints::Vector{Kpoint{T}}

    SdL::Float64
    model::TBG1D
end

function Kpoint(kpt, EcutL, EcutW, G_sheet_vec, G_sheet_cart)

    count = 0
    G_vec = SVector{2,Int64}[]
    G_cart_sum = Float64[]
    ind1 = Int64[]
    ind2 = Int64[]
    val = Int64[]
    for (t1, Gt1) in enumerate(G_sheet_cart[1])
        for (t2, Gt2) in enumerate(G_sheet_cart[2])
            if abs(kpt + Gt1 + Gt2) <= EcutW && abs(Gt1 - Gt2) <= EcutL
                count += 1
                push!(G_vec, SA[G_sheet_vec[1][t1], G_sheet_vec[2][t2]])
                push!(G_cart_sum, Gt1 + Gt2)
                push!(ind1, t1)
                push!(ind2, t2)
                push!(val, count)
            end
        end
    end
    Gmap12 = sparse(ind1, ind2, val)
    Gmap21 = sparse(ind2, ind1, val)
    npw = length(G_cart_sum)
    G0_index = findfirst(iszero, G_vec)
   
    Kpoint(kpt,Gmap12,Gmap21,G_vec,G_cart_sum,npw,G0_index)
end

function Basis(EcutL::Real, EcutW::Real, 
               model::TBG1D, kpts::Vector{T}) where {T<:Real}

    latR = model.latR
    dim = size(latR[1],1)
    SdL = (sqrt(pi) * EcutL) ^ dim * gamma(dim/2 + 1)

    n_fftw = 1 .+ 4 .* ceil.(Int, 0.5*(EcutL+EcutW) ./ [norm(latR[j]) for j = 1:2])
    G_sheet_vec = G_vectors.(n_fftw)
    G_sheet_cart = [latR[j] .* G_sheet_vec[j] for j = 1:2]

    nk = length(kpts)
    kpoints = [Kpoint(x, EcutL, EcutW, G_sheet_vec, G_sheet_cart) for x in kpts]

    Basis(EcutL, EcutW, G_sheet_cart[1], G_sheet_cart[2], nk, kpoints, SdL, model)
end

Basis(EcutL::Real, EcutW::Real, model::TBG1D; kpts=zeros(1)) = Basis(EcutL, EcutW, model, kpts)

function G_vectors(fft_size::Int)
    start = -cld(fft_size - 1, 2)
    stop = fld(fft_size - 1, 2)
    G_vectors = vcat(0:stop, start:-1)
end