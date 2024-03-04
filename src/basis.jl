export Basis

abstract type Basis end

"""
standard planewave cutoff
Ecut : energy cutoff
npw : size of planewave basis
Gimax : max index of the reciprocal lattices of sheet i
Gi : full reciprocal lattices of sheet i
G : reciprocal index of two sheets, i.e., G = (G1,G2)
Gmn : sum of reciprocal lattice i.e., G1 + G2
Gmapij : sorting index of (Gi,Gj)
nk : number of k-points
kpts : k-points
model : TBG1D
"""
struct BasisST{T1<:Real,T2<:Real} <: Basis
    Ecut::T1
    npw::Int64
    G1max::Int64
    G2max::Int64
    G1::Vector{T2}
    G2::Vector{T2}
    G::Matrix{Int64}
    Gmn::Vector{T2}
    Gmap12::SparseMatrixCSC{Int64,Int64}
    Gmap21::SparseMatrixCSC{Int64,Int64}
    nk::Int64
    kpts::Vector{Float64}
    model::TBG1D
end

function basisGen(Ecut::T, model::TBG1D,            
                kpts::Vector{Float64}) where {T<:Real}

    latR = model.latR

    G1max = floor(Int, sqrt(2.0 * Ecut) / norm(latR[1]))
    G2max = floor(Int, sqrt(2.0 * Ecut) / norm(latR[2]))
    G1 = collect(-G1max:G1max) .* latR[1]
    G2 = collect(-G2max:G2max) .* latR[2]

    count = 0
    Gmn = Float64[]
    ind1 = Int64[]
    ind2 = Int64[]
    val = Int64[]
    G = Int64[]
    for t1 = -G1max:G1max, t2 = -G2max:G2max
        @views Gt1 = G1[t1+G1max+1]
        @views Gt2 = G2[t2+G2max+1]
        if Gt1^2 + Gt2^2 < 2Ecut
            count += 1
            push!(G,t1,t2)
            push!(Gmn, Gt1 + Gt2)
            push!(ind1, t1 + G1max + 1)
            push!(ind2, t2 + G2max + 1)
            push!(val, count)
        end
    end
    G = Array(reshape(G, 2, Int(length(G) / 2))')
    Gmap12 = sparse(ind1,ind2,val)
    Gmap21 = sparse(ind2,ind1,val)
    println(" Standard cutoff : Ecut =", Ecut, ";  Matrix DOF = ", count)
    nk = length(kpts)

    BasisST(Ecut, count, G1max, G2max, G1, G2, G, Gmn, Gmap12, Gmap21, nk, kpts, model)
end

Basis(Ecut::T, model::TBG1D; kpts = zeros(1)) where {T<:Real} = basisGen(Ecut, model, kpts)

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
struct BasisLW{T1<:Real,T2<:Real} <:Basis
    EcutL::T1
    EcutW::T1
    npw::Int64
    G1max::Int64
    G2max::Int64
    G1::Vector{T2}
    G2::Vector{T2}
    G::Matrix{Int64}
    Gmn::Vector{T2}
    Gmap12::SparseMatrixCSC{Int64,Int64}
    Gmap21::SparseMatrixCSC{Int64,Int64}
    nk::Int64
    kpts::Vector{Float64}
    SdL::Float64
    model::TBG1D
end

function basisGen(EcutL::T, EcutW::T, model::TBG1D, 
                kpts::Vector{Float64}) where {T<:Real}

    latR = model.latR
    dim = size(latR[1],1)
    SdL = (sqrt(pi) * EcutL / 2) ^ dim * gamma(dim/2 + 1)

    G1max = floor(Int, max(EcutL, EcutW) / norm(latR[1]))
    G2max = floor(Int, max(EcutL, EcutW) / norm(latR[2]))
    G1 = collect(-G1max:G1max) .* latR[1]
    G2 = collect(-G2max:G2max) .* latR[2]

    count = 0
    Gmn = Float64[]
    ind1 = Int64[]
    ind2 = Int64[]
    G = Int64[]
    val = Int64[]
    for t1 = -G1max:G1max, t2 = -G2max:G2max
        @views Gt1 = G1[t1+G1max+1]
        @views Gt2 = G2[t2+G2max+1]
        if abs(Gt1 + Gt2) <= EcutW && abs(Gt1 - Gt2) <= EcutL
            count += 1
            push!(G, t1, t2)
            push!(Gmn, Gt1 + Gt2)
            push!(ind1, t1 + G1max + 1)
            push!(ind2, t2 + G2max + 1)
            push!(val, count)
        end
    end
    G = Array(reshape(G, 2, Int(length(G) / 2))')
    Gmap12 = sparse(ind1, ind2, val)
    Gmap21 = sparse(ind2, ind1, val)
    println(" LW cutoff : EcutL = ", EcutL, ", EcutW = ", EcutW, ";  Matrix DOF = ", count)
    nk = length(kpts)
    
    BasisLW(EcutL, EcutW, count, G1max, G2max, G1, G2, G, Gmn, Gmap12, Gmap21, nk, kpts, SdL, model)
end

Basis(EcutL::T, EcutW::T, model::TBG1D; nk=1, kpts=zeros(1)) where {T<:Real} = basisGen(EcutL, EcutW, model, kpts)
