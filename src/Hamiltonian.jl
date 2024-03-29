export  ham_Kinetic, ham_Potential, hamK, hamFull

function ham_Kinetic(basis::Basis, kgrid)

    Gmn = basis.Gmn
    npw = basis.npw
    vals = 0.5 .* norm.(kgrid.+Gmn).^2

    sparse(1:npw, 1:npw, vals)
end

ham_Kinetic(basis::Basis) = ham_Kinetic(basis, zero(basis.kpts[1]))

function genV(Gmap::SparseMatrixCSC{Int64,Int64}, G, v::Function, indi, indj, vals)
    
    dG = zeros(size(G, 2))
    rowval = Gmap.rowval
    jptr = Gmap.colptr[2:end] - Gmap.colptr[1:end-1]
    count = 0
    for j = 1:length(jptr) #loop for the nonzero elements of column
        jtr = jptr[j]
        for l1 = 1:jtr, l2 = 1:jtr
            i1 = rowval[l1 + count]
            i2 = rowval[l2 + count]
            @views Gi1 = G[i1, :]
            @views Gi2 = G[i2, :]
            @. dG = Gi1 - Gi2
            vdG = v(dG)
            if norm(vdG,Inf) > 1e-10
                push!(vals, vdG)
                push!(indi, Gmap[i1, j])
                push!(indj, Gmap[i2, j])
            end
        end
        count += jtr
    end

    return indi, indj, vals
end

function ham_Potential(basis::Basis)
    model = basis.model
    v1 = model.vft[1]
    v2 = model.vft[2]
    G1 = basis.G1
    G2 = basis.G2
    Gmap12 = basis.Gmap12
    Gmap21 = basis.Gmap21
    npw = basis.npw

    indi = Int[]
    indj = Int[]
    vals = typeof(v1(G1[1]))[]

    genV(Gmap12, G1, v1, indi, indj, vals) # generate V1_{G1-G1'}δ(G2,G2')
    genV(Gmap21, G2, v2, indi, indj, vals) # generate V2_{G2-G2'}δ(G1,G1')

    sparse(indi, indj, vals, npw, npw)
end

hamK(basis::Basis, k::Int64) = ham_Kinetic(basis, basis.kpts[k]) + ham_Potential(basis)

hamK(basis::Basis, k::Int64, HV) = ham_Kinetic(basis, basis.kpts[k]) + HV

hamFull(basis::Basis) = map(k->hamK(basis,k), 1:basis.nk)