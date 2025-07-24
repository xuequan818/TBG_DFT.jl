export  ham_Kinetic, ham_Potential, ham

# HK = 0.5|G1+G2+ξ|^2δ_{G,G'}
function ham_Kinetic(basis::Basis, ik::Int)
    kpt = basis.kpoints[ik]
    vals = @. 0.5 * (kpt.coordinate + kpt.G_cart_sum)^2
    sparse(1:kpt.npw, 1:kpt.npw, vals)
end

function genV(Gmap::SparseMatrixCSC{Int64,Int64}, G, v::Function, indi, indj, vals)
    
    rowval = Gmap.rowval
    jptr = Gmap.colptr[2:end] - Gmap.colptr[1:end-1]
    count = 0
    for j = 1:length(jptr) #loop for the nonzero elements of column
        jtr = jptr[j]
        for l1 = 1:jtr, l2 = 1:jtr
            i1 = rowval[l1 + count]
            i2 = rowval[l2 + count]
            vdG = v(G[i1] - G[i2])
            push!(vals, vdG)
            push!(indi, Gmap[i1, j])
            push!(indj, Gmap[i2, j])
        end
        count += jtr
    end

    return indi, indj, vals
end

# HV = V1_{G1-G1'}δ_{G2,G2'}+V2_{G2-G2'}δ_{G1,G1'}
function ham_Potential(basis::Basis, ik::Int)
    kpt = basis.kpoints[ik]
    model = basis.model
    v1 = model.vft[1]
    v2 = model.vft[2]
    G1 = basis.G1
    G2 = basis.G2
    Gmap12 = kpt.Gmap12
    Gmap21 = kpt.Gmap21
    npw = kpt.npw

    indi = Int[]
    indj = Int[]
    vals = typeof(v1(G1[1]))[]

    genV(Gmap12, G1, v1, indi, indj, vals) # generate V1_{G1-G1'}δ(G2,G2')
    genV(Gmap21, G2, v2, indi, indj, vals) # generate V2_{G2-G2'}δ(G1,G1')

    sparse(indi, indj, vals, npw, npw)
end

ham(basis::Basis, ik::Int64) = ham_Kinetic(basis, ik) + ham_Potential(basis, ik)

ham(basis::Basis) = map(ik->ham(basis,ik), 1:basis.nk)