export  ham_Kinetic, ham_Potential, hamiltonian

function ham_Kinetic(basis::Basis)
    nk = basis.nk
    kpts = basis.kpts
    Gmn = basis.Gmn
    npw = basis.npw

    vals = Float64[]
    Gmnk = similar(Gmn)
    valk = zeros(npw)
    for k = 1:nk #loop for the k points
        @. Gmnk = kpts[k] + Gmn
        @. valk = 0.5 * norm(Gmnk)^2

        append!(vals, valk)
    end

    sparse(1:nk*npw, 1:nk*npw, vals)
end

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
    nk = basis.nk

    indi = Int[]
    indj = Int[]
    vals = typeof(v1(G1[1]))[]

    genV(Gmap12, G1, v1, indi, indj, vals) # generate V1_{G1-G1'}δ(G2,G2')
    genV(Gmap21, G2, v2, indi, indj, vals) # generate V2_{G2-G2'}δ(G1,G1')

    indifull = []
    indjfull = []
    for k = 1:nk
        append!(indifull, indi .+ (k - 1) * npw)
        append!(indjfull, indj .+ (k - 1) * npw)
    end 

    sparse(indifull, indjfull, repeat(vals, nk), nk * npw, nk * npw)
end

hamiltonian(basis::Basis) = ham_Kinetic(basis) + ham_Potential(basis)