using MomentumDOS
using LinearAlgebra
using Test

function ham_potential_test(basis::Basis)
    nk = basis.nk
    v1 = basis.model.vft[1]
    v2 = basis.model.vft[2]
    latR = basis.model.latR

    H = Vector{Matrix{Float64}}(undef, nk)
    for k = 1:nk
        kpt = basis.kpoints[k]
        npw = kpt.npw
        G = kpt.G_vec
        Hk = zeros(npw, npw)
        for n1 = 1:npw, n2 = 1:npw
            @views g11 = G[n1][1]
            @views g12 = G[n1][2]
            @views g21 = G[n2][1]
            @views g22 = G[n2][2]
            v1m = g12 == g22 ? v1((g11 - g21) * latR[1]) : 0.0
            v2n = g11 == g21 ? v2((g12 - g22) * latR[2]) : 0.0
            Hk[n1, n2] = v1m + v2n
        end
        H[k] = Hk
    end

    return H
end

gauss = [Gaussian(7, 0.05), Gaussian(5, 0.05)]
L = 5
ϵ = 0.1
model = TbgToy(L, ϵ, gauss)

EcutL = 50
EcutW = 20
basis = Basis(EcutL, EcutW, model;kpts=collect(-1:0.5:1));
@time H = map(k -> ham_Potential(basis, k), 1:basis.nk);
@time Htest = ham_potential_test(basis);

for k = 1:basis.nk
    e = norm(Array(H[k]) - Htest[k])
    @test e < 1e-8
end
