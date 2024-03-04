using TBG_DFT
using LinearAlgebra
using Test

function ham_potential_test(basis::Basis)
    nk = basis.nk
    npw = basis.npw
    G = basis.G
    kpts = basis.kpts
    v1 = basis.model.vft[1]
    v2 = basis.model.vft[2]
    latR = basis.model.latR

    H = zeros(Float64, nk, npw, npw)
    for k = 1:nk
        for n1 = 1:npw, n2 = 1:npw
            @views g11 = G[n1, 1]
            @views g12 = G[n1, 2]
            @views g21 = G[n2, 1]
            @views g22 = G[n2, 2]
            v1m = g12 == g22 ? v1((g11 - g21) * latR[1]) : 0.0
            v2n = g11 == g21 ? v2((g12 - g22) * latR[2]) : 0.0
            H[k, n1, n2] = v1m + v2n
        end
    end

    return H
end

gauss = [Gaussian(1.0, 0.5), Gaussian(1.1, 0.5)]
L = 5
ϵ = 0.1
model = TbgToy(L, ϵ, gauss)

Ecut = 500
basis = Basis(Ecut, model);
@time H = ham_Potential(basis);
@time Htest = ham_potential_test(basis);

nk = basis.nk
npw = basis.npw
for k = 1:nk
    e = norm(Array(H[npw*(k-1)+1:npw*k, npw*(k-1)+1:npw*k]) - Htest[k, :, :])
    @test e < 1e-8
end
