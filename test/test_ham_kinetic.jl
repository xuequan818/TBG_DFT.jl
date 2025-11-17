using MomentumDOS
using LinearAlgebra
using Test

function ham_kinetic_test(basis::Basis)
	nk = basis.nk

    H = Vector{Matrix{Float64}}(undef, nk)
    for k = 1:nk
		kpt = basis.kpoints[k]
		npw = kpt.npw
        Hk = zeros(npw, npw)
		Gmn = kpt.G_cart_sum
        for (n1, gmn) in enumerate(Gmn)
            Hk[n1, n1] = 0.5 * (gmn + kpt.coordinate)^2 
        end
		H[k] = Hk
    end

	return H
end

gauss = [Gaussian(1.0, 0.5), Gaussian(1.1, 0.5)]
L = 5
ϵ = 0.1
model = TbgToy(L, ϵ, gauss)

EcutL = 100
EcutW = 20
basis = Basis(EcutL,EcutW, model);
@time H = map(k->ham_Kinetic(basis,k),1:basis.nk);
@time Htest = ham_kinetic_test(basis);

for k = 1:basis.nk
	e = norm(Array(H[k])-Htest[k])
	@test e < 1e-8
end
