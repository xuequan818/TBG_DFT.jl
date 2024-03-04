using TBG_DFT
using LinearAlgebra
using Test

function ham_kinetic_test(basis::Basis)
	nk = basis.nk
	npw = basis.npw
	Gmn = basis.Gmn
	kpts = basis.kpts

    H = zeros(Float64, nk, npw, npw)
    for k = 1:nk
        for n1 = 1:npw
            @views gmn = Gmn[n1]
            H[k, n1, n1] = 0.5 * norm(gmn + kpts[k])^2 
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
@time H = ham_Kinetic(basis);
@time Htest = ham_kinetic_test(basis);

nk = basis.nk
npw = basis.npw
for k = 1:nk
	e = norm(Array(H[npw*(k-1)+1:npw*k,npw*(k-1)+1:npw*k])-Htest[k,:,:])
	@test e < 1e-8
end
