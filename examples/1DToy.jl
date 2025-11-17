using MomentumDOS
using LinearAlgebra, KrylovKit

gauss = [Gaussian(7, 0.05), Gaussian(5, 0.05)]
L = 1
ϵ = 0.01
model = TbgToy(L, ϵ, gauss)

EcL = 100
EcW = 20
basis = Basis(EcL, EcW, model;kpts=[-10.]);
@time H = hamFull(basis);
H1 = Array(H[1])
H2 = Array(H[2])
G0ind = basis.Gmap12[basis.G1max+1, basis.G2max+1]
norm(H1[:,G0ind] - H2[:,G0ind])


HK = ham_Kinetic(basis)
HV = ham_Potential(basis)
@time vmax, umax = eigsolve(HV, 1, :LR)
@time vmin, umin = eigsolve(HV, 1, :SR)
Vop = max(abs(vmin[1]), abs(vmax[1]))

Emin, Umin = eigsolve(H, 1, :SR)
Emax, Umax = eigsolve(H, 1, :LR)

e, u = eigen(Array(H[1]))