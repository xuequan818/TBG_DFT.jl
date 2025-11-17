# MomentumDOS.jl

Code companion for [![DOI](https://img.shields.io/badge/DOI-10.48550/arXiv.2510.15369-blue)](https://doi.org/10.48550/arXiv.2510.15369) which computes the density of states using the momentum-space method. The semiclassical version can be found [here](https://github.com/epolack/SacerDOS.jl).

## Installation
MomentumDOS.jl is an unregistered package and therefore needs to be downloaded or cloned to the user's local computer first, and then installed by running

```julia
julia> cd("your-local-path/MomentumDOS.jl")
julia> using Pkg
julia> Pkg.activate(".")
julia> Pkg.instantiate()
```

## Usage
Here are some examples showing how to use the packge.

Simulation of a 1D toy model:  
```math
	H_{\epsilon} = -\frac{1}{2}  \frac{d^2}{dx^2} + V(x,(1+\epsilon)x)
```
where the potentials are given by
```math
\begin{align*}
&V(x,(1+\epsilon)x) := \sum_{R\in \mathbb{Z}} \Big(v_1\big(x-R\big)+v_2\big((1+\epsilon)x-R\big)\Big),\\
& v_j(x)=-\frac{A_j}{\sqrt{2\pi\sigma_j^2}} e^{-\frac{|x|^2}{2\sigma_j^2}},\qquad j= 1,2.
\end{align*}
```
```julia
using MomentumDOS
using Plots

# Define the 1D model
gauss = [Gaussian(7, 0.05), Gaussian(5, 0.05)]
L = 1
ϵ = pi/200
model = TbgToy(L, ϵ, gauss);

# Define the basis
EcL = 300
EcW = 20
basis = Basis(EcL, EcW, model);

# Compute the DoS
ER = collect(-8:0.1:34) # Energy range
h = 0.1 
σ = 0.4 # Gaussian parameter for test function
@time dos = compute_dos_shift_kpm(ER, Gauss(σ), basis, h; Ktrunc=20, tol=1e-4);

plot(ER, dos, label="ϵ=$ϵ")
```

## Data
The data supporting all figures and tables in [![DOI](https://img.shields.io/badge/DOI-10.48550/arXiv.2510.15369-blue)](https://doi.org/10.48550/arXiv.2510.15369) are provided in the `paper/data` folder. All figures and tabular data can be reproduced using the [Pluto](https://plutojl.org/en/docs/) notebook `paper/generate_plots.jl`, which can also be run quickly via 

```julia
julia> include("paper/generate_plots.jl")
```

## Citation
```bibtex
@misc{cances_numerical_2025,
	title = {Numerical computation of the density of states of aperiodic multiscale {Schrödinger} operators},
	url = {http://arxiv.org/abs/2510.15369},
	doi = {10.48550/arXiv.2510.15369},
	publisher = {arXiv},
	author = {Cancès, Eric and Massatt, Daniel and Meng, Long and Polack, Étienne and Quan, Xue},
	year = {2025},
	note = {arXiv:2510.15369},
}
```
