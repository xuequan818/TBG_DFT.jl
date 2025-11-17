module MomentumDOS

using Arpack, LinearAlgebra, KrylovKit
using Printf, Plots, Plots.PlotMeasures, LaTeXStrings
using StaticArrays, SparseArrays
using SpecialFunctions, FFTW
using FoldsThreads, Folds, FLoops
using ProgressMeter

prog(n) = Progress(n; dt=0.5, desc="Computing k-points:")

include("model.jl")
include("basis.jl")
include("Hamiltonian.jl")
include("dos.jl")
include("density.jl")

end # module MomentumDOS
