# YaoExtensions

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://QuantumBFS.github.io/YaoExtensions.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://QuantumBFS.github.io/YaoExtensions.jl/dev)
[![Build Status](https://travis-ci.com/QuantumBFS/YaoExtensions.jl.svg?branch=master)](https://travis-ci.com/QuantumBFS/YaoExtensions.jl)
[![Codecov](https://codecov.io/gh/QuantumBFS/YaoExtensions.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/QuantumBFS/YaoExtensions.jl)

Extensions for [Yao](https://github.com/QuantumBFS/Yao.jl). （Abandoned, circuits and hamiltonians are moved to Yao.EasyBuild)

## List of features
#### Easy constructions
* circuits
  * variational_circuit(n): construct a random parametrized circuit.
  * rand_supremacy2d(nx, ny, depth): construct a quantum supremacy circuit.
  * qft_circuit(n): construct a quantum fourier transformation circuit.
  * rand_google53(): construct a Google's 53 qubit supremacy circuit.

* circuit building blocks
  * general_U4(): an optimal decomposition of a U(4) gate.
  * cphase(nbit, i, j), cz(nbit, i, j)

* hamiltonians
  * heisenberg(n): construct a heisenberg hamiltonian.
  * transverse_ising(n): construct a transverse field Ising hamiltonian.

#### Block extensions
* primitive blocks
  * SqrtX, SqrtY, FSim, ISWAP, SqrtW
  * Mod: modulo operation block.
  * QFT: faster implementation of QFT subroutine, instead of running QFT circuit faithfully, simulate it with classical `fft` (thus much faster).
  * RotBasis: basis rotor, make measurements on different basis easier.
  * MathGate: classical mathematic functions.
  * ReflectGate: used in grover search,
  
* composite blocks
  * Bag: a trivial container block that gives the flexibility to change the sub-block, as well as masking. Mainly used for structure learning.
  * ConditionBlock: conditional control the excusion of two block.
  * Sequence: similar to chain block, but more general, one can put anything inside.
  * PauliString: a paulistring.


#### Utlities
* gatecount, count the number of gates,
* faithful simulation of gradients, including observable loss and MMD loss
