# Overview
This repository contains Python code for running the Nested Sampling algorithm for estimation of the evidence integral (Partition Function) of systems for which analytical determination is intractable. The algorithm works
by partitioning the sample space (phase space) of a system of interest into regions of similar likelihood, weighting them by an estimate for how much of the sample space this region occupies, and summing over many likelihood regions to compute
the evidence. 
# File Structure
- nested_sampling: contains the code for the Nested Sampling algorithm and for implementing Monte Carlo (MC) walkers for the algorithm. Pure Python MC is available as well as C++ implementation (see [mcpele](https://github.com/martiniani-lab/mcpele/tree/master))
- examples: example scripts and use cases for Nested Sampling ranging from a harmonic potential to a Lennard Jones system.
- tests: various tests for classes and functions required for running Nested Sampling.
