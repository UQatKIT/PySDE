# PySDE [<img src="images/uq_logo.png" width="200" height="100" alt="UQ at KIT" align="right">](https://www.scc.kit.edu/forschung/uq.php)

PySDE is a light-weight numerical integrator for stochastic differential equations (SDEs). More 
specifically, we consider vector-valued diffusion processes $\mathbf{X}_t$ on $\Omega\in\mathbb{R}^{d_X}$, continuously indexed over time $t\in\mathbb{R}$. PySDE solves corresponding SDEs of the form

$$
    d\mathbf{X}_t = \mathbf{b}(\mathbf{X}_t,t)dt + \mathbf{\Sigma}(\mathbf{X}_t,t) d\mathbf{W}_t,\quad\mathbf{X}(t=0)=\mathbf{X}_0\ \ a.s,
$$

with vector-valued *drift* $\mathbf{b}: \Omega\to\mathbb{R}^{d_X}$, matrix-valued *diffusion* $\mathbf{\Sigma}: \Omega\to\mathbb{R}^{d_x\times d_W}$, and vector valued *Wiener process* $\mathbf{W}_t \in\mathbb{R}^{d_W}$.

PySDE has a modular core, making it easy to combine different components of the integrator or extend them. At the same time, PySDE is just simple: Simple to use, simple to understand. The user can provide drift and diffusion as callables in Python or numpy syntax. At the same time, PySDE is tailored towards integration of large ensembles of trajectories. Loops over trajectories are jit-compiled and accelerated with [Numba](https://numba.pydata.org/). We further implement custom data structures that support out-of-memory storage of the results. No need to evaluate statistics online or compromise on data resolution.

!!! note "PySDE and Diffrax"

    The excellent [Diffrax](https://github.com/patrick-kidger/diffrax) library is a more powerful general-purpose library for solving SDEs. PySDE has the goal to be complementary to Diffrax, serving a niche: It is very simple to use, understand, and extend (without knowledge of JAX). Moreover, it is specifically tailored towards large trajectory ensembles on HPC architectures. At some point, PySDE might be come a wrapper to Diffrax, but even that wrapper code would probably be more complex than the current internals of PySDE.

!!! warning
    PySDE is a library developed in the course of a research project, not as a dedicated tool. As
    such, it has been tested for a number of example use cases, but not with an exhaustive test suite. Therefore, we currently do not intend to upload this library to a public index.


## Installation

PySDE can be installed via pip in the project root directory,
```bash
pip install pysde
```

To install with MPI support, install with the corresponding extra (MPI needs to be available in the system path),
```bash
pip install pysde[mpi]
```

For development, we recommend using the great [uv](https://docs.astral.sh/uv/) project management tool, for which we provide a universal lock file. To set up a reproducible environment, run
```bash
uv sync --all-groups
```

## Documentation

Under **Usage**, we provide examples of the different ways in which PySDE can be used to integrate SDEs. The **API reference** gives a more detailed overview of the different software components and their internals.


## Acknowledgement and License

PySDE is being developed in the research group [Uncertainty Quantification](https://www.scc.kit.edu/forschung/uq.php) at KIT.
It is distributed as free software under the [MIT License](https://choosealicense.com/licenses/mit/).