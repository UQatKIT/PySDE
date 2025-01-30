![Docs](https://img.shields.io/github/actions/workflow/status/UQatKIT/PySDE/docs.yaml?label=Docs)
![Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FUQatKIT%2FPySDE%2Fmain%2Fpyproject.toml)
![License](https://img.shields.io/github/license/UQatKIT/PySDE) 
![Beartype](https://github.com/beartype/beartype-assets/raw/main/badge/bear-ified.svg)
![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)

# PySDE: A Light-Weight Numerical Integrator for Stochastic Differential Equations

PySDE is a light-weight numerical integrator for stochastic differential equations (SDE). More 
specifically, we consider vector-valued diffusion processes $\mathbf{X}_t$ on $\Omega\in\mathbb{R}^{d_X}$, continuously indexed over time $t\in\mathbb{R}$. PySDE solves corresponding SDEs of the form

$$
    d\mathbf{X}_t = \mathbf{b}(\mathbf{X}_t,t)dt + \mathbf{\Sigma}(\mathbf{X}_t,t) d\mathbf{W}_t,\quad\mathbf{X}(t=0)=\mathbf{X}_0\ \ a.s,
$$

 with vector-valued *drift* $\mathbf{b}: \Omega\to\mathbb{R}^{d_X}$, matrix-valued *diffusion* $\mathbf{\Sigma}: \Omega\to\mathbb{R}^{d_x\times d_W}$, and vector valued *Wiener process* $\mathbf{W}_t \in\mathbb{R}^{d_W}$.

 PySDE has a modular core, making it easy to combine different components of the integrator or extend them. At the same time, PySDE is just simple: Simple to use, simple to understand. The user can provide drift and diffusion in simple numpy arrays. At the same time, PySDE is tailored towards integration of large ensembles of trajectories. Loops over trajectories are jit-compiled and accelerated with [Numba](https://numba.pydata.org/). We further implement custom data structures that import out-of-memory storage of the results. No need to evaluate statistics online or compromise on data resolution.


 > The excellent [Diffrax](https://github.com/patrick-kidger/diffrax) library is a more powerful general-purpose library for solving SDEs. PySDE has the goal to be complementary to Diffrax, serving a niche: It is very simple to use, understand, and extend (without knowledge of JAX). Moreover, it is specifically tailored towards large trajectory ensembles on HPC architectures. At some point, PySDE might be come a wrapper to Diffrax, but even that wrapper code would probably be more complex than the current internals of PySDE.

 ## Installation

 PySDE can be installed via pip,
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

Check out the [documentation](https://uqatkit.github.io/PySDE/) for further information regarding usage and API. We also provide runnable [`examples`](https://github.com/UQatKIT/PySDE/tree/main/examples).


## Acknowledgement and License

PySDE is being developed in the research group [Uncertainty Quantification](https://www.scc.kit.edu/forschung/uq.php) at KIT.
It is distributed as free software under the [MIT License](https://choosealicense.com/licenses/mit/).