# Third-party notices

This repository contains code written for the project itself, plus small pieces derived from,
or built against, the projects below. No third-party library is vendored into this repository;
they are dependencies installed through `requirements*.txt`.

## aligator — BSD 2-Clause License

`utils.py` (the `create_cartpole` helper) is derived from the cart-pole example utilities of the
[aligator](https://github.com/Simple-Robotics/aligator) project:

```
BSD 2-Clause License

Copyright (c) 2022-2025, Inria
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

The expert trajectories in `cartpole_data/` were generated with aligator (ProxDDP). If you use
them or the generation script in academic work, please cite:

```
@article{jallet2025proxddp,
  title   = {{PROXDDP}: Proximal constrained trajectory optimization},
  author  = {Jallet, Wilson and Bambade, Antoine and Arlaud, Etienne and El-Kazdadi, Sarah and
             Mansard, Nicolas and Carpentier, Justin},
  journal = {IEEE Transactions on Robotics},
  volume  = {41},
  pages   = {2605--2624},
  year    = {2025},
  doi     = {10.1109/TRO.2025.3554437},
}
```

## qpth — Apache License 2.0

The differentiable QP layer in `models.py` is built on top of
[qpth](https://github.com/locuslab/qpth) (`QPFunction` / `SpQPFunction`), which is distributed
under the Apache License 2.0 and was released together with the OptNet paper.

## OptNet — Apache License 2.0

The model design follows [OptNet: Differentiable Optimization as a Layer in Neural
Networks](https://github.com/locuslab/optnet) (Amos & Kolter, ICML 2017). Please cite the paper
if you use this code — the BibTeX entry is in the README.
