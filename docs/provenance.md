# Source and mathematical provenance

The project follows `README_DK_Neural_Coarse_Dynamics_and_Statistics_v0_1.md`, supplied on 23 October 2026, with Project A central and B1 as its statistical comparator.

The legacy reference is the user's local `doctoral-thesis/scripts/Dean-Kawasaki/deankawasaki_animation` implementation, inspected read-only. It supplies useful grid, particle, conservative flux, and diagnostic ideas. NN-DK contains newly implemented and tested one-dimensional backends with no runtime dependency on the thesis checkout.

At inspection, that directory was untracked. The public thesis repository [xrough/spde-mf](https://github.com/xrough/spde-mf/tree/9afa694a76ba54a3c1ab9422e5bbb0f839e7526f) at commit `9afa694a76ba54a3c1ab9422e5bbb0f839e7526f` did not include it. The old `jixh-KPZ-1020/spde-mf` URL redirects to the same repository. The public commit should not be cited as containing the local DK solver.

Key corrections relative to the animation workflow are explicit cell/spline conventions, exact finite-mode particle forces, matched IID initialization diagnostics, no state clipping or renormalization, uncertainty-aware statistics, and training/evaluation separation.

Mathematical context:

- [Djurdjevac, Ji, Perkowski: weak error for regularized interacting DK](https://arxiv.org/abs/2502.20929). Its hypotheses do not automatically validate this discrete comparator.
- [Lehmann, Konarovskyi, von Renesse: DK ill-posedness versus triviality](https://arxiv.org/abs/1806.05018). The initial target here is the projected finite particle law.
- [Siddani et al.: history-dependent flux learning](https://arxiv.org/abs/2606.06658). Relevant prior work for the deferred memory/non-Gaussian branch.

No continuum convergence, universal transfer, or neural superiority is claimed by the software.
