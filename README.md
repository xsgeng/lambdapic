# Introduction
<div style="text-align:center;">

![λPIC](lambdaPIC.svg)
</div>
λPIC is a callback-centric Particle-In-Cell framework.
It enables the customization of simulation behavior through callbacks at various stages, even when the modifications are unphysical.
The flexibility of λPIC makes it easy to implement plugins, allowing developers to extend functionality seamlessly without modifying the core simulation logic.

Visit the [documentation](https://lambdapic.readthedocs.io/) for installation and usage instructions.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

A detailed class diagram is available in the [documentation](docs/core-classes.md).

## Development Status

λPIC is currently in active development. The API may change without notice.

## License

This project is licensed under the GPL-3.0 License.

## Citation

If you use λPIC in your research, please cite:

- *λPIC: A callback-centric particle-in-cell framework*, arXiv:2607.13507 [physics.comp-ph] (2026). https://arxiv.org/abs/2607.13507

BibTeX:

```bibtex
@article{geng2026lambdapic,
  title={$\lambda$PIC: A callback-centric particle-in-cell framework},
  author={Geng, Xuesong and Cui, Yunwei and Zhang, Lingang and Ji, Liangliang},
  journal={arXiv preprint arXiv:2607.13507},
  year={2026},
  eprint={2607.13507},
  archivePrefix={arXiv},
  primaryClass={physics.comp-ph}
}
```

## Acknowledgments

This work was supported by the National Natural Science Foundation of China (NSFC) under Grant No. 12304384.

This project was inspired by and adapted elements from the [EPOCH](https://github.com/Warwick-Plasma/epoch) and the [Smilei](https://github.com/SmileiPIC/Smilei) projects.