# Introduction
<div style="text-align:center;">

![λPIC](lambdaPIC.svg)
</div>
λPIC is a callback-centric Particle-In-Cell framework.
It enables the customization of simulation behavior through callbacks at various stages, even when the modifications are unphysical.
The flexibility of λPIC makes it easy to implement plugins, allowing developers to extend functionality seamlessly without modifying the core simulation logic.

Visit the [documentation](https://lambdapic.readthedocs.io/) for installation and usage instructions.

## Core Classes
```mermaid
classDiagram
    Patches --> Patch : contains

    Patch --> ParticlesBase : contains
    Patch --> Fields : contains
    Patch --> PML : contains

    Patch <|-- Patch2D~Patch~
    Patch <|-- Patch3D~Patch~

    RadiationBase --> Patches : contains
    PairProductionBase --> Patches : contains
    CurrentDeposition --> Patches : contains
    PusherBase --> Patches : contains
    FieldInterpolation --> Patches : contains
    MaxwellSolver --> Patches : contains
    MPIManager --> Patches : contains

    Pydantic.BaseModel <|-- Species
    Species <|-- XXX~Species~
    Species <|-- Electron~Species~
    Species --> ParticlesBase : creates

    class Patch {
        index: int
        *_neighbor_index: int
    }

    class Patches {
        sync_particles()
        sync_guard_fields()
        sync_currents()
    }

    class ParticlesBase {
        x,y,z ...: NDArray[float]
        is_dead: NDArray[bool]
    }

    class Fields {
        ex, ey, ...: NDArray[float]
    }

    class Species {
        name, q, m, ...
        density: Callable
    }
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Development Status

λPIC is currently in active development. The API may change without notice.

## License

This project is licensed under the GPL-3.0 License.

## Citation

If you use λPIC in your research, please cite:

- Xuesong Geng, Yunwei Cui, Lingang Zhang, and Liangliang Ji, *λPIC: A callback-centric particle-in-cell framework*, arXiv:2607.13507 [physics.comp-ph] (2026). https://arxiv.org/abs/2607.13507

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