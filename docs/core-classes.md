# Core Classes

The following diagram shows the main classes in λPIC and their relationships.

```mermaid
classDiagram
    Simulation <|-- Simulation3D~Simulation~
    Simulation --> Patches : contains
    Simulation --> MPIManager : has
    Simulation --> MaxwellSolver : has
    Simulation --> FieldInterpolation : has
    Simulation --> CurrentDeposition : has
    Simulation --> PusherBase : has
    Simulation --> RadiationBase : has
    Simulation --> PairProductionBase : has
    Simulation --> Species : has
    Simulation --> LoadBalancer : has

    Patches --> Patch : contains
    Patches --> Species : has
    Patch --> ParticlesBase : contains
    Patch --> Fields : contains
    Patch --> PML : contains
    Patch <|-- Patch2D~Patch~
    Patch <|-- Patch3D~Patch~

    Fields <|-- Fields2D~Fields~
    Fields <|-- Fields3D~Fields~

    ParticlesBase <|-- QEDParticles
    ParticlesBase <|-- SpinParticles
    ParticlesBase <|-- SpinQEDParticles

    Species --> ParticlesBase : creates
    Species <|-- Electron~Species~
    Species <|-- Proton~Species~
    Species <|-- Photon~Species~
    Electron <|-- Positron~Electron~

    RadiationBase --> Patches : contains
    PairProductionBase --> Patches : contains
    CurrentDeposition --> Patches : contains
    PusherBase --> Patches : contains
    FieldInterpolation --> Patches : contains
    MaxwellSolver --> Patches : contains
    MPIManager --> Patches : contains

    MaxwellSolver <|-- MaxwellSolver2D
    MaxwellSolver <|-- MaxwellSolver3D
    CurrentDeposition <|-- CurrentDeposition2D
    CurrentDeposition <|-- CurrentDeposition3D
    FieldInterpolation <|-- FieldInterpolation2D
    FieldInterpolation2D <|-- FieldInterpolation3D
    PusherBase <|-- BorisPusher
    PusherBase <|-- PhotonPusher
    PusherBase <|-- BorisTBMTPusher
    MPIManager <|-- MPIManager2D
    MPIManager <|-- MPIManager3D
    RadiationBase <|-- NonlinearComptonLCFA
    RadiationBase <|-- ContinuousRadiation
    PairProductionBase <|-- NonlinearPairProductionLCFA

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

## Overview

- **Simulation** orchestrates the entire simulation loop and owns all major components.
- **Patches** is a collection of **Patch** instances, each holding local **Fields** and **ParticlesBase**.
- **Species** defines particle properties and creates **ParticlesBase** instances.
- Solver classes (e.g. **MaxwellSolver**, **CurrentDeposition**, **PusherBase**) operate on **Patches**.
