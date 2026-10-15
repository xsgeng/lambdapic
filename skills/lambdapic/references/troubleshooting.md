# Troubleshooting

Known failure modes, ordered by how often they bite. Universal fallback: if nothing below helps, report at https://github.com/xsgeng/lambdapic/issues with the full traceback and reproduction steps.

## Failure Recovery

| Symptom | First fix | If it persists |
|---|---|---|
| `import lambdapic` fails | Activate the venv in the working directory (`source .venv/bin/activate`); if none exists, create one per `installation.md` (`uv venv`) and install | If the install itself fails, check `mpicc` availability; if it still fails, STOP and wait for the user |
| Fields or particles become NaN (diverged run) | Check the CFL condition: `dt < 0.5 * dx / c`; lower `dt` if needed | Check resolution meets the minimum (skin_depth/2 AND wavelength/10) |
| `RuntimeError: Patch data mismatch` | Check `n_guard >= 2`; check for particles escaping the domain | Reproduce single-process to isolate the MPI issue |
| Process killed (OOM) | Reduce `nx`/`ny` or `npatch_x`/`npatch_y` | Chunk the run with `RestartDump` |
| No laser field evolution | Verify the laser callback is registered in `sim.run(callbacks=[...])` and its `position` is inside the domain | Check units — position in meters, time in seconds |
| Custom callback raises | Check `interval` type (int steps / float seconds / callable); check the `sim` attribute it uses exists | Remove that callback and re-run to isolate it |

## Callback Not Firing

- Stage name matches `Simulation.STAGES` exactly
- Interval is appropriate (`interval=1` for every step)
- Callback is passed to `sim.run(callbacks=[...])` — `sim.callbacks` does not exist

## MPI Checklist

- All I/O gated with `if sim.mpi.rank == 0:`
- Cross-rank sums use `sim.mpi.comm.reduce()`
- No two ranks write the same filename

## Field Array Indexing

- Interior region is `:nx_per_patch, :ny_per_patch`; guard cells excluded from physics sums
- Use `get_fields()` for global snapshots (rank 0 only)
