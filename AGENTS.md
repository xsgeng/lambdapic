## Testing Guidelines
- Framework: pytest
- Test structure: Mirror source structure in `tests/` directory
- Avoid mock: Construct real Simulation instance instead of mocking.
- Physical parameters: Use realistic parameters like density~1e27 m^-3, cell size~0.01um, timestep~1e-17 s, ppc~10, etc. To make some results more distinctive, modify the suggested values.

## Coding Style & Conventions
- Python 3.10+: Type hints and modern Python features
- Naming: Descriptive names, camelCase for classes, snake_case for functions
- Type hints: Use `|` instead of `Union`, `| None` instead of `Optional`
- Docstrings: Prefer google-style docstrings with a one-line summary, optional context, and explicit sections such as `Parameters: arg (type): description`, `Attributes`, `Returns`, `Notes`, and `Examples`.