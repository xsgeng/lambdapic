#!/bin/env python
import argparse
import math

PI = 3.141592653589793
REF_WAVELENGTH = 0.8e-6   # m, Ti:Sapphire laser wavelength
REF_NC = 1.744e27         # critical density (m^-3) at REF_WAVELENGTH


def skin_depth(ne: float) -> float:
    """Calculate basic skin depth."""
    return REF_WAVELENGTH / 2 / PI * math.sqrt(REF_NC / ne)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate skin depth from electron density."
    )
    parser.add_argument(
        "--ne",
        type=float,
        default=REF_NC,
        help="Electron density (m^-3)"
    )
    parser.add_argument(
        "--a0",
        type=float,
        default=0,
        help="Laser intensity parameter (optional). If provided, applies relativistic correction factor (1+a0^2/2)^(1/4)"
    )

    args = parser.parse_args()

    # Calculate basic skin depth
    sd = skin_depth(args.ne) * (1 + args.a0**2 / 2) ** (1/4)
    print(f"Skin depth: {sd*1e9:.3e} nm")


if __name__ == "__main__":
    main()
