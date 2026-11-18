from scipy.constants import c
from math import sqrt
from numba import njit


def tbmt_spin_inline(sx, sy, sz, ux, uy, uz, Ex, Ey, Ez, Bx, By, Bz, q, m, dt, ae):
    """Standalone TBMT (Thomas-Bargmann-Michel-Telegdi) spin rotation.

    To be applied AFTER the full Boris push: the momentum is not modified
    (no Stern-Gerlach feedback), only the spin is rotated.

    The effective field must be evaluated at the mid-step momentum
    :math:`\\mathbf{u}^-` (after the first E half kick), like in the fused
    Boris+TBMT scheme. It is reconstructed exactly from the final momentum
    :math:`\\mathbf{u}^+` at no extra storage cost: the Boris rotation
    preserves the norm, so

    .. math::
        \\mathbf{u}_\\mathrm{rot} = \\mathbf{u}^+ - \\frac{q\\Delta t}{2mc}\\mathbf{E},
        \\qquad \\gamma^- = \\sqrt{1 + |\\mathbf{u}_\\mathrm{rot}|^2}

    and :math:`\\mathbf{u}^- = M(-T)\\,\\mathbf{u}_\\mathrm{rot}` with
    :math:`T = q\\Delta t\\,\\mathbf{B}/(2m\\gamma^-)`, where :math:`M(-T)`
    is the inverse Boris rotation.

    The spin is then Boris-rotated around the TBMT effective field

    .. math::
        \\mathbf{B}_\\mathrm{eff} = \\left(a + \\frac{1}{\\gamma}\\right)\\mathbf{B}
        - \\frac{a\\gamma}{\\gamma+1}(\\boldsymbol{\\beta}\\cdot\\mathbf{B})\\boldsymbol{\\beta}
        - \\left(a + \\frac{1}{\\gamma+1}\\right)\\frac{\\boldsymbol{\\beta}\\times\\mathbf{E}}{c}

    so that :math:`d\\mathbf{s}/dt = (q/m)\\,\\mathbf{s}\\times\\mathbf{B}_\\mathrm{eff}`.

    Args:
        sx, sy, sz (float): spin vector components
        ux, uy, uz (float): normalized momentum :math:`u = \\gamma\\beta`
        Ex, Ey, Ez, Bx, By, Bz (float): fields at the particle position (SI)
        q (float): signed charge
        m (float): mass
        dt (float): timestep
        ae (float): anomalous magnetic moment a = (g-2)/2

    Returns:
        tuple: (sx_new, sy_new, sz_new)
    """
    efactor = q*dt/(2*m*c)
    bfactor = q*dt/(2*m)

    # --- reconstruct the mid-step (half E kick) momentum u^- ---
    urx = ux - efactor * Ex
    ury = uy - efactor * Ey
    urz = uz - efactor * Ez

    inv_gamma = 1 / sqrt(1 + urx*urx + ury*ury + urz*urz)
    gamma = 1 / inv_gamma

    # inverse Boris rotation: u^- = M(-T) u_rot with T = bfactor*B/gamma^-
    Tx = -bfactor * Bx * inv_gamma
    Ty = -bfactor * By * inv_gamma
    Tz = -bfactor * Bz * inv_gamma

    upx = urx + ury*Tz - urz*Ty
    upy = ury + urz*Tx - urx*Tz
    upz = urz + urx*Ty - ury*Tx

    Tfactor = 2 / (1 + Tx*Tx + Ty*Ty + Tz*Tz)
    Sx = Tfactor * Tx
    Sy = Tfactor * Ty
    Sz = Tfactor * Tz

    umx = urx + upy*Sz - upz*Sy
    umy = ury + upz*Sx - upx*Sz
    umz = urz + upx*Sy - upy*Sx

    # beta = u / gamma
    betax = umx * inv_gamma
    betay = umy * inv_gamma
    betaz = umz * inv_gamma

    # --- TBMT effective field (SI units) ---
    beta_dot_B = betax*Bx + betay*By + betaz*Bz
    pref1 = ae * gamma / (gamma + 1) * beta_dot_B
    pref2 = ae + 1 / (gamma + 1)

    Bx_eff = (ae + inv_gamma) * Bx - pref1 * betax - pref2 * (betay*Ez - betaz*Ey)/c
    By_eff = (ae + inv_gamma) * By - pref1 * betay - pref2 * (betaz*Ex - betax*Ez)/c
    Bz_eff = (ae + inv_gamma) * Bz - pref1 * betaz - pref2 * (betax*Ey - betay*Ex)/c

    # --- spin rotation (Boris scheme) ---
    Tsx = bfactor * Bx_eff
    Tsy = bfactor * By_eff
    Tsz = bfactor * Bz_eff

    T2 = Tsx*Tsx + Tsy*Tsy + Tsz*Tsz
    Ssx = 2 * Tsx / (1 + T2)
    Ssy = 2 * Tsy / (1 + T2)
    Ssz = 2 * Tsz / (1 + T2)

    spx = sx + sy*Tsz - sz*Tsy
    spy = sy + sz*Tsx - sx*Tsz
    spz = sz + sx*Tsy - sy*Tsx

    sx_new = sx + spy*Ssz - spz*Ssy
    sy_new = sy + spz*Ssx - spx*Ssz
    sz_new = sz + spx*Ssy - spy*Ssx

    return sx_new, sy_new, sz_new


tbmt_spin_cpu = njit(tbmt_spin_inline, inline="always")


@njit(cache=True)
def tbmt_spin(sx, sy, sz, ux, uy, uz,
              ex_part, ey_part, ez_part, bx_part, by_part, bz_part,
              q, m, ae, npart, is_dead, dt):
    for ip in range(npart):
        if is_dead[ip]:
            continue

        sx[ip], sy[ip], sz[ip] = tbmt_spin_cpu(
            sx[ip], sy[ip], sz[ip], ux[ip], uy[ip], uz[ip],
            ex_part[ip], ey_part[ip], ez_part[ip], bx_part[ip], by_part[ip], bz_part[ip],
            q, m, dt, ae)
