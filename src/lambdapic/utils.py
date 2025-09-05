import re
from .core.species import Species
from packaging.version import Version, InvalidVersion
import numpy as np

def uniquify_species_names(existing_species: list[Species], new_species: list[Species]):
    """Directly modify the name attribute of species in the input list"""
    existing_names = [s.name for s in existing_species]
    
    for s in new_species:
        base_name = s.name
        pattern = re.compile(rf'^{re.escape(base_name)}(\.\d+)?$')
        max_suffix = -1
        
        # Check existing names and already processed names in current batch
        for name in existing_names + [s.name for s in new_species[:new_species.index(s)]]:
            if match := pattern.match(name):
                if suffix_part := match.group(1):
                    current_suffix = int(suffix_part[1:])
                    max_suffix = max(max_suffix, current_suffix)
                else:
                    max_suffix = max(max_suffix, 0)
        
        # Directly modify the name attribute of the original object
        if max_suffix >= 0:
            print(f"warning: species name {base_name} already exists, rename to {base_name}.{max_suffix + 1}")
            s.name = f"{base_name}.{max_suffix + 1}"
        else:
            s.name = base_name
        
        # Add the new name to the existing list
        existing_names.append(s.name)

def check_newer_version_on_pypi() -> tuple[str|None, str|None]:
    """Return (current_version, latest_version) for the given package, or (current, None) on error."""
    import importlib.metadata
    try:
        current_version = importlib.metadata.version('lambdapic')
    except Exception:
        current_version = None
    latest_version = None
    try:
        import requests
        resp = requests.get('https://pypi.org/pypi/lambdapic/json', timeout=3)
        if resp.ok:
            data = resp.json()
            latest_version = data['info']['version']
    except ImportError:
        pass  # requests not installed
    except Exception:
        pass  # network or other error
    return current_version, latest_version

def is_version_outdated(local: str, remote: str) -> bool:
    """Return True if local version is older than remote version."""
    try:
        return Version(local) < Version(remote)
    except InvalidVersion:
        return False

def get_num_threads() -> int:
    from threadpoolctl import threadpool_info
    for info in threadpool_info():
        if info['internal_api'] == 'openmp':
            return info['num_threads']
    return 0

def find_divisors(n):
    divisors = set()
    n_abs = abs(n)
    
    for i in range(1, int(np.sqrt(n_abs)) + 1):
        if n_abs % i == 0:
            divisors.add(i)
            divisors.add(n_abs // i)
    
    return sorted(divisors)

def auto_patch_2d(nx: int, ny: int, n_guard: int, cpml_thickness: int, npatch_min: int) -> tuple[int, int]:
    possible_npatch_x = find_divisors(nx)
    possible_npatch_y = find_divisors(ny)

    npatch_x_min = npatch_y_min = npatch_y_max = 2
    npatch_x_max = min(nx//cpml_thickness, nx//(2*n_guard))
    npatch_y_max = min(ny//cpml_thickness, ny//(2*n_guard))
    
    npatches_min = np.inf
    ind_min = (0, 0)
    npatch_xy_diff_min = np.inf # prefer square patch
    for i, npatch_x in enumerate(possible_npatch_x):
        for j, npatch_y in enumerate(possible_npatch_y):
            npatches = npatch_x*npatch_y
            nx_per_patch, ny_per_patch = nx // npatch_x, ny // npatch_y
            
            if npatches < npatch_min:
                continue
            if (npatch_x_min <= npatch_x <= npatch_x_max) or (npatch_y_min <= npatch_y <= npatch_y_max):
                if npatches <= npatches_min and abs(nx_per_patch-ny_per_patch) <= npatch_xy_diff_min:
                    npatches_min = npatches
                    ind_min = i, j
                    npatch_xy_diff_min = abs(nx_per_patch-ny_per_patch)

    i, j = ind_min
    npatch_x, npatch_y = possible_npatch_x[i], possible_npatch_y[j]

    return npatch_x, npatch_y

def auto_patch_3d(nx: int, ny: int, nz: int, n_guard: int, cpml_thickness: int, npatch_min: int) -> tuple[int, int, int]:
    possible_npatch_x = find_divisors(nx)
    possible_npatch_y = find_divisors(ny)
    possible_npatch_z = find_divisors(nz)

    npatch_x_min = npatch_y_min = npatch_z_min = npatch_y_max = npatch_z_max = 2
    npatch_x_max = min(nx//cpml_thickness, nx//(2*n_guard))
    npatch_y_max = min(ny//cpml_thickness, ny//(2*n_guard))
    npatch_z_max = min(nz//cpml_thickness, nz//(2*n_guard))
    
    npatches_min = np.inf
    ind_min = (0, 0, 0)
    npatch_xyz_diff_min = np.inf # prefer cube patch
    for i, npatch_x in enumerate(possible_npatch_x):
        for j, npatch_y in enumerate(possible_npatch_y):
            for k, npatch_z in enumerate(possible_npatch_z):
                npatches = npatch_x*npatch_y*npatch_z
                nx_per_patch, ny_per_patch, nz_per_patch = nx // npatch_x, ny // npatch_y, nz // npatch_z
                
                if npatches < npatch_min:
                    continue
                if (npatch_x_min <= npatch_x <= npatch_x_max) or (npatch_y_min <= npatch_y <= npatch_y_max) or (npatch_z_min <= npatch_z <= npatch_z_max):
                    npatch_xyz_diff = max(abs(nx_per_patch-ny_per_patch), abs(ny_per_patch-nz_per_patch), abs(nz_per_patch-nx_per_patch))
                    if npatches <= npatches_min and npatch_xyz_diff <= npatch_xyz_diff_min:
                        npatches_min = npatches
                        ind_min = i, j, k
                        npatch_xyz_diff_min = npatch_xyz_diff
                    
    i, j, k = ind_min
    npatch_x, npatch_y, npatch_z = possible_npatch_x[i], possible_npatch_y[j], possible_npatch_z[k]

    return npatch_x, npatch_y, npatch_z