import os
import subprocess
import sys

import numpy as np
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

include_dirs = [np.get_include()]
extra_compile_args = ['-Xpreprocessor', '-fopenmp', '-O3', '-march=native', '-ftree-vectorize']
extra_link_args = ['-fopenmp']

extensions = [
    Extension(
        name="lambdapic.core.current.cpu2d",
        sources=["lambdapic/core/current/cpu2d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="lambdapic.core.current.cpu3d",
        sources=["lambdapic/core/current/cpu3d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="lambdapic.core.interpolation.cpu2d",
        sources=["lambdapic/core/interpolation/cpu2d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="lambdapic.core.interpolation.cpu3d",
        sources=["lambdapic/core/interpolation/cpu3d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="lambdapic.core.sort.cpu2d", 
        sources=["lambdapic/core/sort/cpu2d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="lambdapic.core.sort.cpu3d", 
        sources=["lambdapic/core/sort/cpu3d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="lambdapic.core.pusher.unified.unified_pusher_2d", 
        sources=["lambdapic/core/pusher/unified/unified_pusher_2d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="lambdapic.core.pusher.unified.unified_pusher_3d", 
        sources=["lambdapic/core/pusher/unified/unified_pusher_3d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="lambdapic.core.patch.sync_fields2d", 
        sources=["lambdapic/core/patch/sync_fields2d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="lambdapic.core.patch.sync_fields3d", 
        sources=["lambdapic/core/patch/sync_fields3d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="lambdapic.core.patch.sync_particles_2d", 
        sources=["lambdapic/core/patch/sync_particles_2d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="lambdapic.core.patch.sync_particles_3d", 
        sources=["lambdapic/core/patch/sync_particles_3d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="lambdapic.core.mpi.sync_particles_2d", 
        sources=["lambdapic/core/mpi/sync_particles_2d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        libraries=["mpi"],
    ),
    Extension(
        name="lambdapic.core.mpi.sync_particles_3d", 
        sources=["lambdapic/core/mpi/sync_particles_3d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        libraries=["mpi"],
    ),
    Extension(
        name="lambdapic.core.mpi.sync_fields2d", 
        sources=["lambdapic/core/mpi/sync_fields2d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        libraries=["mpi"],
    ),
    Extension(
        name="lambdapic.core.mpi.sync_fields3d", 
        sources=["lambdapic/core/mpi/sync_fields3d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        libraries=["mpi"],
    ),
]

def _generate_qed_tables():
    for table in ["optical_depth_tables_sigmoid", "optical_depth_tables"]:
        if os.path.exists(f"lambdapic/core/qed/{table}.npz"):
            continue
        gen_script = os.path.join(
            os.path.dirname(__file__),
            f"lambdapic/core/qed/{table}.py"
        )
    
        cmd = [
            sys.executable,
            gen_script,
        ]

        print(f"\n🚀 Generating {table}:")
        print(" ".join(cmd))
        subprocess.check_call(cmd)
        
_generate_qed_tables()

setup(
    name="lambdapic",
    ext_modules=extensions,
)