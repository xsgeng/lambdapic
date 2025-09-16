import os
import subprocess
import sys

import numpy as np
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
import mpi4py

os.environ['CC'] = os.environ.get('CC', 'mpicc')

include_dirs = [np.get_include(), mpi4py.get_include()]
extra_compile_args = ['-Xpreprocessor', '-fopenmp', '-O3', '-march=native', '-ftree-vectorize', '-Werror=incompatible-pointer-types']
extra_link_args = ['-fopenmp']

extensions = [
    Extension(
        name="lambdapic.core.current.cpu2d",
        sources=["src/lambdapic/core/current/cpu2d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="lambdapic.core.current.cpu3d",
        sources=["src/lambdapic/core/current/cpu3d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="lambdapic.core.interpolation.cpu2d",
        sources=["src/lambdapic/core/interpolation/cpu2d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="lambdapic.core.interpolation.cpu3d",
        sources=["src/lambdapic/core/interpolation/cpu3d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="lambdapic.core.sort.cpu2d", 
        sources=["src/lambdapic/core/sort/cpu2d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="lambdapic.core.sort.cpu3d", 
        sources=["src/lambdapic/core/sort/cpu3d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="lambdapic.core.pusher.unified.unified_pusher_2d", 
        sources=["src/lambdapic/core/pusher/unified/unified_pusher_2d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="lambdapic.core.pusher.unified.unified_pusher_3d", 
        sources=["src/lambdapic/core/pusher/unified/unified_pusher_3d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="lambdapic.core.patch.sync_fields2d", 
        sources=["src/lambdapic/core/patch/sync_fields2d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="lambdapic.core.patch.sync_fields3d", 
        sources=["src/lambdapic/core/patch/sync_fields3d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="lambdapic.core.patch.sync_particles_2d", 
        sources=["src/lambdapic/core/patch/sync_particles_2d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="lambdapic.core.patch.sync_particles_3d", 
        sources=["src/lambdapic/core/patch/sync_particles_3d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        name="lambdapic.core.mpi.sync_particles_2d", 
        sources=["src/lambdapic/core/mpi/sync_particles_2d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        libraries=["mpi"],
    ),
    Extension(
        name="lambdapic.core.mpi.sync_particles_3d", 
        sources=["src/lambdapic/core/mpi/sync_particles_3d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        libraries=["mpi"],
    ),
    Extension(
        name="lambdapic.core.mpi.sync_fields2d", 
        sources=["src/lambdapic/core/mpi/sync_fields2d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        libraries=["mpi"],
    ),
    Extension(
        name="lambdapic.core.mpi.sync_fields3d", 
        sources=["src/lambdapic/core/mpi/sync_fields3d.c"],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        libraries=["mpi"],
    ),
]

def _generate_qed_tables():
    for table in ["optical_depth_tables_sigmoid", "optical_depth_tables"]:
        if os.path.exists(f"src/lambdapic/core/qed/{table}.npz"):
            continue
        gen_script = os.path.join(
            os.path.dirname(__file__),
            f"src/lambdapic/core/qed/{table}.py"
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
    entry_points={
    'console_scripts': [
        'lambdapic = lambdapic.cli:app',
        ],
    },
)
