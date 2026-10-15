from collections import namedtuple

ParticleData = namedtuple(
    'ParticleData',
    [ 'x', 'y', 'z', 'ux', 'uy', 'uz', 'inv_gamma', 'w', 'is_dead', 'm', 'q' ]
)
