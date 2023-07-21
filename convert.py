import os

import numpy as np
import mdtraj as md
import OpenMiChroM.CndbTools as ctools


def cndb2xtc(cndb_file: str, topfile: str, output: str = None, frames=[1, None, 1]):
    if output is None:
        output, _ = os.path.splitext(cndb_file)

    _, extension = os.path.splitext(topfile)

    reader = ctools.cndbTools()
    reader.load(filename=cndb_file)
    positions = reader.xyz(frames=frames)

    traj = md.load(topfile)

    if extension != ".gro":
        traj.xyz = positions / 10
    else:
        traj.xyz = positions

    traj.time = np.arange(0, 0.002 * len(positions), 0.002)
    traj.save_xtc(output + ".xtc")
