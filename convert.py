import os

import numpy as np
import mdtraj as md
import OpenMiChroM.CndbTools as ctools


def cndb2xtc(cndb_file: str, topfile: str = None, output: str = None, frames=[1, None, 1]):
    if output is None:
        output, _ = os.path.splitext(cndb_file)

    _, extension = os.path.splitext(topfile)

    reader = ctools.cndbTools()
    reader.load(filename=cndb_file)
    positions = reader.xyz(frames=frames)

    if topfile is None:
        annotations = reader.ChromSeq
        chr2aa = {
            "A1": "ASP",
            "A2": "GLU",
            "B1": "ARG",
            "B2": "LYS",
            "B3": "HIS",
            "B4": "HIS",
            "NA": "GLY",
        }

        with open(output + ".pdb", "w") as pdb_file:
            for i in range(len(positions[0])):
                j = ["" for _ in range(9)]
                j[0] = "ATOM".ljust(6)  # atom#6s
                j[1] = str(i + 1).rjust(5)  # aomnum#5d
                j[2] = "CA".center(4)  # atomname$#4s

                try:
                    j[3] = chr2aa[annotations[i]].ljust(3)  # resname#1s
                except:
                    j[3] = "ALA".ljust(3)
                    
                j[4] = "A".rjust(1)  # Astring
                j[5] = str(i + 1).rjust(4)  # resnum
                j[6] = str("%8.3f" % (float(positions[0][i][0]))).rjust(8)  # x
                j[7] = str("%8.3f" % (float(positions[0][i][1]))).rjust(8)  # y
                j[8] = str("%8.3f" % (float(positions[0][i][2]))).rjust(8)  # z
                pdb_file.write(
                    "{}{} {} {} {}{}    {}{}{}\n".format(
                        j[0], j[1], j[2], j[3], j[4], j[5], j[6], j[7], j[8]
                    )
                )
            pdb_file.write("END")
        traj = md.load(output + ".pdb")
    else:
        traj = md.load(topfile)

    if extension != ".gro":
        traj.xyz = positions / 10
    else:
        traj.xyz = positions

    traj.time = np.arange(0, 0.002 * len(positions), 0.002)
    traj.save_xtc(output + ".xtc")
