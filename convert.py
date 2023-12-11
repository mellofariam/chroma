import os

import numpy as np
import mdtraj as md
import OpenMiChroM.CndbTools as ctools


def cndb2xtc(
    cndb_file: str,
    topfile: str = None,
    filename: str = None,
    frames=[1, None, 1],
):
    if filename is None:
        filename, _ = os.path.splitext(cndb_file)

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

        with open(filename + ".pdb", "w") as pdb_file:
            xyz = positions[0]
            for i in range(len(xyz)):
                j = ["" for _ in range(9)]
                j[0] = "ATOM".ljust(6)  # atom#6s
                j[1] = str(i + 1).rjust(5)  # aomnum#5d
                j[2] = "CA".center(4)  # atomname$#4s

                try:
                    j[3] = chr2aa[annotations[i]].ljust(
                        3
                    )  # resname#1s
                except:
                    j[3] = "ALA".ljust(3)

                j[4] = "A".rjust(1)  # Astring
                j[5] = str(i + 1).rjust(4)  # resnum
                j[6] = str("%8.3f" % (float(xyz[i][0]))).rjust(8)  # x
                j[7] = str("%8.3f" % (float(xyz[i][1]))).rjust(8)  # y
                j[8] = str("%8.3f" % (float(xyz[i][2]))).rjust(8)  # z
                pdb_file.write(
                    "{}{} {} {} {}{}    {}{}{}\n".format(
                        j[0],
                        j[1],
                        j[2],
                        j[3],
                        j[4],
                        j[5],
                        j[6],
                        j[7],
                        j[8],
                    )
                )
            pdb_file.write("END")
        traj = md.load(filename + ".pdb")
        extension = ".pdb"
    else:
        _, extension = os.path.splitext(topfile)
        traj = md.load(topfile)

    if extension != ".gro":
        traj.xyz = positions / 10
    else:
        traj.xyz = positions

    traj.time = np.arange(0, 0.002 * len(positions), 0.002)
    traj.save_xtc(filename + ".xtc")


def build_md_topology(chain_length):
    """
    Creates the topology object to mdtraj for a chromosome, considering it as
    a sequence of CA, in which each bead is a residue.
    """
    topology = md.Topology()
    topology.add_chain()
    for i in range(chain_length):
        topology.add_residue(name="ALA", chain=topology.chain(0))
        topology.insert_atom(
            "CA", md.element.carbon, topology.residue(i)
        )

        try:
            topology.add_bond(topology.atom(i), topology.atom(i - 1))
        except ValueError:
            pass
    return topology


def write_pdb_from_xyz(xyz, filename="frame", resseq=None):
    with open(filename + ".pdb", "w") as pdb_file:
        for i in range(len(xyz)):
            j = ["" for _ in range(9)]
            j[0] = "ATOM".ljust(6)  # atom#6s
            j[1] = str(i + 1).rjust(5)  # aomnum#5d
            j[2] = "CA".center(4)  # atomname$#4s

            if resseq is None:
                j[3] = "ALA".ljust(3)
            else:
                j[3] = resseq[i].ljust(3)  # resname#1s

            j[4] = "A".rjust(1)  # Astring
            j[5] = str(i + 1).rjust(4)  # resnum
            j[6] = str("%8.3f" % (float(xyz[i][0]))).rjust(8)  # x
            j[7] = str("%8.3f" % (float(xyz[i][1]))).rjust(8)  # y
            j[8] = str("%8.3f" % (float(xyz[i][2]))).rjust(8)  # z
            pdb_file.write(
                "{}{} {} {} {}{}    {}{}{}\n".format(
                    j[0],
                    j[1],
                    j[2],
                    j[3],
                    j[4],
                    j[5],
                    j[6],
                    j[7],
                    j[8],
                )
            )
        pdb_file.write("END")


def count2prob(hic_matrix):
    prob_matrix = np.triu(hic_matrix, k=1)

    for i in range(prob_matrix.shape[0] - 1):
        if prob_matrix[i, i + 1]:
            prob_matrix[i, :] /= prob_matrix[
                i, i + 1
            ]  ## normalize by the first neighbor
        else:
            prob_matrix[i, :] = 0.0

    ## correct for the values greater than the 1st neighbor
    for i in range(prob_matrix.shape[0]):
        for j in range(i, prob_matrix.shape[1]):
            if (
                i != j
                and np.abs(i - j) > 1
                and prob_matrix[i, j] > 1.0
            ):
                prob_matrix[i, j] = np.mean(
                    prob_matrix.diagonal(offset=np.abs(i - j))[
                        prob_matrix.diagonal(offset=np.abs(i - j))
                        < 1.0
                    ]
                )

    prob_matrix += prob_matrix.transpose() + np.diag(
        np.ones(prob_matrix.shape[0])
    )

    ## remove centromere:
    idx_to_remove = []
    for i in range(hic_matrix.shape[0]):
        if hic_matrix[i, :].max() == 0:
            idx_to_remove.append(i)

    prob_matrix[idx_to_remove, :] = 0.0
    prob_matrix[:, idx_to_remove] = 0.0

    return prob_matrix
