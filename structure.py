import h5py
import numpy as np
import collections


def lamina_distance(
    nucleus_path: str,
    radius: float = None,
    mode: str = "compartments",
    frames: range = None,
    chr_range: range = range(1, 47),
):
    traj_files = {
        chr: h5py.File(f"{nucleus_path}/chromosome{chr}.cndb", "r")
        for chr in chr_range
    }

    if frames is None:
        frames = range(1, len(traj_files[chr_range[0]].keys()), 1)

    subcompartments = []
    for chr in chr_range:
        subcompartments.extend(
            [annot for annot in traj_files[chr]["types"].asstr()]
        )

    if mode == "compartments":
        index = {"A": [], "B": [], "NA": []}
        to_compt = {
            "A1": "A",
            "A2": "A",
            "B1": "B",
            "B2": "B",
            "B3": "B",
            "B4": "B",
            "NA": "NA",
        }
        for idx, annot in enumerate(subcompartments):
            index[to_compt[annot]].append(idx)

    elif mode == "subcompartments":
        index = {
            "A1": [],
            "A2": [],
            "B1": [],
            "B2": [],
            "B3": [],
            "B4": [],
            "NA": [],
        }
        for idx, annot in enumerate(subcompartments):
            index[annot].append(idx)

    distances = {}
    for key in index.keys():
        distances[key] = []

    for n, frame in enumerate(frames):
        if n % 100 == 0:
            print(f"Analyzing frame {frame}")
        frame_nucleus = np.concatenate(
            [
                np.array(traj_files[chr][str(frame)])
                for chr in chr_range
            ]
        )

        frame_distances = radius - np.linalg.norm(
            frame_nucleus, axis=1
        )
        for key in index.keys():
            distances[key].extend(frame_distances[index[key]])

    return distances


def radial_density(
    nucleus_path: str,
    chr_filename: str = "nucleus_",
    radius: float = None,
    mode: str = "compartments",
    frames: range = None,
    chr_range: range = range(46),
    nbins: int = 100,
):
    traj_files = {
        chr: h5py.File(
            f"{nucleus_path}/{chr_filename}{chr}.cndb", "r"
        )
        for chr in chr_range
    }

    if frames is None:
        frames = range(1, len(traj_files[chr_range[0]].keys()), 1)

    subcompartments = []
    for chr in chr_range:
        subcompartments.extend(
            [annot for annot in traj_files[chr]["types"].asstr()]
        )

    if mode == "compartments":
        index = {"A": [], "B": [], "NA": []}
        to_compt = {
            "A1": "A",
            "A2": "A",
            "B1": "B",
            "B2": "B",
            "B3": "B",
            "B4": "B",
            "NA": "NA",
        }
        for idx, annot in enumerate(subcompartments):
            index[to_compt[annot]].append(idx)

    elif mode == "subcompartments":
        index = {
            "A1": [],
            "A2": [],
            "B1": [],
            "B2": [],
            "B3": [],
            "B4": [],
            "NA": [],
        }
        for idx, annot in enumerate(subcompartments):
            index[annot].append(idx)

    elif mode == "all":
        index = {"all": [i for i, _ in enumerate(subcompartments)]}

    radial_density = {}
    for key in index.keys():
        radial_density[key] = np.zeros(nbins)

    for n, frame in enumerate(frames):
        if n % 100 == 0:
            print(f"Analyzing frame {frame}")

        frame_nucleus = np.concatenate(
            [
                np.array(traj_files[chr][str(frame)])
                for chr in chr_range
            ]
        )

        distance_to_center = np.linalg.norm(
            frame_nucleus - np.mean(frame_nucleus, axis=0), axis=1
        )

        bin_numbers = np.ceil(distance_to_center / radius * nbins)
        dr = radius / nbins

        for key in index.keys():
            beads_num = collections.Counter(bin_numbers[index[key]])
            for bin_id in beads_num.keys():
                if bin_id < nbins:
                    radial_density[key][int(bin_id)] += np.divide(
                        beads_num[bin_id],
                        4 * np.pi * dr * ((bin_id + 1) * dr) ** 2,
                    )

    for key in index.keys():
        radial_density[key] /= len(frames)

    return radial_density


def reduce_resolution(traj, divide_by=20):
    print(
        f"Reducing resolution by a factor of {divide_by}", flush=True
    )

    temp = traj[:, : (traj.shape[1] // divide_by) * divide_by, :]

    temp = temp.reshape(
        (temp.shape[0], temp.shape[1] // divide_by, divide_by, 3)
    )
    coarsed_traj = np.mean(temp, axis=2)

    if traj.shape[1] % divide_by != 0:
        rest = traj[:, (traj.shape[1] // divide_by) * divide_by :, :]
        rest = np.reshape(
            np.mean(rest, axis=1), (traj.shape[0], 1, 3)
        )
        coarsed_traj = np.concatenate((coarsed_traj, rest), axis=1)

    return coarsed_traj
