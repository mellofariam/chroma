import h5py
import numpy as np


def lamina_distance(
    nucleus_path: str,
    radius: float = None,
    mode: str = "compartments",
    frames: range = None,
    chr_range: range = range(1, 47),
):
    traj_files = {
        chr: h5py.File(f"{nucleus_path}/chromosome{chr}.cndb", "r") for chr in chr_range
    }

    if frames is None:
        frames = range(1, len(traj_files[chr_range[0]].keys()), 1)

    subcompartments = []
    for chr in chr_range:
        subcompartments.extend([annot for annot in traj_files[chr]["types"].asstr()])

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
        index = {"A1": [], "A2": [], "B1": [], "B2": [], "B3": [], "B4": [], "NA": []}
        for idx, annot in enumerate(subcompartments):
            index[annot].append(idx)

    distances = {}
    for key in index.keys():
        distances[key] = []

    for n, frame in enumerate(frames):
        if n % 100 == 0:
            print(f"Analyzing frame {frame}")
        frame_nucleus = np.concatenate(
            [np.array(traj_files[chr][str(frame)]) for chr in chr_range]
        )

        frame_distances = radius - np.linalg.norm(frame_nucleus, axis=1)
        for key in index.keys():
            distances[key].extend(frame_distances[index[key]])

    return distances
