import collections

import h5py
import numba
import numpy as np


def _build_index_dict(traj_files: dict, chr_range: range, mode: str):
    subcompartments = []
    for chr in chr_range:
        try:
            subcompartments.extend(
                [annot for annot in traj_files[chr]["types"].asstr()]
            )
        except:
            num2annot = {
                0: "A1",
                1: "A2",
                2: "B1",
                3: "B2",
                4: "B3",
                5: "B4",
                6: "NA",
            }
            subcompartments.extend(
                [num2annot[num] for num in traj_files[chr]["types"]]
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

    elif mode == "chr":
        index = {str(chr): [] for chr in chr_range}
        i = 0
        for chr in chr_range:
            index[str(chr)].extend(
                list(range(i, i + len(traj_files[chr]["1"])))
            )
            i += len(traj_files[chr]["1"])

    return index


def lamina_distance(
    nucleus_path: str,
    chr_filename: str = "nucleus_",
    radius: float = None,
    mode: str = "compartments",
    frames: range = None,
    chr_range: range = range(1, 47),
):
    traj_files = {
        chr: h5py.File(
            f"{nucleus_path}/{chr_filename}{chr}.cndb", "r"
        )
        for chr in chr_range
    }

    if frames is None:
        frames = range(1, len(traj_files[chr_range[0]].keys()), 1)

    index = _build_index_dict(traj_files, chr_range, mode)

    distances = {}
    for key in index.keys():
        distances[key] = []

    for n, frame in enumerate(frames):
        if n % 100 == 0:
            print(f"Analyzing frame {frame}", flush=True)
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


def distance_to_center(
    nucleus_path: str,
    chr_filename: str = "nucleus_",
    mode: str = "compartments",
    frames: range = None,
    chr_range: range = range(1, 47),
):
    traj_files = {
        chr: h5py.File(
            f"{nucleus_path}/{chr_filename}{chr}.cndb", "r"
        )
        for chr in chr_range
    }

    if frames is None:
        frames = range(1, len(traj_files[chr_range[0]].keys()), 1)

    index = _build_index_dict(traj_files, chr_range, mode)

    distances = {}
    for key in index.keys():
        distances[key] = []

    for n, frame in enumerate(frames):
        if n % 100 == 0:
            print(f"Analyzing frame {frame}", flush=True)
        frame_nucleus = np.concatenate(
            [
                np.array(traj_files[chr][str(frame)])
                for chr in chr_range
            ]
        )

        distance_to_center = np.linalg.norm(
            frame_nucleus - np.mean(frame_nucleus, axis=0), axis=1
        )

        for key in index.keys():
            distances[key].extend(distance_to_center[index[key]])

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

    index = _build_index_dict(traj_files, chr_range, mode)

    radial_density = {}
    for key in index.keys():
        radial_density[key] = np.zeros(nbins)

    for n, frame in enumerate(frames):
        if n % 100 == 0:
            print(f"Analyzing frame {frame}", flush=True)

        frame_nucleus = np.concatenate(
            [
                np.array(traj_files[chr][str(frame)])
                for chr in chr_range
            ]
        )

        distance_to_center = np.linalg.norm(
            frame_nucleus - np.mean(frame_nucleus, axis=0), axis=1
        )

        bin_numbers = np.floor(distance_to_center / radius * nbins)
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


@numba.njit(fastmath=True, parallel=True)
def compute_distances(A, B):
    """
    Refer to https://github.com/numba/numba-scipy/issues/38
    """

    assert A.shape[1] == B.shape[1]
    C = np.empty((A.shape[0], B.shape[0]), A.dtype)

    # workaround to get the right datatype for acc
    init_val_arr = np.zeros(1, A.dtype)
    init_val = init_val_arr[0]

    for i in numba.prange(A.shape[0]):
        for j in range(B.shape[0]):
            acc = init_val
            for k in range(A.shape[1]):
                acc += (A[i, k] - B[j, k]) ** 2
            C[i, j] = np.sqrt(acc)
    return C


@numba.njit(fastmath=True, parallel=True)
def compute_distances_trajectory(A, B):
    assert (A.shape[0], A.shape[2]) == (B.shape[0], B.shape[2])

    C = np.empty((A.shape[0], A.shape[1], B.shape[1]), A.dtype)

    # workaround to get the right datatype for acc
    init_val_arr = np.zeros(1, A.dtype)
    init_val = init_val_arr[0]

    for t in numba.prange(A.shape[0]):
        for i in numba.prange(A.shape[1]):
            for j in range(B.shape[1]):
                acc = init_val
                for k in range(A.shape[2]):
                    acc += (A[t, i, k] - B[t, j, k]) ** 2
                C[t, i, j] = np.sqrt(acc)
    return C


@numba.njit(fastmath=True, parallel=True)
def compute_similarity_from_distances(pairwise_distances, sigma):
    """Compute similarity, Q, when the pairwise distances were already computed."""
    n_frames = pairwise_distances.shape[0]
    n_pairs = pairwise_distances.shape[1]

    q = np.empty(n_frames, dtype=pairwise_distances.dtype)
    for lag in numba.prange(n_frames):

        q_lag = np.empty(
            n_frames - lag, dtype=pairwise_distances.dtype
        )

        for start in numba.prange(n_frames - lag):
            total = 0.0
            for i in numba.prange(n_pairs):
                total += np.exp(
                    -1
                    / 2
                    * (
                        (
                            (
                                pairwise_distances[start + lag, i]
                                - pairwise_distances[start, i]
                            )
                            ** 2
                        )
                        / (sigma**2)
                    )
                )

            q_lag[start] = total / n_pairs

        q[lag] = np.mean(q_lag)

    return q


@numba.njit(fastmath=True)
def _different_replicas(frame1, frame2, frames_per_replica):
    rep1 = (frame1 // frames_per_replica) + 1
    rep2 = (frame2 // frames_per_replica) + 1

    if rep1 == rep2:
        return False
    else:
        return True


@numba.njit(fastmath=True, parallel=True)
def compute_similarity_from_distances_between_replicas(
    pairwise_distances, sigma, frames_per_replica
):

    n_frames = pairwise_distances.shape[0]
    n_pairs = pairwise_distances.shape[1]

    i = 0
    indexed_frame_pair = {}
    for frame1 in range(n_frames):
        for frame2 in range(frame1 + 1, n_frames):
            if _different_replicas(
                frame1, frame2, frames_per_replica=frames_per_replica
            ):
                indexed_frame_pair[i] = (frame1, frame2)
                i += 1

    n_similarity_values = len(indexed_frame_pair.keys())
    similarity = np.empty(
        n_similarity_values, dtype=pairwise_distances.dtype
    )
    for i in numba.prange(n_similarity_values):
        frame1, frame2 = indexed_frame_pair[numba.int64(i)]

        total = 0.0
        for pair in numba.prange(n_pairs):
            total += np.exp(
                -1
                / 2
                * (
                    (
                        (
                            pairwise_distances[frame1, pair]
                            - pairwise_distances[frame2, pair]
                        )
                        ** 2
                    )
                    / (sigma**2)
                )
            )

        similarity[i] = total / n_pairs

    return similarity


@numba.njit(fastmath=True, parallel=True)
def _get_from_matrix(matrix, row_indices, col_indices):
    n_elements = len(row_indices)
    array_of_values = np.empty(n_elements, dtype=matrix.dtype)

    for i in numba.prange(n_elements):
        array_of_values[i] = matrix[row_indices[i], col_indices[i]]

    return array_of_values


@numba.njit(fastmath=True, parallel=True)
def compute_similarity_from_positions(
    positions, sigma, starting_neighbor=2
):
    """
    Compute similarity, Q, directly from the positions.
    Useful when storing the distances is not possible due to the number of beads.
    """

    n_frames = positions.shape[0]

    similarity = np.zeros(n_frames)
    normalize_by = np.zeros(n_frames)

    i, j = np.triu_indices(positions.shape[1], k=starting_neighbor)
    n_pairs = i.size

    for start in numba.prange(n_frames):
        pairwise_distances_start = compute_distances(
            positions[start], positions[start]
        )
        pairwise_distances_start = _get_from_matrix(
            pairwise_distances_start, i, j
        )

        for lag in numba.prange(n_frames - start):
            pairwise_distances_lag = compute_distances(
                positions[start + lag], positions[start + lag]
            )
            pairwise_distances_lag = _get_from_matrix(
                pairwise_distances_lag, i, j
            )

            total = 0.0
            for pair in range(n_pairs):
                total += np.exp(
                    -1
                    / 2
                    * (
                        (
                            (
                                pairwise_distances_lag[pair]
                                - pairwise_distances_start[pair]
                            )
                            ** 2
                        )
                        / (sigma**2)
                    )
                )

            similarity[lag] += total / n_pairs
            normalize_by[lag] += 1

    similarity = np.divide(similarity, normalize_by)

    return similarity
