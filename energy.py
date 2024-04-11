import numpy as np
import numpy.typing as npt
import itertools
from mdtraj.geometry.distance import compute_distances_core


def flat_bottom_harmonic(
    xyz: np.ndarray, kr: float = 5 * 10**-3, r0: float = 10.0
) -> np.ndarray:

    radial_position = np.linalg.norm(xyz, axis=2)

    return (
        np.heaviside(radial_position - r0, 0)
        * 1
        / 2
        * kr
        * np.square(radial_position - r0)
    )


def LJ_spherical_confinement(
    xyz: np.ndarray,
    radius: float = None,
    density: float = 0.1,
    sigma: float = 1.0,
    epsilon: float = 1.0,
) -> np.ndarray:

    if not radius:  # density only used when radius not specified
        radius = (3 * xyz.shape[1] / (4 * np.pi * density)) ** (
            1 / 3.0
        )

    distance_to_wall = radius - np.linalg.norm(xyz, axis=2)

    return (
        4
        * epsilon
        * (
            (sigma / distance_to_wall) ** 12
            - (sigma / distance_to_wall) ** 6
            + 1 / 4
        )
        * np.heaviside(sigma * 2 ** (1 / 6) - distance_to_wall, 1)
    )


def FENE_bonds(
    xyz: np.ndarray,
    kb: float = 30.0,
    R0: float = 1.5,
) -> np.ndarray:

    bonded_pairs = np.asarray(
        [[i, i + 1] for i in range(xyz.shape[1] - 1)]
    )

    bond_length = compute_distances_core(
        xyz,
        atom_pairs=bonded_pairs,
        periodic=False,
        opt=True,
    )

    return (
        -1 / 2 * kb * R0**2 * np.log(1 - bond_length / R0)
    ) * np.heaviside(R0 - bond_length, 1)


def hard_core(
    xyz: np.ndarray, sigma: float = 1.0, epsilon: float = 1.0
) -> np.ndarray:

    bonded_pairs = np.asarray(
        [[i, i + 1] for i in range(xyz.shape[1] - 1)]
    )

    bond_length = compute_distances_core(
        xyz,
        atom_pairs=bonded_pairs,
        periodic=False,
        opt=True,
    )

    return (
        4
        * epsilon
        * (
            (sigma / bond_length) ** 12
            - (sigma / bond_length) ** 6
            + 1 / 4
        )
        * np.heaviside(sigma * 2 ** (1 / 6) - bond_length, 1)
    )


def angle(
    xyz: np.ndarray, ka: float = 1.0, theta0: float = np.pi
) -> np.ndarray:

    vector_between_bonded_beads = np.diff(xyz, n=1, axis=1)

    angles = np.arccos(
        np.clip(
            np.sum(
                np.multiply(
                    vector_between_bonded_beads[:, :-1, :],
                    vector_between_bonded_beads[:, 1:, :],
                ),
                axis=2,
            )
            / np.linalg.norm(
                vector_between_bonded_beads[:, :-1, :], axis=2
            )
            / np.linalg.norm(
                vector_between_bonded_beads[:, 1:, :], axis=2
            ),
            -1.0,
            1.0,
        )
    )

    return ka * (1 - np.cos(angles - theta0))


def _lennard_jones(rij, epsilon, sigma):
    return (
        4
        * epsilon
        * ((sigma / rij) ** 12 - (sigma / rij) ** 6 + 1 / 4)
    )


def soft_core_intra(
    xyz: np.ndarray,
    sigma: float = 1.0,
    epsilon: float = 1.0,
    Ecut: float = 4.0,
):
    non_bounded_pairs = np.asarray(
        np.triu_indices(xyz.shape[1], k=2)
    ).transpose()

    non_bounded_distances = compute_distances_core(
        xyz,
        atom_pairs=non_bounded_pairs,
        periodic=False,
        opt=True,
    )

    r0 = sigma * ((1 + np.sqrt(Ecut / 2 / epsilon)) / 2) ** (-1 / 6)

    return np.heaviside(
        sigma * 2 ** (1 / 6) - non_bounded_distances, 1
    ) * _lennard_jones(
        non_bounded_distances, sigma, epsilon
    ) * np.heaviside(
        non_bounded_distances - r0, 1
    ) + np.heaviside(
        r0 - non_bounded_distances, 0
    ) * 1 / 2 * Ecut * (
        1
        + np.tanh(
            (
                2
                * _lennard_jones(
                    non_bounded_distances, sigma, epsilon
                )
            )
            / Ecut
            - 1
        )
    )


def energies_homopolymer(
    xyz: np.ndarray,
    sigma,
    epsilon,
    ka: 2,
    kb: float = 30,
    R0: float = 1.5,
    Ecut: float = 4,
    theta0: float = np.pi,
) -> np.ndarray:
    return (
        np.sum(FENE_bonds(xyz, kb, R0), axis=1)
        + np.sum(hard_core(xyz, sigma, epsilon), axis=1)
        + np.sum(angle(xyz, ka, theta0), axis=1)
        + np.sum(soft_core_intra(xyz, sigma, epsilon, Ecut), axis=1)
    )


def _contact_switch(
    rij: float | np.ndarray, mu: float = 3.22, rc: float = 1.78
) -> float | np.ndarray:
    return 1 / 2 * (1 + np.tanh(mu * (rc - rij)))


def ideal_chromosome(
    xyz: np.ndarray,
    dmin: int = 3,
    dmax: int = 500,
    gamma1=-0.030,
    gamma2=-0.351,
    gamma3=-3.737,
    mu: float = 3.22,
    rc: float = 1.78,
) -> np.ndarray:

    upper_tri = set(zip(*np.triu_indices(xyz.shape[1], k=dmin)))
    lower_tri = set(zip(*np.tril_indices(xyz.shape[1], k=dmax)))

    non_bounded_pairs_set = upper_tri & lower_tri

    non_bounded_pairs = np.array(list(non_bounded_pairs_set))

    non_bounded_distances = compute_distances_core(
        xyz,
        atom_pairs=non_bounded_pairs,
        periodic=False,
        opt=True,
    )
    genomic_distance = np.abs(
        np.diff(non_bounded_pairs, axis=1)
    ).reshape(-1)
    gamma = (
        np.divide(gamma1, np.log(genomic_distance))
        + np.divide(gamma2, genomic_distance)
        + np.divide(gamma3, np.square(genomic_distance))
    )

    return (
        _contact_switch(non_bounded_distances, mu, rc)
        * gamma[None, :]
    )


alpha_table = {
    "A1": {
        "A1": -0.268028,
        "A2": -0.274604,
        "B1": -0.262513,
        "B2": -0.258880,
        "B3": -0.266760,
        "B4": -0.266760,
        "NA": -0.225646,
    },
    "A2": {
        "A1": -0.274604,
        "A2": -0.299261,
        "B1": -0.286952,
        "B2": -0.281154,
        "B3": -0.301320,
        "B4": -0.301320,
        "NA": -0.245080,
    },
    "B1": {
        "A1": -0.262513,
        "A2": -0.286952,
        "B1": -0.342020,
        "B2": -0.321726,
        "B3": -0.336630,
        "B4": -0.336630,
        "NA": -0.209919,
    },
    "B2": {
        "A1": -0.258880,
        "A2": -0.281154,
        "B1": -0.321726,
        "B2": -0.330443,
        "B3": -0.329350,
        "B4": -0.329350,
        "NA": -0.282536,
    },
    "B3": {
        "A1": -0.266760,
        "A2": -0.301320,
        "B1": -0.336630,
        "B2": -0.329350,
        "B3": -0.341230,
        "B4": -0.341230,
        "NA": -0.349490,
    },
    "B4": {
        "A1": -0.266760,
        "A2": -0.301320,
        "B1": -0.336630,
        "B2": -0.329350,
        "B3": -0.341230,
        "B4": -0.341230,
        "NA": -0.349490,
    },
    "NA": {
        "A1": -0.225646,
        "A2": -0.245080,
        "B1": -0.209919,
        "B2": -0.282536,
        "B3": -0.349490,
        "B4": -0.349490,
        "NA": -0.255994,
    },
}


def type_to_type_intra(
    xyz: np.ndarray,
    annotations: list,
    alpha_table: dict = alpha_table,
    mu: float = 3.22,
    rc: float = 1.78,
    report_contact_annot: bool = False,
):
    non_bounded_pairs = np.asarray(
        np.triu_indices(xyz.shape[1], k=1)
    ).transpose()

    non_bounded_distances = compute_distances_core(
        xyz,
        atom_pairs=non_bounded_pairs,
        periodic=False,
        opt=True,
    )

    alpha_values = np.asarray(
        [
            alpha_table[annotations[i]][annotations[j]]
            for [i, j] in non_bounded_pairs
        ]
    )

    if report_contact_annot:
        contacts_annot = [
            f"{annotations[i]}:{annotations[j]}"
            for [i, j] in non_bounded_pairs
        ]

        return (
            _contact_switch(non_bounded_distances, mu, rc)
            * alpha_values[None, :]
            * np.heaviside(3 - non_bounded_distances, 1)
        ), contacts_annot

    else:
        return (
            _contact_switch(non_bounded_distances, mu, rc)
            * alpha_values[None, :]
            * np.heaviside(3 - non_bounded_distances, 1)
        )


def type_to_type_inter(
    xyz_ref: np.ndarray,
    xyz_inter: np.ndarray,
    annotations_ref: list,
    annotations_inter: list,
    alpha_table: dict = alpha_table,
    mu: float = 3.22,
    rc: float = 1.78,
    report_contact_annot: bool = False,
):
    xyz = np.concatenate([xyz_ref, xyz_inter], axis=1)
    annotations = annotations_ref.copy()
    annotations.extend(annotations_inter)

    index_ref = [x for x in range(xyz_ref.shape[1])]
    index_inter = [
        xyz_ref.shape[1] + x for x in range(xyz_inter.shape[1])
    ]

    inter_pairs = np.asarray(
        list(itertools.product(index_ref, index_inter))
    )

    inter_distances = compute_distances_core(
        xyz,
        atom_pairs=inter_pairs,
        periodic=False,
        opt=True,
    )

    alpha_values = np.asarray(
        [
            alpha_table[annotations[i]][annotations[j]]
            for [i, j] in inter_pairs
        ]
    )

    if report_contact_annot:
        contacts_annot = [
            f"{annotations[i]}:{annotations[j]}"
            for [i, j] in inter_pairs
        ]

        return (
            _contact_switch(inter_distances, mu, rc)
            * alpha_values[None, :]
            * np.heaviside(3 - inter_distances, 1)
        ), contacts_annot

    else:
        return (
            _contact_switch(inter_distances, mu, rc)
            * alpha_values[None, :]
            * np.heaviside(3 - inter_distances, 1)
        )


def soft_core_inter(
    xyz_ref: np.ndarray,
    xyz_inter: np.ndarray,
    sigma: float = 1.0,
    epsilon: float = 1.0,
    Ecut: float = 4.0,
):
    xyz = np.concatenate([xyz_ref, xyz_inter], axis=1)

    index_ref = [x for x in range(xyz_ref.shape[1])]
    index_inter = [
        xyz_ref.shape[1] + x for x in range(xyz_inter.shape[1])
    ]

    inter_pairs = np.asarray(
        list(itertools.product(index_ref, index_inter))
    )

    inter_distances = compute_distances_core(
        xyz,
        atom_pairs=inter_pairs,
        periodic=False,
        opt=True,
    )

    if np.count_nonzero(inter_distances) != inter_distances.size:
        print("WARNING! There are beads with the same coordinates!")
        print("Lennard-Jones calculation will generate exceptions.")

    r0 = sigma * ((1 + np.sqrt(Ecut / 2 / epsilon)) / 2) ** (-1 / 6)

    return np.heaviside(
        sigma * 2 ** (1 / 6) - inter_distances, 1
    ) * _lennard_jones(
        inter_distances, sigma, epsilon
    ) * np.heaviside(
        inter_distances - r0, 1
    ) + np.heaviside(
        r0 - inter_distances, 0
    ) * 1 / 2 * Ecut * (
        1
        + np.tanh(
            (2 * _lennard_jones(inter_distances, sigma, epsilon))
            / Ecut
            - 1
        )
    )
