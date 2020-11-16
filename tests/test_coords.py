#!/usr/bin/env python3

"""Tests for module coords."""

import numpy as np
import pytest

from overreact import coords
from overreact.datasets import logfiles


# TODO(schneiderfelipe): add one extra atom
def test_can_understand_K_symmetry():
    """Ensure values match regression logfiles for K symmetry."""
    data = logfiles["tanaka1996"]["Cl·@UMP2/6-311G(2df,2pd)"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([0.0, 0.0, 0.0])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 1
    assert len(groups[0]) == 1
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("spheric", "atomic")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 0
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 0
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 0
    assert coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "K"
    assert coords.symmetry_number(point_group) == 1


def test_can_understand_C1_symmetry():
    """Ensure values match regression logfiles for C1 symmetry."""
    data = logfiles["symmetries"]["chlorobromofluoromethane"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([81.70347257, 264.62028172, 335.60557643])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 5
    assert len(groups[0]) == 1
    assert len(groups[1]) == 1
    assert len(groups[2]) == 1
    assert len(groups[3]) == 1
    assert len(groups[4]) == 1
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("asymmetric", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 0
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 0
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 0
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "C1"
    assert coords.symmetry_number(point_group) == 1


def test_can_understand_Cs_symmetry():
    """Ensure values match regression logfiles for Cs symmetry."""
    data = logfiles["symmetries"]["NHF2"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([9.58233074, 49.04289888, 56.82386749])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 3
    assert len(groups[0]) == 1
    assert len(groups[1]) == 1
    assert len(groups[2]) == 2
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("asymmetric", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 0
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 0
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 1
    assert mirror_axes[0][0] == ""
    assert mirror_axes[0][1] == pytest.approx([1.0, 0.0, 0.0])
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "Cs"
    assert coords.symmetry_number(point_group) == 1

    data = logfiles["symmetries"]["1-bromo-2-chloroethylene"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([17.61945078, 253.37069181, 267.61052366])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 4
    assert len(groups[0]) == 1
    assert len(groups[1]) == 1
    assert len(groups[2]) == 1
    assert len(groups[3]) == 2
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("asymmetric", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 0
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 0
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 1
    assert mirror_axes[0][0] == ""
    assert mirror_axes[0][1] == pytest.approx([0.0, 0.0, 1.0])
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "Cs"
    assert coords.symmetry_number(point_group) == 1

    data = logfiles["symmetries"]["1-iodo-2-chloroethylene"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([18.69862033, 334.44171615, 349.75324294])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 4
    assert len(groups[0]) == 1
    assert len(groups[1]) == 1
    assert len(groups[2]) == 1
    assert len(groups[3]) == 2
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("asymmetric", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 0
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 0
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 1
    assert mirror_axes[0][0] == ""
    assert mirror_axes[0][1] == pytest.approx([0.0, 0.0, 1.0])
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "Cs"
    assert coords.symmetry_number(point_group) == 1


def test_can_understand_Ci_symmetry():
    """Ensure values match regression logfiles for Ci symmetry."""
    data = logfiles["symmetries"]["1,2-dichloro-1,2-difluoroethane"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([130.04075032, 358.98131538, 473.66138286])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    # assert axes == pytest.approx(
    #     np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    # )
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 4
    assert len(groups[0]) == 2
    assert len(groups[1]) == 2
    assert len(groups[2]) == 2
    assert len(groups[3]) == 2
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("asymmetric", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 0
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 0
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 0
    assert coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "Ci"
    assert coords.symmetry_number(point_group) == 1

    data = logfiles["symmetries"]["meso-tartaric-acid"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([213.53202466, 543.08552098, 732.14870909])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 8
    assert len(groups[0]) == 2
    assert len(groups[1]) == 2
    assert len(groups[2]) == 2
    assert len(groups[3]) == 2
    assert len(groups[4]) == 2
    assert len(groups[5]) == 2
    assert len(groups[6]) == 2
    assert len(groups[7]) == 2
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("asymmetric", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 0
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 0
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 0
    assert coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "Ci"
    assert coords.symmetry_number(point_group) == 1


def test_can_understand_Cinfv_symmetry():
    """Ensure values match regression logfiles for C∞v symmetry."""
    data = logfiles["tanaka1996"]["HCl@UMP2/6-311G(2df,2pd)"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([0.0, 1.58676025, 1.58676025], 5e-3)
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 2
    assert len(groups[0]) == 1
    assert len(groups[1]) == 1
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric prolate", "linear")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 0
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 0
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 0
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "C∞v"
    assert coords.symmetry_number(point_group) == 1

    data = logfiles["symmetries"]["SCO"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([1.17654558e-8, 8.53818341e1, 8.53818341e1])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    # assert axes == pytest.approx(
    #     np.array(
    #         [[1.0, 0.0, 0.0], [0.0, -1.0, -8.61130797e-7], [0.0, 8.61130797e-7, -1.0]]
    #     )
    # )
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 3
    assert len(groups[0]) == 1
    assert len(groups[1]) == 1
    assert len(groups[2]) == 1
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric prolate", "linear")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 0
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 0
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 0
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "C∞v"
    assert coords.symmetry_number(point_group) == 1


def test_can_understand_Dinfh_symmetry():
    """Ensure values match regression logfiles for D∞h symmetry."""
    data = logfiles["symmetries"]["dihydrogen"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([0.0, 2.96199592e-1, 2.96199592e-1])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    # assert axes == pytest.approx(
    #     np.array(
    #         [
    #             [-1.0, 0.0, 0.0],
    #             [0.0, -9.85287297e-1, 1.70906240e-1],
    #             [0.0, 1.70906240e-1, 9.85287297e-1],
    #         ]
    #     )
    # )
    assert atomcoords == pytest.approx(
        np.array([[3.83307188e-1, 0.0, 0.0], [-3.83307188e-1, 0.0, 0.0]])
    )
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 1
    assert len(groups[0]) == 2
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric prolate", "linear")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 1
    assert proper_axes[0][0] == 2
    # assert proper_axes[0][1] == pytest.approx([-1.0, 0.0, 0.0])
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 1
    assert improper_axes[0][0] == 4
    assert improper_axes[0][1] == proper_axes[0][1]
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 1
    assert mirror_axes[0][0] == "h"
    # assert mirror_axes[0][1] == pytest.approx([-1.0, 0.0, 0.0])
    assert coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "D∞h"
    assert coords.symmetry_number(point_group) == 2

    data = logfiles["symmetries"]["carbon-dioxide"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([8.94742236e-8, 4.44644189e1, 4.44644190e1])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    # assert axes == pytest.approx(
    #     np.array(
    #         [[1.0, 0.0, 0.0], [0.0, -1.0, 2.45551350e-7], [0.0, -2.45551349e-7, -1.0]]
    #     )
    # )
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 2
    assert len(groups[0]) == 1
    assert len(groups[1]) == 2
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric prolate", "linear")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 1
    assert proper_axes[0][0] == 2
    assert proper_axes[0][1] == pytest.approx([1.0, 0.0, 0.0])
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 1
    assert improper_axes[0][0] == 4
    assert improper_axes[0][1] == proper_axes[0][1]
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 1
    assert mirror_axes[0][0] == "h"
    assert mirror_axes[0][1] == pytest.approx([1.0, 0.0, 0.0])
    assert coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "D∞h"
    assert coords.symmetry_number(point_group) == 2


def test_can_understand_C2_symmetry():
    """Ensure values match regression logfiles for C2 symmetry."""
    data = logfiles["symmetries"]["hydrogen-peroxide"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([1.74210646, 19.61466369, 20.420849])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    # assert axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 2
    assert len(groups[0]) == 2
    assert len(groups[1]) == 2
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("asymmetric", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 1
    assert proper_axes[0][0] == 2
    # assert proper_axes[0][1] == pytest.approx([0.0, 0.0, 1.0])
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 0
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 0
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "C2"
    assert coords.symmetry_number(point_group) == 2

    data = logfiles["symmetries"]["hydrazine"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([3.48031691, 20.67234093, 20.67777505])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    # assert axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 3
    assert len(groups[0]) == 2
    assert len(groups[1]) == 2
    assert len(groups[2]) == 2
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric prolate", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 1
    assert proper_axes[0][0] == 2
    assert proper_axes[0][1] == pytest.approx(
        [-6.265379700455943e-5, 0.0388451244158036, -0.9992452413615102]
    )
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 0
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 0
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "C2"
    assert coords.symmetry_number(point_group) == 2


def test_can_understand_C3_symmetry():
    """Ensure values match regression logfiles for C3 symmetry."""
    data = logfiles["symmetries"]["H3PO4"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([117.15458225, 119.69622329, 119.71729381])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    # assert axes == pytest.approx(
    #     np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    # )
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 4
    assert len(groups[0]) == 1
    assert len(groups[1]) == 1
    assert len(groups[2]) == 3
    assert len(groups[3]) == 3
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric prolate", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 1
    assert proper_axes[0][0] == 3
    assert proper_axes[0][1] == pytest.approx([1.0, 0.0, 0.0])
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 0
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 0
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "C3"
    assert coords.symmetry_number(point_group) == 3

    data = logfiles["symmetries"]["triphenylphosphine"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([1398.63089282, 1403.26185999, 2362.23380009])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(
        np.array(
            [[1.0, 3.60567321e-14, 0.0], [-3.60567321e-14, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
    )
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 12
    assert len(groups[0]) == 1
    assert len(groups[1]) == 3
    assert len(groups[3]) == 3
    assert len(groups[3]) == 3
    assert len(groups[4]) == 3
    assert len(groups[5]) == 3
    assert len(groups[6]) == 3
    assert len(groups[7]) == 3
    assert len(groups[8]) == 3
    assert len(groups[9]) == 3
    assert len(groups[10]) == 3
    assert len(groups[11]) == 3
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric oblate", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 1
    assert proper_axes[0][0] == 3
    assert proper_axes[0][1] == pytest.approx([0.0, 0.0, 1.0])
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 0
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 0
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "C3"
    assert coords.symmetry_number(point_group) == 3


def test_can_understand_C2h_symmetry():
    """Ensure values match regression logfiles for C2h symmetry."""
    data = logfiles["symmetries"]["trans-1,2-dichloroethylene"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([9.8190931, 342.02181465, 351.84090775])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 3
    assert len(groups[0]) == 2
    assert len(groups[1]) == 2
    assert len(groups[2]) == 2
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("asymmetric", "irregular planar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 1
    assert proper_axes[0][0] == 2
    assert proper_axes[0][1] == pytest.approx([0.0, 0.0, 1.0])
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 1
    assert improper_axes[0][0] == 2
    assert improper_axes[0][1] == proper_axes[0][1]
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 1
    assert mirror_axes[0][0] == "h"
    assert mirror_axes[0][1] == pytest.approx([0.0, 0.0, 1.0])
    assert coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "C2h"
    assert coords.symmetry_number(point_group) == 2

    data = logfiles["symmetries"]["transplatin"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([154.36235242, 392.59853004, 541.53866791])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(
        np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    )
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 4
    assert len(groups[0]) == 1
    assert len(groups[1]) == 2
    assert len(groups[2]) == 2
    assert len(groups[3]) == 6
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("asymmetric", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 1
    assert proper_axes[0][0] == 2
    assert proper_axes[0][1] == pytest.approx([0.0, 0.0, -1.0])
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 1
    assert improper_axes[0][0] == 2
    assert improper_axes[0][1] == proper_axes[0][1]
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 1
    assert mirror_axes[0][0] == "h"
    assert mirror_axes[0][1] == pytest.approx([0.0, 0.0, -1.0])
    assert coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    # TODO(schneiderfelipe): people say this should be D2h, but it is
    # impossible if we consider hydrogen atoms: add an option to
    # find_point_group that ignores hydrogen atoms.
    assert point_group == "C2h"
    assert coords.symmetry_number(point_group) == 2


def test_can_understand_C3h_symmetry():
    """Ensure values match regression logfiles for C3h symmetry."""
    data = logfiles["symmetries"]["boric-acid"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([52.01309348, 52.01530317, 104.02839606])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(np.eye(3), abs=1e-6)
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 3
    assert len(groups[0]) == 1
    assert len(groups[1]) == 3
    assert len(groups[2]) == 3
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric oblate", "regular planar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 1
    assert proper_axes[0][0] == 3
    assert proper_axes[0][1] == pytest.approx([0.0, 0.0, 1.0])
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 1
    assert improper_axes[0][0] == 3
    assert improper_axes[0][1] == proper_axes[0][1]
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 1
    assert mirror_axes[0][0] == "h"
    assert mirror_axes[0][1] == pytest.approx([0.0, 0.0, 1.0])
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "C3h"
    assert coords.symmetry_number(point_group) == 3


def test_can_understand_C2v_symmetry():
    """Ensure values match regression logfiles for C2v symmetry."""
    data = logfiles["symmetries"]["water"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([0.6768072475, 1.1582103375, 1.835017585])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 2
    assert len(groups[0]) == 1
    assert len(groups[1]) == 2
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("asymmetric", "irregular planar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 1
    assert proper_axes[0][0] == 2
    assert proper_axes[0][1] == pytest.approx([0.0, 1.0, 0.0])
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 0
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 2
    assert mirror_axes[0][0] == "v"
    assert mirror_axes[1][0] == "v"
    assert mirror_axes[0][1] == pytest.approx([1.0, 0.0, 0.0])
    assert mirror_axes[1][1] == pytest.approx([0.0, 0.0, 1.0])
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "C2v"
    assert coords.symmetry_number(point_group) == 2

    data = logfiles["symmetries"]["SF4"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([81.52806583, 133.34202281, 167.8488049])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 3
    assert len(groups[0]) == 1
    assert len(groups[1]) == 2
    assert len(groups[2]) == 2
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("asymmetric", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 1
    assert proper_axes[0][0] == 2
    assert proper_axes[0][1] == pytest.approx([0.0, 0.0, 1.0])
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 0
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 2
    assert mirror_axes[0][0] == "v"
    assert mirror_axes[1][0] == "v"
    assert mirror_axes[0][1] == pytest.approx([1.0, 0.0, 0.0])
    assert mirror_axes[1][1] == pytest.approx([0.0, 1.0, 0.0])
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "C2v"
    assert coords.symmetry_number(point_group) == 2

    data = logfiles["symmetries"]["cyclohexane-boat"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([119.38090371, 123.2008681, 206.20634797])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 5
    assert len(groups[0]) == 2
    assert len(groups[1]) == 2
    assert len(groups[2]) == 4
    assert len(groups[3]) == 4
    # TODO(schneiderfelipe): I believe the following group should be split into
    # two groups, one of 2 atoms and one of 4 atoms:
    assert len(groups[4]) == 6
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric oblate", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 1
    assert proper_axes[0][0] == 2
    assert proper_axes[0][1] == pytest.approx([0.0, 0.0, 1.0])
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 0
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 2
    assert mirror_axes[0][0] == "v"
    assert mirror_axes[1][0] == "v"
    assert mirror_axes[0][1] == pytest.approx(
        [-3.09874756655998e-7, -0.9999999999999518, -1.85084408834979e-8]
    )
    assert mirror_axes[1][1] == pytest.approx(
        [-0.999999999999952, 3.098747565480977e-7, 3.719778376187453e-9]
    )
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "C2v"
    assert coords.symmetry_number(point_group) == 2

    data = logfiles["symmetries"]["cisplatin"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([231.33596051, 300.06695463, 525.9371719])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 5
    assert len(groups[0]) == 1
    assert len(groups[1]) == 2
    assert len(groups[2]) == 2
    assert len(groups[3]) == 2
    assert len(groups[4]) == 4
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("asymmetric", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 1
    assert proper_axes[0][0] == 2
    assert proper_axes[0][1] == pytest.approx([0.0, 1.0, 0.0])
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 0
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 2
    assert mirror_axes[0][0] == "v"
    assert mirror_axes[1][0] == "v"
    assert mirror_axes[0][1] == pytest.approx([1.0, 0.0, 0.0])
    assert mirror_axes[1][1] == pytest.approx([0.0, 0.0, 1.0])
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "C2v"
    assert coords.symmetry_number(point_group) == 2

    data = logfiles["symmetries"]["1,2-dichlorobenzene"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([266.78313761, 355.80034163, 622.58347924])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 6
    assert len(groups[0]) == 2
    assert len(groups[1]) == 2
    assert len(groups[2]) == 2
    assert len(groups[3]) == 2
    assert len(groups[4]) == 2
    assert len(groups[5]) == 2
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("asymmetric", "irregular planar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 1
    assert proper_axes[0][0] == 2
    assert proper_axes[0][1] == pytest.approx([1.0, 0.0, 0.0])
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 0
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 2
    assert mirror_axes[0][0] == "v"
    assert mirror_axes[1][0] == "v"
    assert mirror_axes[0][1] == pytest.approx([0.0, 1.0, 0.0])
    assert mirror_axes[1][1] == pytest.approx([0.0, 0.0, 1.0])
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "C2v"
    assert coords.symmetry_number(point_group) == 2

    data = logfiles["symmetries"]["1,3-dichlorobenzene"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([179.02244122, 596.70030705, 775.72274827])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 5
    assert len(groups[0]) == 1
    assert len(groups[1]) == 2
    # TODO(schneiderfelipe): I believe the following should be two groups, one
    # of two atoms, one of one atom each
    assert len(groups[2]) == 3
    # TODO(schneiderfelipe): I believe the following should be two groups, one
    # of two atoms, one of one atom each
    assert len(groups[3]) == 3
    # TODO(schneiderfelipe): I believe the following should be two groups, one
    # of two atoms, one of one atom each
    assert len(groups[4]) == 3
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("asymmetric", "irregular planar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 1
    assert proper_axes[0][0] == 2
    assert proper_axes[0][1] == pytest.approx([0.0, 1.0, 0.0])
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 0
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 2
    assert mirror_axes[0][0] == "v"
    assert mirror_axes[1][0] == "v"
    assert mirror_axes[0][1] == pytest.approx([1.0, 0.0, 0.0])
    assert mirror_axes[1][1] == pytest.approx([0.0, 0.0, 1.0])
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "C2v"
    assert coords.symmetry_number(point_group) == 2

    data = logfiles["symmetries"]["tetracarbonyldicloro-OsII"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([785.21973892, 809.59902436, 817.20306192])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 6
    assert len(groups[0]) == 1
    assert len(groups[1]) == 2
    assert len(groups[2]) == 2
    assert len(groups[3]) == 2
    assert len(groups[4]) == 2
    assert len(groups[5]) == 2
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric prolate", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 1
    assert proper_axes[0][0] == 2
    assert proper_axes[0][1] == pytest.approx(
        [0.001877205661635017, 0.011104681211935762, 0.9999365790659351]
    )
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 0
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 2
    assert mirror_axes[0][0] == "v"
    assert mirror_axes[1][0] == "v"
    assert mirror_axes[0][1] == pytest.approx([1.0, 0.0, 0.0])
    assert mirror_axes[1][1] == pytest.approx(
        [-0.009273225981885458, 0.9999054852612625, -0.010150262278787719]
    )
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "C2v"
    assert coords.symmetry_number(point_group) == 2


def test_can_understand_C3v_symmetry():
    """Ensure values match regression logfiles for C3v symmetry."""
    data = logfiles["symmetries"]["ammonia"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([1.70511527, 1.70683927, 2.6588982])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 2
    assert len(groups[0]) == 1
    assert len(groups[1]) == 3
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric oblate", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 1
    assert proper_axes[0][0] == 3
    assert proper_axes[0][1] == pytest.approx([0.0, 0.0, 1.0])
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 0
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 3
    assert mirror_axes[0][0] == "v"
    assert mirror_axes[1][0] == "v"
    assert mirror_axes[2][0] == "v"
    assert mirror_axes[0][1] == pytest.approx(
        [0.9068776087435643, 0.4213940450284593, -0.0002481413053416406]
    )
    assert mirror_axes[1][1] == pytest.approx(
        [0.8186942945658886, -0.5742296156207778, -2.427500595304519e-5]
    )
    assert mirror_axes[2][1] == pytest.approx(
        [0.08831067425523016, 0.9960929548201968, -0.00022398695457168588]
    )
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "C3v"
    assert coords.symmetry_number(point_group) == 3

    data = logfiles["symmetries"]["trichloromethane"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([161.52970548, 161.53643691, 311.48496042])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(
        np.array(
            [[1.0, 2.39535015e-12, 0.0], [-2.39535015e-12, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        abs=1e-6,
    )
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 3
    assert len(groups[0]) == 1
    assert len(groups[1]) == 1
    assert len(groups[2]) == 3
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric oblate", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 1
    assert proper_axes[0][0] == 3
    assert proper_axes[0][1] == pytest.approx([0.0, 0.0, 1.0])
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 0
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 3
    assert mirror_axes[0][0] == "v"
    assert mirror_axes[1][0] == "v"
    assert mirror_axes[2][0] == "v"
    assert mirror_axes[0][1] == pytest.approx(
        [0.9788875135448617, 0.20439969624408313, -1.1334809837856751e-6]
    )
    assert mirror_axes[1][1] == pytest.approx(
        [0.31243041249605286, 0.9499406493815911, 1.6540176920563067e-7]
    )
    assert mirror_axes[2][1] == pytest.approx(
        [-0.6664703153892539, 0.745531567878424, 1.298894105965732e-6]
    )
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "C3v"
    assert coords.symmetry_number(point_group) == 3

    data = logfiles["symmetries"]["phosphorous-oxychloride"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([269.55650843, 269.5984722, 366.72364626])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    # assert axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 3
    assert len(groups[0]) == 1
    assert len(groups[1]) == 1
    assert len(groups[2]) == 3
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric oblate", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 1
    assert proper_axes[0][0] == 3
    # assert proper_axes[0][1] == pytest.approx([0.0, 0.0, 1.0])
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 0
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 3
    assert mirror_axes[0][0] == "v"
    assert mirror_axes[1][0] == "v"
    assert mirror_axes[2][0] == "v"
    assert mirror_axes[0][1] == pytest.approx(
        [0.9968788620819952, 0.07894638512700726, -5.108311413560213e-5]
    )
    assert mirror_axes[1][1] == pytest.approx(
        [0.5668604598156988, -0.8238138215701102, 8.054554709513842e-5]
    )
    assert mirror_axes[2][1] == pytest.approx(
        [-0.4301018285979955, -0.902780371800905, 0.00013163464672069048]
    )
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "C3v"
    assert coords.symmetry_number(point_group) == 3

    data = logfiles["symmetries"]["benzenetricarbonylchromium"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([532.79651409, 683.7802014, 684.19947951])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 5
    assert len(groups[0]) == 1
    assert len(groups[1]) == 3
    assert len(groups[2]) == 3
    # TODO(schneiderfelipe): I believe the following should be split into two
    # groups of three each
    assert len(groups[3]) == 6
    # TODO(schneiderfelipe): I believe the following should be split into two
    # groups of three each
    assert len(groups[4]) == 6
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric prolate", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 1
    assert proper_axes[0][0] == 3
    assert proper_axes[0][1] == pytest.approx([1.0, 0.0, 0.0])
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 0
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 3
    assert mirror_axes[0][0] == "v"
    assert mirror_axes[1][0] == "v"
    assert mirror_axes[2][0] == "v"
    assert mirror_axes[0][1] == pytest.approx(
        [0.0007833624576096415, -0.6235845722860999, -0.7817555036902663]
    )
    assert mirror_axes[1][1] == pytest.approx(
        [0.00042780112102959694, 0.3652328771996655, -0.930916087732216]
    )
    assert mirror_axes[2][1] == pytest.approx(
        [0.00035562884006787933, -0.9888277564519781, 0.14906220714277532]
    )
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "C3v"
    assert coords.symmetry_number(point_group) == 3


def test_can_understand_C4v_symmetry():
    """Ensure values match regression logfiles for C4v symmetry."""
    data = logfiles["symmetries"]["OF4Xe"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([198.56886522, 198.66454795, 298.68512748])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 3
    assert len(groups[0]) == 1
    assert len(groups[1]) == 1
    assert len(groups[2]) == 4
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric oblate", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 1
    assert proper_axes[0][0] == 4
    assert proper_axes[0][1] == pytest.approx([0.0, 0.0, 1.0])
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 0
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 4
    assert mirror_axes[0][0] == "v"
    assert mirror_axes[1][0] == "v"
    assert mirror_axes[2][0] == "v"
    assert mirror_axes[3][0] == "v"
    assert mirror_axes[0][1] == pytest.approx(
        [0.9999835391906019, 0.005719503675732009, -0.00045675545024613507]
    )
    assert mirror_axes[1][1] == pytest.approx(
        [-0.00505092778575004, 0.9999871320942353, -0.0004730480416265371]
    )
    assert mirror_axes[2][1] == pytest.approx(
        [-0.7031857811796408, -0.7110059899310466, 0.0004893147767132002]
    )
    assert mirror_axes[3][1] == pytest.approx(
        [-0.7112312161823279, 0.7029581474041968, -1.1219010926151304e-5]
    )
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "C4v"
    assert coords.symmetry_number(point_group) == 4


def test_can_understand_C5v_symmetry():
    """Ensure values match regression logfiles for C5v symmetry."""
    data = logfiles["symmetries"]["corannulene"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([1010.15093506, 1010.21973269, 1945.45456697])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 4
    assert len(groups[0]) == 5
    assert len(groups[1]) == 5
    assert len(groups[2]) == 10
    assert len(groups[3]) == 10
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric oblate", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 1
    assert proper_axes[0][0] == 5
    assert proper_axes[0][1] == pytest.approx([0.0, 0.0, 1.0])
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 0
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 5
    assert mirror_axes[0][0] == "v"
    assert mirror_axes[1][0] == "v"
    assert mirror_axes[2][0] == "v"
    assert mirror_axes[3][0] == "v"
    assert mirror_axes[4][0] == "v"
    assert mirror_axes[0][1] == pytest.approx(
        [0.967469868826218, -0.2529862670397663, -4.0033292367072414e-5]
    )
    assert mirror_axes[1][1] == pytest.approx(
        [0.9314050489666861, 0.363984387941639, -9.701312325106918e-6]
    )
    assert mirror_axes[2][1] == pytest.approx(
        [0.6340056793426564, -0.7733283878974474, -5.5074423239961966e-5]
    )
    assert mirror_axes[3][1] == pytest.approx(
        [0.5395523596554447, 0.841952045842404, 6.076209222153875e-5]
    )
    assert mirror_axes[4][1] == pytest.approx(
        [-0.058396984482266025, 0.9982934340845996, 0.000108013679569529]
    )
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "C5v"
    assert coords.symmetry_number(point_group) == 5


def test_can_understand_D2_symmetry():
    """Ensure values match regression logfiles for D2 symmetry."""
    data = logfiles["symmetries"]["biphenyl"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([180.57613675, 942.28140722, 1083.10838697])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 7
    assert len(groups[0]) == 2
    assert len(groups[1]) == 2
    assert len(groups[2]) == 2
    assert len(groups[3]) == 4
    assert len(groups[4]) == 4
    assert len(groups[5]) == 4
    assert len(groups[6]) == 4
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("asymmetric", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 3
    assert proper_axes[0][0] == 2
    assert proper_axes[1][0] == 2
    assert proper_axes[2][0] == 2
    assert proper_axes[0][1] == pytest.approx([1.0, 0.0, 0.0])
    assert proper_axes[1][1] == pytest.approx([0.0, 1.0, 0.0])
    assert proper_axes[2][1] == pytest.approx([0.0, 0.0, 1.0])
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 0
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 0
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "D2"
    assert coords.symmetry_number(point_group) == 4


def test_can_understand_D3_symmetry():
    """Ensure values match regression logfiles for D3 symmetry."""
    data = logfiles["symmetries"]["tris-ethylenediamine-RuII"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([716.90346743, 716.9163632, 1112.40527375])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    # assert axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 7
    assert len(groups[0]) == 1
    assert len(groups[1]) == 6
    assert len(groups[2]) == 6
    assert len(groups[3]) == 6
    assert len(groups[4]) == 6
    assert len(groups[5]) == 6
    assert len(groups[6]) == 6
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric oblate", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 4
    assert proper_axes[0][0] == 3
    assert proper_axes[1][0] == 2
    assert proper_axes[2][0] == 2
    assert proper_axes[3][0] == 2
    # assert proper_axes[0][1] == pytest.approx([0.0, 0.0, 1.0])
    assert proper_axes[1][1] == pytest.approx(
        [-0.007801928433068849, 0.999969564493194, -1.3114109559063874e-7]
    )
    assert proper_axes[2][1] == pytest.approx(
        [-0.8628070046702555, -0.5055325145794547, 0.0009743689829073926]
    )
    assert proper_axes[3][1] == pytest.approx(
        [-0.8695351060281812, 0.49387058756810087, 0.0007362877823667492]
    )
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 0
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 0
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "D3"
    assert coords.symmetry_number(point_group) == 6

    data = logfiles["symmetries"]["tris-ethylenediamine-CoIII"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([661.90920375, 662.85444032, 1018.71597285])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 7
    assert len(groups[0]) == 1
    assert len(groups[1]) == 6
    assert len(groups[2]) == 6
    assert len(groups[3]) == 6
    assert len(groups[4]) == 6
    assert len(groups[5]) == 6
    assert len(groups[6]) == 6
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric oblate", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 4
    assert proper_axes[0][0] == 3
    assert proper_axes[1][0] == 2
    assert proper_axes[2][0] == 2
    assert proper_axes[3][0] == 2
    assert proper_axes[0][1] == pytest.approx([0.0, 0.0, 1.0])
    assert proper_axes[1][1] == pytest.approx(
        [-8.396686394741882e-5, 0.9999999964747823, -3.439237791100686e-8]
    )
    assert proper_axes[2][1] == pytest.approx(
        [-0.865289138168098, 0.5012697710705389, 0.001823178323558723]
    )
    assert proper_axes[3][1] == pytest.approx(
        [-0.8668052702358067, -0.49864677853044065, 0.00011727166713889237]
    )
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 0
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 0
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "D3"
    assert coords.symmetry_number(point_group) == 6


def test_can_understand_D5_symmetry():
    """Ensure values match regression logfiles for D5 symmetry."""
    data = logfiles["symmetries"]["ferrocene-twisted"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([233.36764327, 470.55800877, 470.55805987])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(
        np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, -2.49091225e-9], [0.0, 2.49091214e-9, 1.0]]
        ),
        abs=1e-6,
    )
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 3
    assert len(groups[0]) == 1
    assert len(groups[1]) == 10
    assert len(groups[2]) == 10
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric prolate", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 6
    assert proper_axes[0][0] == 5
    assert proper_axes[1][0] == 2
    assert proper_axes[2][0] == 2
    assert proper_axes[3][0] == 2
    assert proper_axes[4][0] == 2
    assert proper_axes[5][0] == 2
    assert proper_axes[0][1] == pytest.approx([1.0, 0.0, 0.0])
    assert proper_axes[1][1] == pytest.approx(
        [0.0, 0.309016595317643, 0.9510566459566391]
    )
    assert proper_axes[2][1] == pytest.approx(
        [0.0, 0.8090168824555769, 0.5877854063362405]
    )
    assert proper_axes[3][1] == pytest.approx(
        [0.0, 0.3090161353437463, -0.9510567954108816]
    )
    assert proper_axes[4][1] == pytest.approx([0.0, 1.0, 0.0])
    assert proper_axes[5][1] == pytest.approx(
        [0.0, 0.8090168500740541, -0.5877854509055626]
    )
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 0
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 0
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "D5"
    assert coords.symmetry_number(point_group) == 10


def test_can_understand_D2h_symmetry():
    """Ensure values match regression logfiles for D2h symmetry."""
    data = logfiles["symmetries"]["ethylene"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([3.56497952, 17.24901988, 20.8139994])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 2
    assert len(groups[0]) == 2
    assert len(groups[1]) == 4
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("asymmetric", "irregular planar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 3
    assert proper_axes[0][0] == 2
    assert proper_axes[1][0] == 2
    assert proper_axes[2][0] == 2
    assert proper_axes[0][1] == pytest.approx([1.0, 0.0, 0.0])
    # assert proper_axes[1][1] == pytest.approx([0.0, 1.0, 0.0])
    # assert proper_axes[2][1] == pytest.approx([0.0, 0.0, 1.0])
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 3
    assert improper_axes[0][0] == 2
    assert improper_axes[1][0] == 2
    assert improper_axes[2][0] == 2
    assert improper_axes[0][1] == proper_axes[0][1]
    assert improper_axes[1][1] == proper_axes[1][1]
    assert improper_axes[2][1] == proper_axes[2][1]
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 3
    assert mirror_axes[0][0] == "h"
    assert mirror_axes[1][0] == "h"
    assert mirror_axes[2][0] == "h"
    assert mirror_axes[0][1] == pytest.approx([1.0, 0.0, 0.0])
    # assert mirror_axes[1][1] == pytest.approx([0.0, 1.0, 0.0])
    # assert mirror_axes[2][1] == pytest.approx([0.0, 0.0, 1.0])
    assert coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "D2h"
    assert coords.symmetry_number(point_group) == 4

    data = logfiles["symmetries"]["diborane"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([6.26383693, 26.96388268, 29.40025824])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    # assert axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 3
    assert len(groups[0]) == 2
    assert len(groups[1]) == 2
    assert len(groups[2]) == 4
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("asymmetric", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 3
    assert proper_axes[0][0] == 2
    assert proper_axes[1][0] == 2
    assert proper_axes[2][0] == 2
    assert proper_axes[0][1] == pytest.approx([1.0, 0.0, 0.0])
    # assert proper_axes[1][1] == pytest.approx([0.0, 0.0, 1.0])
    # assert proper_axes[2][1] == pytest.approx([0.0, 1.0, 0.0])
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 3
    assert improper_axes[0][0] == 2
    assert improper_axes[1][0] == 2
    assert improper_axes[2][0] == 2
    assert improper_axes[0][1] == proper_axes[0][1]
    assert improper_axes[1][1] == proper_axes[1][1]
    assert improper_axes[2][1] == proper_axes[2][1]
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 3
    assert mirror_axes[0][0] == "h"
    assert mirror_axes[1][0] == "h"
    assert mirror_axes[2][0] == "h"
    assert mirror_axes[0][1] == pytest.approx([1.0, 0.0, 0.0])
    # assert mirror_axes[1][1] == pytest.approx([0.0, 0.0, 1.0])
    # assert mirror_axes[2][1] == pytest.approx([0.0, 1.0, 0.0])
    assert coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "D2h"
    assert coords.symmetry_number(point_group) == 4

    data = logfiles["symmetries"]["1,4-dichlorobenzene"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([88.31658644, 769.01913704, 857.3357217])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 3
    assert len(groups[0]) == 2
    assert len(groups[1]) == 4
    # TODO(schneiderfelipe): I believe this group below should optimally split
    # into 2 groups of 2 and 4 atoms each
    assert len(groups[2]) == 6
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("asymmetric", "irregular planar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 3
    assert proper_axes[0][0] == 2
    assert proper_axes[1][0] == 2
    assert proper_axes[2][0] == 2
    assert proper_axes[0][1] == pytest.approx([1.0, 0.0, 0.0])
    assert proper_axes[1][1] == pytest.approx([0.0, 1.0, 0.0])
    assert proper_axes[2][1] == pytest.approx([0.0, 0.0, 1.0])
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 3
    assert improper_axes[0][0] == 2
    assert improper_axes[1][0] == 2
    assert improper_axes[2][0] == 2
    assert improper_axes[0][1] == proper_axes[0][1]
    assert improper_axes[1][1] == proper_axes[1][1]
    assert improper_axes[2][1] == proper_axes[2][1]
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 3
    assert mirror_axes[0][0] == "h"
    assert mirror_axes[1][0] == "h"
    assert mirror_axes[2][0] == "h"
    assert mirror_axes[0][1] == pytest.approx([1.0, 0.0, 0.0])
    assert mirror_axes[1][1] == pytest.approx([0.0, 1.0, 0.0])
    assert mirror_axes[2][1] == pytest.approx([0.0, 0.0, 1.0])
    assert coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "D2h"
    assert coords.symmetry_number(point_group) == 4

    data = logfiles["symmetries"]["Mn2F6"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([182.07906581, 784.89021644, 966.96922024])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 3
    assert len(groups[0]) == 2
    assert len(groups[1]) == 2
    assert len(groups[2]) == 4
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("asymmetric", "irregular planar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 3
    assert proper_axes[0][0] == 2
    assert proper_axes[1][0] == 2
    assert proper_axes[2][0] == 2
    assert proper_axes[0][1] == pytest.approx([1.0, 0.0, 0.0])
    assert proper_axes[1][1] == pytest.approx([0.0, 1.0, 0.0])
    assert proper_axes[2][1] == pytest.approx([0.0, 0.0, 1.0])
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 3
    assert improper_axes[0][0] == 2
    assert improper_axes[1][0] == 2
    assert improper_axes[2][0] == 2
    assert improper_axes[0][1] == proper_axes[0][1]
    assert improper_axes[1][1] == proper_axes[1][1]
    assert improper_axes[2][1] == proper_axes[2][1]
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 3
    assert mirror_axes[0][0] == "h"
    assert mirror_axes[1][0] == "h"
    assert mirror_axes[2][0] == "h"
    assert mirror_axes[0][1] == pytest.approx([1.0, 0.0, 0.0])
    assert mirror_axes[1][1] == pytest.approx([0.0, 1.0, 0.0])
    assert mirror_axes[2][1] == pytest.approx([0.0, 0.0, 1.0])
    assert coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "D2h"
    assert coords.symmetry_number(point_group) == 4


def test_can_understand_D3h_symmetry():
    """Ensure values match regression logfiles for D3h symmetry."""
    data = logfiles["ethane"]["eclipsed@B97-3c"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([6.16370157, 25.5194263, 25.51959008])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(
        np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, -5.44461143e-11], [0.0, 5.44456702e-11, 1.0]]
        ),
        abs=1e-6,
    )
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 2
    assert len(groups[0]) == 2
    assert len(groups[1]) == 6
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric prolate", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 4
    assert proper_axes[0][0] == 3
    assert proper_axes[1][0] == 2
    assert proper_axes[2][0] == 2
    assert proper_axes[3][0] == 2
    assert proper_axes[0][1] == pytest.approx([1.0, 0.0, 0.0])
    assert proper_axes[1][1] == pytest.approx(
        [7.684797155608644e-5, -0.3613920169943295, 0.932413966083284]
    )
    assert proper_axes[2][1] == pytest.approx(
        [-3.735917817880401e-5, 0.6268578875506137, 0.7791336133294943]
    )
    assert proper_axes[3][1] == pytest.approx(
        [-5.13760417702473e-5, 0.988178974905995, -0.15330463435343078]
    )
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 1
    assert improper_axes[0][0] == 3
    assert improper_axes[0][1] == proper_axes[0][1]
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 4
    assert mirror_axes[0][0] == "h"
    assert mirror_axes[1][0] == "v"
    assert mirror_axes[2][0] == "v"
    assert mirror_axes[3][0] == "v"
    assert mirror_axes[0][1] == pytest.approx([1.0, 0.0, 0.0])
    assert mirror_axes[1][1] == pytest.approx(
        [1.3400325115110517e-5, -0.1518387969594028, -0.9884052709078156]
    )
    assert mirror_axes[2][1] == pytest.approx(
        [-7.399041283459149e-5, 0.7800955898768939, -0.6256603433014112]
    )
    assert mirror_axes[3][1] == pytest.approx(
        [-8.738720232596221e-5, 0.9318966227508982, 0.36272396787219213]
    )
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "D3h"
    assert coords.symmetry_number(point_group) == 6

    data = logfiles["symmetries"]["BF3"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([50.77255975, 50.7862414, 101.55880103])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 2
    assert len(groups[0]) == 1
    assert len(groups[1]) == 3
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric oblate", "regular planar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 4
    assert proper_axes[0][0] == 3
    assert proper_axes[1][0] == 2
    assert proper_axes[2][0] == 2
    assert proper_axes[3][0] == 2
    assert proper_axes[0][1] == pytest.approx([0.0, 0.0, 1.0])
    assert proper_axes[1][1] == pytest.approx(
        [0.786485610329844, -0.6176086014998113, 9.876227954894027e-6]
    )
    assert proper_axes[2][1] == pytest.approx(
        [-0.14171352623954847, -0.9899077108955692, 1.975466712182074e-5]
    )
    assert proper_axes[3][1] == pytest.approx(
        [-0.9281566611554543, -0.37218975302286383, 9.876181775103538e-6]
    )
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 1
    assert improper_axes[0][0] == 3
    assert improper_axes[0][1] == proper_axes[0][1]
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 4
    assert mirror_axes[0][0] == "h"
    assert mirror_axes[1][0] == "v"
    assert mirror_axes[2][0] == "v"
    assert mirror_axes[3][0] == "v"
    assert mirror_axes[0][1] == pytest.approx([0.0, 0.0, 1.0])
    assert mirror_axes[1][1] == pytest.approx(
        [0.3722848618205517, -0.9281185170328478, 1.1628857803315213e-9]
    )
    assert mirror_axes[2][1] == pytest.approx(
        [-0.6177198528302001, -0.7863982346238041, 1.3109907332481445e-9]
    )
    assert mirror_axes[3][1] == pytest.approx(
        [-0.9899127272168826, 0.14167848281949386, 1.4813107352227758e-10]
    )
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "D3h"
    assert coords.symmetry_number(point_group) == 6

    data = logfiles["symmetries"]["PCl5"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([452.37354879, 558.48639882, 558.6442966])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    # assert axes == pytest.approx(
    #     np.array(
    #         [[1.0, 0.0, 0.0], [0.0, 1.0, -1.11821663e-12], [0.0, 1.11810561e-12, 1.0]]
    #     )
    # )
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 3
    assert len(groups[0]) == 1
    assert len(groups[1]) == 2
    assert len(groups[2]) == 3
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric prolate", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 4
    assert proper_axes[0][0] == 3
    assert proper_axes[1][0] == 2
    assert proper_axes[2][0] == 2
    assert proper_axes[3][0] == 2
    assert proper_axes[0][1] == pytest.approx([1.0, 0.0, 0.0])
    assert proper_axes[1][1] == pytest.approx(
        [0.0001997060253834037, -0.33600385261575644, 0.9418605900794833]
    )
    assert proper_axes[2][1] == pytest.approx(
        [-0.00012559744182296924, -0.9836343455164556, 0.18017618751016082]
    )
    assert proper_axes[3][1] == pytest.approx(
        [-0.00032535420422918877, -0.6480520885057931, -0.7615959458452503]
    )
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 1
    assert improper_axes[0][0] == 3
    assert improper_axes[0][1] == proper_axes[0][1]
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 4
    assert mirror_axes[0][0] == "h"
    assert mirror_axes[1][0] == "v"
    assert mirror_axes[2][0] == "v"
    assert mirror_axes[3][0] == "v"
    assert mirror_axes[0][1] == pytest.approx([1.0, 0.0, 0.0])
    assert mirror_axes[1][1] == pytest.approx(
        [0.0001544181847713839, -0.76190637662992, 0.647687153960684]
    )
    assert mirror_axes[2][1] == pytest.approx(
        [-0.0001486955790155652, -0.941961644198146, -0.33572047710729574]
    )
    assert mirror_axes[3][1] == pytest.approx(
        [-0.00030318766112059263, -0.18019063014134656, -0.9836316611854798]
    )
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "D3h"
    assert coords.symmetry_number(point_group) == 6


def test_can_understand_D4h_symmetry():
    """Ensure values match regression logfiles for D4h symmetry."""
    data = logfiles["symmetries"]["XeF4"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([153.07544899, 153.15479772, 306.23024671])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 2
    assert len(groups[0]) == 1
    assert len(groups[1]) == 4
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric oblate", "regular planar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 5
    assert proper_axes[0][0] == 4
    assert proper_axes[1][0] == 2
    assert proper_axes[2][0] == 2
    assert proper_axes[3][0] == 2
    assert proper_axes[4][0] == 2
    assert proper_axes[0][1] == pytest.approx([0.0, 0.0, 1.0])
    assert proper_axes[1][1] == pytest.approx(
        [-0.1038959947038264, -0.9945881671749733, -1.722176640137799e-7]
    )
    assert proper_axes[2][1] == pytest.approx(
        [-0.6297499896948396, 0.7767978826434407, -1.2938984477698114e-7]
    )
    assert proper_axes[3][1] == pytest.approx(
        [-0.7768900176339207, -0.6296363240004132, -1.1409827074055784e-7]
    )
    assert proper_axes[4][1] == pytest.approx(
        [-0.9945789090852918, 0.10398458348563748, -1.721587770986498e-7]
    )
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 5
    assert improper_axes[0][0] == 4
    assert improper_axes[1][0] == 2
    assert improper_axes[2][0] == 2
    assert improper_axes[3][0] == 2
    assert improper_axes[4][0] == 2
    assert improper_axes[0][1] == proper_axes[0][1]
    assert improper_axes[1][1] == proper_axes[1][1]
    assert improper_axes[2][1] == proper_axes[2][1]
    assert improper_axes[3][1] == proper_axes[3][1]
    assert improper_axes[4][1] == proper_axes[4][1]
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 5
    assert mirror_axes[0][0] == "h"
    assert mirror_axes[1][0] == "v"
    assert mirror_axes[2][0] == "v"
    assert mirror_axes[3][0] == "v"
    assert mirror_axes[4][0] == "v"
    assert mirror_axes[0][1] == pytest.approx([0.0, 0.0, 1.0])
    assert mirror_axes[1][1] == pytest.approx(
        [0.9945939007358418, -0.10384109311376788, -1.0806855373154067e-8]
    )
    assert mirror_axes[2][1] == pytest.approx(
        [0.10411808372214175, 0.994564942395432, -1.0802010730760494e-8]
    )
    assert mirror_axes[3][1] == pytest.approx(
        [-0.6298927524585732, 0.7766821231367197, 6.24992618671696e-12]
    )
    assert mirror_axes[4][1] == pytest.approx(
        [-0.776841781663576, -0.6296958363065147, -2.913051691777635e-11]
    )
    assert coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "D4h"
    assert coords.symmetry_number(point_group) == 8

    data = logfiles["symmetries"]["tetracarbonylnickel"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([385.29998789, 385.47071543, 770.77067351])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(np.eye(3), abs=1e-6)
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 3
    assert len(groups[0]) == 1
    assert len(groups[1]) == 4
    assert len(groups[2]) == 4
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric oblate", "regular planar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 5
    assert proper_axes[0][0] == 4
    assert proper_axes[1][0] == 2
    assert proper_axes[2][0] == 2
    assert proper_axes[3][0] == 2
    assert proper_axes[4][0] == 2
    assert proper_axes[0][1] == pytest.approx([0.0, 0.0, 1.0])
    assert proper_axes[1][1] == pytest.approx(
        [0.999999985120203, -0.00017250910677172902, -4.4908920788199634e-7]
    )
    assert proper_axes[2][1] == pytest.approx(
        [0.7073372010244388, 0.706876277158874, 0.00011330014792838772]
    )
    assert proper_axes[3][1] == pytest.approx(
        [0.7069499144603176, -0.7072636039398956, -0.00011395832590658424]
    )
    assert proper_axes[4][1] == pytest.approx(
        [-0.000153000029355287, -0.9999999882933196, -2.0860089700576687e-6]
    )
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 5
    assert improper_axes[0][0] == 4
    assert improper_axes[1][0] == 2
    assert improper_axes[2][0] == 2
    assert improper_axes[3][0] == 2
    assert improper_axes[4][0] == 2
    assert improper_axes[0][1] == proper_axes[0][1]
    assert improper_axes[1][1] == proper_axes[1][1]
    assert improper_axes[2][1] == proper_axes[2][1]
    assert improper_axes[3][1] == proper_axes[3][1]
    assert improper_axes[4][1] == proper_axes[4][1]
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 5
    assert mirror_axes[0][0] == "h"
    assert mirror_axes[1][0] == "v"
    assert mirror_axes[2][0] == "v"
    assert mirror_axes[3][0] == "v"
    assert mirror_axes[4][0] == "v"
    assert mirror_axes[0][1] == pytest.approx([0.0, 0.0, 1.0])
    assert mirror_axes[1][1] == pytest.approx(
        [0.9999999101379913, -0.0003929572891462238, -0.00015908669994704126]
    )
    assert mirror_axes[2][1] == pytest.approx(
        [0.7073365863129996, 0.7068769013496857, 1.1571974206612855e-6]
    )
    assert mirror_axes[3][1] == pytest.approx(
        [0.7069496160828133, -0.707263911365157, -9.068453524048463e-7]
    )
    assert mirror_axes[4][1] == pytest.approx(
        [0.0003752463460537577, 0.9999999166821465, 0.00016070432460672592]
    )
    assert coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "D4h"
    assert coords.symmetry_number(point_group) == 8


def test_can_understand_D5h_symmetry():
    """Ensure values match regression logfiles for D5h symmetry."""
    data = logfiles["symmetries"]["cyclopentadienyl-"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([57.92720765, 57.92813018, 115.85533765])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(
        np.array(
            [[1.0, -1.44414533e-12, 0.0], [1.44414533e-12, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        abs=1e-6,
    )
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 2
    assert len(groups[0]) == 5
    assert len(groups[1]) == 5
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric oblate", "regular planar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 6
    assert proper_axes[0][0] == 5
    assert proper_axes[1][0] == 2
    assert proper_axes[2][0] == 2
    assert proper_axes[3][0] == 2
    assert proper_axes[4][0] == 2
    assert proper_axes[5][0] == 2
    assert proper_axes[0][1] == pytest.approx([0.0, 0.0, 1.0])
    assert proper_axes[1][1] == pytest.approx(
        [0.9852635321302113, -0.1710431876018575, -1.5142524986039497e-5]
    )
    assert proper_axes[2][1] == pytest.approx(
        [0.8976888382803988, 0.4406299460094585, 1.7506847806145937e-5]
    )
    assert proper_axes[3][1] == pytest.approx(
        [0.46736242714771226, 0.8840658119171837, 4.347194495240259e-5]
    )
    assert proper_axes[4][1] == pytest.approx(
        [-0.14169717322488817, 0.989910050980385, 4.547623249634363e-5]
    )
    assert proper_axes[5][1] == pytest.approx(
        [-0.6966055119393046, 0.717454360799123, 3.0101515311866436e-5]
    )
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 1
    assert improper_axes[0][0] == 5
    assert improper_axes[0][1] == proper_axes[0][1]
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 6
    assert mirror_axes[0][0] == "h"
    assert mirror_axes[1][0] == "v"
    assert mirror_axes[2][0] == "v"
    assert mirror_axes[3][0] == "v"
    assert mirror_axes[4][0] == "v"
    assert mirror_axes[5][0] == "v"
    assert mirror_axes[0][1] == pytest.approx([0.0, 0.0, 1.0])
    assert mirror_axes[1][1] == pytest.approx(
        [0.7175304768309594, 0.6965271097667781, -1.3366924475017621e-5]
    )
    assert mirror_axes[2][1] == pytest.approx(
        [0.1712137266526045, 0.9852339109556842, 2.2554246915974823e-5]
    )
    assert mirror_axes[3][1] == pytest.approx(
        [-0.44063308377390153, 0.8976872968896024, 4.986923808926847e-5]
    )
    assert mirror_axes[4][1] == pytest.approx(
        [-0.8842015694852551, 0.4671055383465718, 2.3785937837845682e-5]
    )
    assert mirror_axes[5][1] == pytest.approx(
        [-0.9899102667835931, -0.14169567243616252, -1.1370992919649144e-5]
    )
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "D5h"
    assert coords.symmetry_number(point_group) == 10

    data = logfiles["symmetries"]["ferrocene-eclipsed"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([228.8186135, 480.633784, 480.63796274])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    # assert axes == pytest.approx(
    #     np.array(
    #         [[1.0, 0.0, 0.0], [0.0, 1.0, 1.73114301e-12], [0.0, -1.73112913e-12, 1.0]]
    #     )
    # )
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 3
    assert len(groups[0]) == 1
    assert len(groups[1]) == 10
    assert len(groups[2]) == 10
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric prolate", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 6
    assert proper_axes[0][0] == 5
    assert proper_axes[1][0] == 2
    assert proper_axes[2][0] == 2
    assert proper_axes[3][0] == 2
    assert proper_axes[4][0] == 2
    assert proper_axes[5][0] == 2
    assert proper_axes[0][1] == pytest.approx([1.0, 0.0, 0.0])
    assert proper_axes[1][1] == pytest.approx([0.0, 0.0, 1.0])
    assert proper_axes[2][1] == pytest.approx(
        [0.0, -0.5877684885201441, 0.8090291737031151]
    )
    assert proper_axes[3][1] == pytest.approx(
        [-8.839583850607428e-5, -0.9510798203224098, -0.3089452501037498]
    )
    assert proper_axes[4][1] == pytest.approx(
        [-0.00010263758129107958, -0.9510465807149229, 0.3090475542307049]
    )
    assert proper_axes[5][1] == pytest.approx(
        [-0.00026868634145792505, 0.5877676345338332, 0.8090297495161426]
    )
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 1
    assert improper_axes[0][0] == 5
    assert improper_axes[0][1] == proper_axes[0][1]
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 6
    assert mirror_axes[0][0] == "h"
    assert mirror_axes[1][0] == "v"
    assert mirror_axes[2][0] == "v"
    assert mirror_axes[3][0] == "v"
    assert mirror_axes[4][0] == "v"
    assert mirror_axes[5][0] == "v"
    assert mirror_axes[0][1] == pytest.approx([1.0, 0.0, 0.0])
    assert mirror_axes[1][1] == pytest.approx([0.0, -1.0, 0.0])
    assert mirror_axes[2][1] == pytest.approx(
        [-2.872053664793665e-5, -0.30901367413134206, -0.9510575946676308]
    )
    assert mirror_axes[3][1] == pytest.approx(
        [-9.479875249986214e-5, 0.8090196172706197, 0.5877816345246735]
    )
    assert mirror_axes[4][1] == pytest.approx(
        [-9.479875250002724e-5, -0.8090196172706197, 0.5877816345246735]
    )
    assert mirror_axes[5][1] == pytest.approx(
        [-0.00014127095221938595, 0.30903875859031504, -0.9510494339052388]
    )
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "D5h"
    assert coords.symmetry_number(point_group) == 10


def test_can_understand_D6h_symmetry():
    """Ensure values match regression logfiles for D6h symmetry."""
    data = logfiles["symmetries"]["benzene"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([90.78768809, 90.79030869, 181.57799671])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    # assert axes == pytest.approx(
    #     np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    # )
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 2
    assert len(groups[0]) == 6
    assert len(groups[1]) == 6
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric oblate", "regular planar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 7
    assert proper_axes[0][0] == 6
    assert proper_axes[1][0] == 2
    assert proper_axes[2][0] == 2
    assert proper_axes[3][0] == 2
    assert proper_axes[4][0] == 2
    assert proper_axes[5][0] == 2
    assert proper_axes[6][0] == 2
    # assert proper_axes[0][1] == pytest.approx([0.0, 0.0, -1.0])
    assert proper_axes[1][1] == pytest.approx(
        [0.9815106205020463, -0.19140768362000038, 2.2201659599405544e-5]
    )
    assert proper_axes[2][1] == pytest.approx(
        [0.9457315005746486, 0.3249491170386625, -1.2514676957750245e-5]
    )
    assert proper_axes[3][1] == pytest.approx(
        [0.754296161937252, -0.6565343096012828, 2.0077290154722725e-5]
    )
    assert proper_axes[4][1] == pytest.approx(
        [0.6565889253439499, 0.7542486202774172, -4.3878122453110744e-5]
    )
    assert proper_axes[5][1] == pytest.approx(
        [0.32495152328162413, -0.9457306737961274, 1.257278573109447e-5]
    )
    assert proper_axes[6][1] == pytest.approx(
        [-0.19146486729407206, -0.9814994674434222, 5.848736671162612e-7]
    )
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 7
    assert improper_axes[0][0] == 6
    assert improper_axes[1][0] == 2
    assert improper_axes[2][0] == 2
    assert improper_axes[3][0] == 2
    assert improper_axes[4][0] == 2
    assert improper_axes[5][0] == 2
    assert improper_axes[6][0] == 2
    assert improper_axes[0][1] == proper_axes[0][1]
    assert improper_axes[1][1] == proper_axes[1][1]
    assert improper_axes[2][1] == proper_axes[2][1]
    assert improper_axes[3][1] == proper_axes[3][1]
    assert improper_axes[4][1] == proper_axes[4][1]
    assert improper_axes[5][1] == proper_axes[5][1]
    assert improper_axes[6][1] == proper_axes[6][1]
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 7
    assert mirror_axes[0][0] == "h"
    assert mirror_axes[1][0] == "v"
    assert mirror_axes[2][0] == "v"
    assert mirror_axes[3][0] == "v"
    assert mirror_axes[4][0] == "v"
    assert mirror_axes[5][0] == "v"
    assert mirror_axes[6][0] == "v"
    # assert mirror_axes[0][1] == pytest.approx([0.0, 0.0, -1.0])
    assert mirror_axes[1][1] == pytest.approx(
        [0.9815122368811188, -0.19139939609355305, -5.257917527921303e-6]
    )
    assert mirror_axes[2][1] == pytest.approx(
        [0.9457298111799615, 0.3249540310413546, -4.422090716741355e-5]
    )
    assert mirror_axes[3][1] == pytest.approx(
        [0.7543038379225381, -0.6565254907161294, -1.1630769178832298e-5]
    )
    assert mirror_axes[4][1] == pytest.approx(
        [0.656537579036802, 0.7542933164358446, 9.628587595848346e-6]
    )
    assert mirror_axes[5][1] == pytest.approx(
        [0.19147625644815636, 0.9814972451078651, -3.25925006682842e-5]
    )
    assert mirror_axes[6][1] == pytest.approx(
        [-0.32494344338723713, 0.9457334477707772, -6.6085126787184e-5]
    )
    assert coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "D6h"
    assert coords.symmetry_number(point_group) == 12


def test_can_understand_D7h_symmetry():
    """Ensure values match regression logfiles for D7h symmetry."""
    data = logfiles["symmetries"]["C7H7+"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([137.6924278, 137.70343972, 275.39586735])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 2
    assert len(groups[0]) == 7
    assert len(groups[1]) == 7
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric oblate", "regular planar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 8
    assert proper_axes[0][0] == 7
    assert proper_axes[1][0] == 2
    assert proper_axes[2][0] == 2
    assert proper_axes[3][0] == 2
    assert proper_axes[4][0] == 2
    assert proper_axes[5][0] == 2
    assert proper_axes[6][0] == 2
    assert proper_axes[7][0] == 2
    assert proper_axes[0][1] == pytest.approx([0.0, 0.0, 1.0])
    assert proper_axes[1][1] == pytest.approx(
        [0.9899264948348552, -0.14158225346562558, -1.8098975614729268e-5]
    )
    assert proper_axes[2][1] == pytest.approx(
        [0.9533272376098568, 0.30193903013335843, -1.0643615087070967e-5]
    )
    assert proper_axes[3][1] == pytest.approx(
        [0.8304605271230312, -0.5570774746834517, 9.53129133772328e-6]
    )
    assert proper_axes[4][1] == pytest.approx(
        [0.7279229355768501, 0.6856589530225752, -1.0803454883699417e-6]
    )
    assert proper_axes[5][1] == pytest.approx(
        [0.5065029636654298, -0.86223821914473, 3.5274085819386375e-5]
    )
    assert proper_axes[6][1] == pytest.approx(
        [0.35834036626998306, 0.9335910138147736, 2.873669294814405e-5]
    )
    assert proper_axes[7][1] == pytest.approx(
        [-0.08224227562969463, 0.9966123646156138, 5.286325905118405e-5]
    )
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 1
    assert improper_axes[0][0] == 7
    assert improper_axes[0][1] == proper_axes[0][1]
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 8
    assert mirror_axes[0][0] == "h"
    assert mirror_axes[1][0] == "v"
    assert mirror_axes[2][0] == "v"
    assert mirror_axes[3][0] == "v"
    assert mirror_axes[4][0] == "v"
    assert mirror_axes[5][0] == "v"
    assert mirror_axes[6][0] == "v"
    assert mirror_axes[7][0] == "v"
    assert mirror_axes[0][1] == pytest.approx([0.0, 0.0, 1.0])
    assert mirror_axes[1][1] == pytest.approx(
        [0.9966138210185994, 0.08222463171996094, 4.1148799697364515e-5]
    )
    assert mirror_axes[2][1] == pytest.approx(
        [0.8622384070955342, 0.506502644812164, -1.1297739746409977e-5]
    )
    assert mirror_axes[3][1] == pytest.approx(
        [0.5570924929465411, 0.8304504503699968, -6.150510183707668e-5]
    )
    assert mirror_axes[4][1] == pytest.approx(
        [0.14161223422329736, 0.9899222063262088, -2.3249449799447845e-5]
    )
    assert mirror_axes[5][1] == pytest.approx(
        [-0.30193157118177943, 0.9533295998441732, 1.961250861937452e-5]
    )
    assert mirror_axes[6][1] == pytest.approx(
        [-0.6856932447695671, 0.7278906319069466, 4.538158759723139e-5]
    )
    assert mirror_axes[7][1] == pytest.approx(
        [-0.9336164234837634, 0.3582741547160115, 6.216038584942583e-5]
    )
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "D7h"
    assert coords.symmetry_number(point_group) == 14


def test_can_understand_D8h_symmetry():
    """Ensure values match regression logfiles for D8h symmetry."""
    data = logfiles["symmetries"]["C8H8-2"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([203.63009236, 203.63614994, 407.26624219])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(
        np.array(
            [[1.0, -2.34596450e-12, 0.0], [2.34596450e-12, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        abs=1e-6,
    )
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 2
    assert len(groups[0]) == 8
    assert len(groups[1]) == 8
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric oblate", "regular planar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 9
    assert proper_axes[0][0] == 8
    assert proper_axes[1][0] == 2
    assert proper_axes[2][0] == 2
    assert proper_axes[3][0] == 2
    assert proper_axes[4][0] == 2
    assert proper_axes[5][0] == 2
    assert proper_axes[6][0] == 2
    assert proper_axes[7][0] == 2
    assert proper_axes[8][0] == 2
    assert proper_axes[0][1] == pytest.approx([0.0, 0.0, 1.0])
    assert proper_axes[1][1] == pytest.approx(
        [0.9898081173368313, -0.1424074815286812, 7.666555040225157e-6]
    )
    assert proper_axes[2][1] == pytest.approx(
        [0.9689611482367443, 0.2472130522483646, 2.4036192032153514e-6]
    )
    assert proper_axes[3][1] == pytest.approx(
        [0.8599671316732153, -0.5103494217723674, 1.1762357145401266e-5]
    )
    assert proper_axes[4][1] == pytest.approx(
        [0.800588646595334, 0.599214334709109, 4.665960705456225e-6]
    )
    assert proper_axes[5][1] == pytest.approx(
        [0.5103202528765483, 0.859984441408886, 6.217899608067287e-6]
    )
    assert proper_axes[6][1] == pytest.approx(
        [0.14239131532233637, 0.9898104430305723, 1.3725776141335332e-5]
    )
    assert proper_axes[7][1] == pytest.approx(
        [-0.2472158212887212, 0.968960441575424, 1.9144252268139394e-5]
    )
    assert proper_axes[8][1] == pytest.approx(
        [-0.5992048424224103, 0.8005957511766552, 3.927016327532347e-6]
    )
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 9
    assert improper_axes[0][0] == 8
    assert improper_axes[1][0] == 2
    assert improper_axes[2][0] == 2
    assert improper_axes[3][0] == 2
    assert improper_axes[4][0] == 2
    assert improper_axes[5][0] == 2
    assert improper_axes[6][0] == 2
    assert improper_axes[7][0] == 2
    assert improper_axes[8][0] == 2
    assert improper_axes[0][1] == proper_axes[0][1]
    assert improper_axes[1][1] == proper_axes[1][1]
    assert improper_axes[2][1] == proper_axes[2][1]
    assert improper_axes[3][1] == proper_axes[3][1]
    assert improper_axes[4][1] == proper_axes[4][1]
    assert improper_axes[5][1] == proper_axes[5][1]
    assert improper_axes[6][1] == proper_axes[6][1]
    assert improper_axes[7][1] == proper_axes[7][1]
    assert improper_axes[8][1] == proper_axes[8][1]
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 9
    assert mirror_axes[0][0] == "h"
    assert mirror_axes[1][0] == "v"
    assert mirror_axes[2][0] == "v"
    assert mirror_axes[3][0] == "v"
    assert mirror_axes[4][0] == "v"
    assert mirror_axes[5][0] == "v"
    assert mirror_axes[6][0] == "v"
    assert mirror_axes[7][0] == "v"
    assert mirror_axes[8][0] == "v"
    assert mirror_axes[0][1] == pytest.approx([0.0, 0.0, 1.0])
    assert mirror_axes[1][1] == pytest.approx(
        [0.9898116815335909, -0.14238270546096124, -1.688970919637159e-5]
    )
    assert mirror_axes[2][1] == pytest.approx(
        [0.968956498735555, 0.24723127511344176, 1.280297031747992e-5]
    )
    assert mirror_axes[3][1] == pytest.approx(
        [0.8006045479259707, 0.5991930875738841, 4.054538210451526e-5]
    )
    assert mirror_axes[4][1] == pytest.approx(
        [0.5103574774941007, 0.859962350879539, 1.5349110859577596e-5]
    )
    assert mirror_axes[5][1] == pytest.approx(
        [0.1424127646366742, 0.9898073571756374, -1.2227806417238052e-5]
    )
    assert mirror_axes[6][1] == pytest.approx(
        [-0.24723561035883226, 0.968955392654959, -3.920484908028873e-6]
    )
    assert mirror_axes[7][1] == pytest.approx(
        [-0.5992235716499141, 0.800581732963154, 4.983322690691982e-6]
    )
    assert mirror_axes[8][1] == pytest.approx(
        [-0.8599694047950066, 0.5103455914146812, 1.1837289236101089e-5]
    )
    assert coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "D8h"
    assert coords.symmetry_number(point_group) == 16


def test_can_understand_D2d_symmetry():
    """Ensure values match regression logfiles for D2d symmetry."""
    data = logfiles["symmetries"]["allene"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([3.59943307, 58.28028795, 58.2804547])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    # assert axes == pytest.approx(
    #     np.array(
    #         [[1.0, 0.0, 0.0], [0.0, 1.0, 8.04637468e-11], [0.0, -8.04637468e-11, 1.0]]
    #     )
    # )
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 3
    assert len(groups[0]) == 1
    assert len(groups[1]) == 2
    assert len(groups[2]) == 4
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric prolate", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 3
    assert proper_axes[0][0] == 2
    assert proper_axes[1][0] == 2
    assert proper_axes[2][0] == 2
    assert proper_axes[0][1] == pytest.approx([1.0, 0.0, 0.0])
    assert proper_axes[1][1] == pytest.approx(
        [1.0104893446543213e-5, -0.9309114503766126, -0.36524494720064615]
    )
    assert proper_axes[2][1] == pytest.approx(
        [-0.00012735231509610425, 0.365345598359124, -0.9308719447598587]
    )
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 1
    assert improper_axes[0][0] == 4
    assert improper_axes[0][1] == proper_axes[0][1]
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 2
    assert mirror_axes[0][0] == "v"
    assert mirror_axes[1][0] == "v"
    assert mirror_axes[0][1] == pytest.approx(
        [-2.890827176799215e-5, -0.3999675757433585, -0.9165292889582376]
    )
    assert mirror_axes[1][1] == pytest.approx(
        [-9.719161506363647e-5, 0.9165428631171116, -0.39993645823165147]
    )
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "D2d"
    assert coords.symmetry_number(point_group) == 4

    data = logfiles["symmetries"]["cyclooctatetraene"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([191.055088155, 191.055088155, 338.69081546])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    # assert axes == pytest.approx(
    #     np.array(
    #         [[-1.0, -6.58647258e-10, 0.0],
    #          [-6.58647258e-10, 1.0, 0.0],
    #          [0.0, 0.0, 1.0]]
    #     )
    # )
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 2
    assert len(groups[0]) == 8
    assert len(groups[1]) == 8
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric oblate", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 3
    assert proper_axes[0][0] == 2
    assert proper_axes[1][0] == 2
    assert proper_axes[2][0] == 2
    # assert proper_axes[0][1] == pytest.approx(
    #     [0.99999999999998, 1.9484724096899716e-7, -4.445039596172754e-8]
    # )
    # assert proper_axes[1][1] == pytest.approx([2.527746904635184e-10, -1.0, 0.0])
    assert proper_axes[2][1] == pytest.approx([0.0, 0.0, 1.0])
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 1
    assert improper_axes[0][0] == 4
    assert improper_axes[0][1] == proper_axes[2][1]
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 2
    assert mirror_axes[0][0] == "v"
    assert mirror_axes[1][0] == "v"
    assert mirror_axes[0][1] == pytest.approx(
        [0.7071067813652857, -0.7071067810078088, -3.143117639000838e-8]
    )
    assert mirror_axes[1][1] == pytest.approx(
        [-0.707106781007808, -0.7071067813652863, 3.143117640046285e-8]
    )
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "D2d"
    assert coords.symmetry_number(point_group) == 4


def test_can_understand_D3d_symmetry():
    """Ensure values match regression logfiles for D3d symmetry."""
    data = logfiles["ethane"]["staggered@B97-3c"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([6.22039386, 25.07248713, 25.07267747])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(
        np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 4.65287808e-11], [0.0, -4.65286698e-11, 1.0]]
        ),
        abs=1e-6,
    )
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 2
    assert len(groups[0]) == 2
    assert len(groups[1]) == 6
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric prolate", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 4
    assert proper_axes[0][0] == 3
    assert proper_axes[1][0] == 2
    assert proper_axes[2][0] == 2
    assert proper_axes[3][0] == 2
    assert proper_axes[0][1] == pytest.approx([1.0, 0.0, 0.0])
    assert proper_axes[1][1] == pytest.approx(
        [5.114492593761218e-8, -0.7705968636328516, -0.6373228959948086]
    )
    assert proper_axes[2][1] == pytest.approx(
        [-0.00015351432373582197, 0.9372449845530069, -0.3486715006472228]
    )
    assert proper_axes[3][1] == pytest.approx(
        [-0.00015396128883686421, 0.1666592192271586, -0.9860145439812312]
    )
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 4
    assert improper_axes[0][0] == 6
    assert improper_axes[1][0] == 2
    assert improper_axes[2][0] == 2
    assert improper_axes[3][0] == 2
    assert improper_axes[0][1] == proper_axes[0][1]
    assert improper_axes[1][1] == proper_axes[1][1]
    assert improper_axes[2][1] == proper_axes[2][1]
    assert improper_axes[3][1] == proper_axes[3][1]
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 3
    assert mirror_axes[0][0] == "v"
    assert mirror_axes[1][0] == "v"
    assert mirror_axes[2][0] == "v"
    assert mirror_axes[0][1] == pytest.approx(
        [-4.430441013332084e-7, -0.7705965310112377, -0.6373232981723249]
    )
    assert mirror_axes[1][1] == pytest.approx(
        [-0.00015394529321121787, 0.937245178895647, -0.3486709780548323]
    )
    assert mirror_axes[2][1] == pytest.approx(
        [-0.00015439235712643497, 0.16665964180691425, -0.9860144724880011]
    )
    assert coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "D3d"
    assert coords.symmetry_number(point_group) == 6

    data = logfiles["symmetries"]["cyclohexane-chair"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([119.83069224, 119.84744745, 209.85483434])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    # assert axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 3
    assert len(groups[0]) == 6
    assert len(groups[1]) == 6
    assert len(groups[2]) == 6
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric oblate", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 4
    assert proper_axes[0][0] == 3
    assert proper_axes[1][0] == 2
    assert proper_axes[2][0] == 2
    assert proper_axes[3][0] == 2
    assert proper_axes[0][1] == pytest.approx([0.0, 0.0, 1.0])
    assert proper_axes[1][1] == pytest.approx(
        [0.9793333381651573, 0.20225284225808615, -2.3596930174470247e-5]
    )
    assert proper_axes[2][1] == pytest.approx(
        [0.6648303313785495, -0.7469943964168028, 4.691465433562472e-5]
    )
    assert proper_axes[3][1] == pytest.approx(
        [0.3145021354292675, 0.9492567628633137, -7.048895828035251e-5]
    )
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 4
    assert improper_axes[0][0] == 6
    assert improper_axes[1][0] == 2
    assert improper_axes[2][0] == 2
    assert improper_axes[3][0] == 2
    assert improper_axes[0][1] == proper_axes[0][1]
    assert improper_axes[1][1] == proper_axes[1][1]
    assert improper_axes[2][1] == proper_axes[2][1]
    assert improper_axes[3][1] == proper_axes[3][1]
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 3
    assert mirror_axes[0][0] == "v"
    assert mirror_axes[1][0] == "v"
    assert mirror_axes[2][0] == "v"
    assert mirror_axes[0][1] == pytest.approx(
        [0.9793333411211597, 0.20225282791876842, -2.3818641596810946e-5]
    )
    assert mirror_axes[1][1] == pytest.approx(
        [0.3145021532611901, 0.9492567569536162, -7.051225388736743e-5]
    )
    assert mirror_axes[2][1] == pytest.approx(
        [-0.6648303780404605, 0.7469943548888298, -4.689135533812219e-5]
    )
    assert coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "D3d"
    assert coords.symmetry_number(point_group) == 6


def test_can_understand_D4d_symmetry():
    """Ensure values match regression logfiles for D4d symmetry."""
    data = logfiles["symmetries"]["S8"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([810.52396682, 810.88788286, 1489.78398196])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 1
    assert len(groups[0]) == 8
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric oblate", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 5
    assert proper_axes[0][0] == 4
    assert proper_axes[1][0] == 2
    assert proper_axes[2][0] == 2
    assert proper_axes[3][0] == 2
    assert proper_axes[4][0] == 2
    assert proper_axes[0][1] == pytest.approx([0.0, 0.0, 1.0])
    assert proper_axes[1][1] == pytest.approx(
        [0.963244226087016, 0.26862717752913695, -2.0068760230898394e-5]
    )
    assert proper_axes[2][1] == pytest.approx(
        [0.8711000424991444, -0.49110557620917344, -0.00017021822609194343]
    )
    assert proper_axes[3][1] == pytest.approx(
        [0.2686942695811404, -0.9632255126101048, 3.675946444367664e-5]
    )
    assert proper_axes[4][1] == pytest.approx(
        [-0.49116983754349636, -0.8710638121055401, 0.00016102068144324463]
    )
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 1
    assert improper_axes[0][0] == 8
    assert improper_axes[0][1] == proper_axes[0][1]
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 4
    assert mirror_axes[0][0] == "v"
    assert mirror_axes[1][0] == "v"
    assert mirror_axes[2][0] == "v"
    assert mirror_axes[3][0] == "v"
    assert mirror_axes[0][1] == pytest.approx(
        [0.9927314777520891, -0.12035026681982787, -0.00016234700399761942]
    )
    assert mirror_axes[1][1] == pytest.approx(
        [0.7872478334947965, 0.6166366641024777, -0.00027044836345276453]
    )
    assert mirror_axes[2][1] == pytest.approx(
        [0.6170237569414777, -0.7869444830981367, 0.00025276714708717554]
    )
    assert mirror_axes[3][1] == pytest.approx(
        [0.12051512195987304, 0.9927114721259069, 0.0001961851471428286]
    )
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "D4d"
    assert coords.symmetry_number(point_group) == 8


def test_can_understand_D5d_symmetry():
    """Ensure values match regression logfiles for D5d symmetry."""
    data = logfiles["symmetries"]["ferrocene-staggered"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([233.36759737, 470.55761366, 470.55870003])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(
        np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, -3.14344106e-11], [0.0, 3.14344106e-11, 1.0]]
        ),
        abs=1e-6,
    )
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 3
    assert len(groups[0]) == 1
    assert len(groups[1]) == 10
    assert len(groups[2]) == 10
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric prolate", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 6
    assert proper_axes[0][0] == 5
    assert proper_axes[1][0] == 2
    assert proper_axes[2][0] == 2
    assert proper_axes[3][0] == 2
    assert proper_axes[4][0] == 2
    assert proper_axes[5][0] == 2
    assert proper_axes[0][1] == pytest.approx([1.0, 0.0, 0.0])
    assert proper_axes[1][1] == pytest.approx(
        [3.573651507251015e-5, -0.8090202704430335, 0.5877807420587906]
    )
    assert proper_axes[2][1] == pytest.approx(
        [4.784454416974403e-6, -0.30902881328774495, 0.951052676004372]
    )
    assert proper_axes[3][1] == pytest.approx([0.0, -1.0, 0.0])
    assert proper_axes[4][1] == pytest.approx(
        [-2.956966317827157e-6, -0.8090171488269479, -0.5877850397000359]
    )
    assert proper_axes[5][1] == pytest.approx(
        [-4.784454416974403e-6, -0.30902881328774495, -0.951052676004372]
    )
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 6
    assert improper_axes[0][0] == 10
    assert improper_axes[1][0] == 2
    assert improper_axes[2][0] == 2
    assert improper_axes[3][0] == 2
    assert improper_axes[4][0] == 2
    assert improper_axes[5][0] == 2
    assert improper_axes[0][1] == proper_axes[0][1]
    assert improper_axes[1][1] == proper_axes[1][1]
    assert improper_axes[2][1] == proper_axes[2][1]
    assert improper_axes[3][1] == proper_axes[3][1]
    assert improper_axes[4][1] == proper_axes[4][1]
    assert improper_axes[5][1] == proper_axes[5][1]
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 5
    assert mirror_axes[0][0] == "v"
    assert mirror_axes[1][0] == "v"
    assert mirror_axes[2][0] == "v"
    assert mirror_axes[3][0] == "v"
    assert mirror_axes[4][0] == "v"
    assert mirror_axes[0][1] == pytest.approx(
        [2.504350433851091e-5, -0.3090138858276231, 0.9510575259880631]
    )
    assert mirror_axes[1][1] == pytest.approx(
        [4.784454416974403e-6, 0.30902881328774495, 0.951052676004372]
    )
    assert mirror_axes[2][1] == pytest.approx([0.0, -1.0, 0.0])
    assert mirror_axes[3][1] == pytest.approx(
        [-3.573651507251015e-5, 0.8090202704430335, -0.5877807420587906]
    )
    assert mirror_axes[4][1] == pytest.approx(
        [-3.573651507251015e-5, -0.8090202704430335, -0.5877807420587906]
    )
    assert coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "D5d"
    assert coords.symmetry_number(point_group) == 10


def test_can_understand_S4_symmetry():
    """Ensure values match regression logfiles for S4 symmetry."""
    data = logfiles["symmetries"]["tetrachloroneopentane"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([675.95701835, 948.8492401, 948.9291113])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(
        np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, -3.49142937e-12], [0.0, 3.49131835e-12, 1.0]]
        ),
        abs=1e-6,
    )
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 4
    assert len(groups[0]) == 1
    assert len(groups[1]) == 4
    assert len(groups[2]) == 4
    assert len(groups[3]) == 8
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric prolate", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 1
    assert proper_axes[0][0] == 2
    assert proper_axes[0][1] == pytest.approx([1.0, 0.0, 0.0])
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 1
    assert improper_axes[0][0] == 4
    assert improper_axes[0][1] == proper_axes[0][1]
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 0
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "S4"
    assert coords.symmetry_number(point_group) == 2

    data = logfiles["symmetries"]["1,3,5,7-tetrachlorocyclooctatetraene"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([1124.75960399, 1124.76010676, 1717.87114398])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    # assert axes == pytest.approx(
    #     np.array(
    #         [[1.0, -8.47946613e-11, 0.0], [8.47946613e-11, 1.0, 0.0], [0.0, 0.0, 1.0]]
    #     )
    # )
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 3
    assert len(groups[0]) == 4
    assert len(groups[1]) == 4
    assert len(groups[2]) == 8
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric oblate", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 1
    assert proper_axes[0][0] == 2
    assert proper_axes[0][1] == pytest.approx([0.0, 0.0, 1.0])
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 1
    assert improper_axes[0][0] == 4
    assert improper_axes[0][1] == proper_axes[0][1]
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 0
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "S4"
    assert coords.symmetry_number(point_group) == 2

    data = logfiles["symmetries"]["tetraphenylborate-"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([2328.48397615, 2573.6635109, 2573.66376861])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    # assert axes == pytest.approx(
    #     np.array(
    #         [[1.0, 0.0, 0.0], [0.0, -1.0, 1.78502646e-9], [0.0, -1.78502657e-9, -1.0]]
    #     )
    # )
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 12
    assert len(groups[0]) == 1
    assert len(groups[1]) == 4
    assert len(groups[2]) == 4
    assert len(groups[3]) == 4
    assert len(groups[4]) == 4
    assert len(groups[5]) == 4
    assert len(groups[6]) == 4
    assert len(groups[7]) == 4
    assert len(groups[8]) == 4
    assert len(groups[9]) == 4
    assert len(groups[10]) == 4
    assert len(groups[11]) == 4
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric prolate", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 1
    assert proper_axes[0][0] == 2
    assert proper_axes[0][1] == pytest.approx([1.0, 0.0, 0.0])
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 1
    assert improper_axes[0][0] == 4
    assert improper_axes[0][1] == proper_axes[0][1]
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 0
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "S4"
    assert coords.symmetry_number(point_group) == 2


def test_can_understand_Td_symmetry():
    """Ensure values match regression logfiles for Td symmetry."""
    data = logfiles["tanaka1996"]["methane@UMP2/6-311G(2df,2pd)"]  # tetrahedron
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([3.182947905, 3.182947905, 3.182947905], 1e-2)
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 2
    assert len(groups[0]) == 1
    assert len(groups[1]) == 4
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("spheric", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 7
    assert proper_axes[0][0] == 3
    assert proper_axes[1][0] == 3
    assert proper_axes[2][0] == 3
    assert proper_axes[3][0] == 3
    assert proper_axes[4][0] == 2
    assert proper_axes[5][0] == 2
    assert proper_axes[6][0] == 2
    assert proper_axes[0][1] == pytest.approx(
        [0.5773502691896257, 0.5773502691896257, 0.5773502691896257]
    )
    assert proper_axes[1][1] == pytest.approx(
        [0.5773502691896257, -0.5773502691896257, -0.5773502691896257]
    )
    assert proper_axes[2][1] == pytest.approx(
        [-0.5773502691896257, 0.5773502691896257, -0.5773502691896257]
    )
    assert proper_axes[3][1] == pytest.approx(
        [-0.5773502691896257, -0.5773502691896257, 0.5773502691896257]
    )
    assert proper_axes[4][1] == pytest.approx([0.0, 0.0, -1.0])
    assert proper_axes[5][1] == pytest.approx([0.0, -1.0, 0.0])
    assert proper_axes[6][1] == pytest.approx([-1.0, 0.0, 0.0])
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 3
    assert improper_axes[0][0] == 4
    assert improper_axes[1][0] == 4
    assert improper_axes[2][0] == 4
    assert improper_axes[0][1] == proper_axes[4][1]
    assert improper_axes[1][1] == proper_axes[5][1]
    assert improper_axes[2][1] == proper_axes[6][1]
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 6
    assert mirror_axes[0][0] == "v"
    assert mirror_axes[1][0] == "v"
    assert mirror_axes[2][0] == "v"
    assert mirror_axes[3][0] == "v"
    assert mirror_axes[4][0] == "v"
    assert mirror_axes[5][0] == "v"
    assert mirror_axes[0][1] == pytest.approx(
        [0.7071067811865476, -0.7071067811865476, 0.0]
    )
    assert mirror_axes[1][1] == pytest.approx(
        [0.0, -0.7071067811865476, 0.7071067811865476]
    )
    assert mirror_axes[2][1] == pytest.approx(
        [0.0, -0.7071067811865476, -0.7071067811865476]
    )
    assert mirror_axes[3][1] == pytest.approx(
        [-0.7071067811865476, 0.0, 0.7071067811865476]
    )
    assert mirror_axes[4][1] == pytest.approx(
        [-0.7071067811865476, 0.0, -0.7071067811865476]
    )
    assert mirror_axes[5][1] == pytest.approx(
        [-0.7071067811865476, -0.7071067811865476, 0.0]
    )
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "Td"
    assert coords.symmetry_number(point_group) == 12

    data = logfiles["symmetries"]["tetrahedrane"]  # tetrahedron
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([37.54433184, 37.54433184, 37.54433184])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 2
    assert len(groups[0]) == 4
    assert len(groups[1]) == 4
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("spheric", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 7
    assert proper_axes[0][0] == 3
    assert proper_axes[1][0] == 3
    assert proper_axes[2][0] == 3
    assert proper_axes[3][0] == 3
    assert proper_axes[4][0] == 2
    assert proper_axes[5][0] == 2
    assert proper_axes[6][0] == 2
    assert proper_axes[0][1] == pytest.approx(
        [0.5773502691896257, 0.5773502691896257, 0.5773502691896257]
    )
    assert proper_axes[1][1] == pytest.approx(
        [0.5773502691896257, -0.5773502691896257, -0.5773502691896257]
    )
    assert proper_axes[2][1] == pytest.approx(
        [-0.5773502691896257, 0.5773502691896257, -0.5773502691896257]
    )
    assert proper_axes[3][1] == pytest.approx(
        [-0.5773502691896257, -0.5773502691896257, 0.5773502691896257]
    )
    assert proper_axes[4][1] == pytest.approx([1.0, 0.0, 0.0])
    assert proper_axes[5][1] == pytest.approx([0.0, 0.0, 1.0])
    assert proper_axes[6][1] == pytest.approx([0.0, -1.0, 0.0])
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 3
    assert improper_axes[0][0] == 4
    assert improper_axes[1][0] == 4
    assert improper_axes[2][0] == 4
    assert improper_axes[0][1] == proper_axes[4][1]
    assert improper_axes[1][1] == proper_axes[5][1]
    assert improper_axes[2][1] == proper_axes[6][1]
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 6
    assert mirror_axes[0][0] == "v"
    assert mirror_axes[1][0] == "v"
    assert mirror_axes[2][0] == "v"
    assert mirror_axes[3][0] == "v"
    assert mirror_axes[4][0] == "v"
    assert mirror_axes[5][0] == "v"
    assert mirror_axes[0][1] == pytest.approx(
        [0.7071067811865476, 0.0, 0.7071067811865476]
    )
    assert mirror_axes[1][1] == pytest.approx(
        [0.7071067811865476, 0.0, -0.7071067811865476]
    )
    assert mirror_axes[2][1] == pytest.approx(
        [0.7071067811865476, -0.7071067811865476, 0.0]
    )
    assert mirror_axes[3][1] == pytest.approx(
        [0.0, -0.7071067811865476, 0.7071067811865476]
    )
    assert mirror_axes[4][1] == pytest.approx(
        [0.0, -0.7071067811865476, -0.7071067811865476]
    )
    assert mirror_axes[5][1] == pytest.approx(
        [-0.7071067811865476, -0.7071067811865476, 0.0]
    )
    assert not coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "Td"
    assert coords.symmetry_number(point_group) == 12


def test_can_understand_Oh_symmetry():
    """Ensure values match regression logfiles for Oh symmetry."""
    data = logfiles["symmetries"]["cubane"]  # hexahedron aka cube
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([152.81707349, 152.82212504, 152.82540554])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(
        np.array(
            [
                [1.0, 1.58240729e-12, 2.34515131e-12],
                [-1.58240729e-12, 1.0, -1.34253719e-11],
                [-2.34515131e-12, 1.34252609e-11, 1.0],
            ]
        ),
        abs=1e-6,
    )
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 2
    assert len(groups[0]) == 8
    assert len(groups[1]) == 8
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("spheric", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 13
    assert proper_axes[0][0] == 4
    assert proper_axes[1][0] == 4
    assert proper_axes[2][0] == 4
    assert proper_axes[3][0] == 3
    assert proper_axes[4][0] == 3
    assert proper_axes[5][0] == 3
    assert proper_axes[6][0] == 3
    assert proper_axes[7][0] == 2
    assert proper_axes[8][0] == 2
    assert proper_axes[9][0] == 2
    assert proper_axes[10][0] == 2
    assert proper_axes[11][0] == 2
    assert proper_axes[12][0] == 2
    assert proper_axes[0][1] == pytest.approx(
        [0.968308520495324, -0.11479281066604227, -0.22181347965249296]
    )
    assert proper_axes[1][1] == pytest.approx(
        [0.20662295557340749, 0.8670054245107972, 0.45344078786426145]
    )
    assert proper_axes[2][1] == pytest.approx(
        [0.14026927413438356, -0.4849352789341263, 0.863227841290406]
    )
    assert proper_axes[3][1] == pytest.approx(
        [0.7594175301605084, 0.15422201930436788, 0.632060585424053]
    )
    assert proper_axes[4][1] == pytest.approx(
        [0.5207643090392388, -0.846788719179745, 0.10841309653732524]
    )
    assert proper_axes[5][1] == pytest.approx(
        [-0.35890424644962937, 0.2869097992842958, 0.8881838261053108]
    )
    assert proper_axes[6][1] == pytest.approx(
        [-0.5974379572306865, -0.7141760979809005, 0.36471960234240147]
    )
    assert proper_axes[7][1] == pytest.approx(
        [0.8308871547755614, 0.5317883462007422, 0.16379160806968113]
    )
    assert proper_axes[8][1] == pytest.approx(
        [0.7839254119807916, -0.42410024141273434, 0.45343128882379397]
    )
    assert proper_axes[9][1] == pytest.approx(
        [0.2452941410444507, 0.27016222944019364, 0.9310441204116856]
    )
    assert proper_axes[10][1] == pytest.approx(
        [-0.04695657771738413, -0.9559494552467201, 0.2897511325647759]
    )
    assert proper_axes[11][1] == pytest.approx(
        [-0.5386892802030987, 0.6941984955014298, 0.47739114805138466]
    )
    assert proper_axes[12][1] == pytest.approx(
        [-0.5856129657464667, -0.26164449901693576, 0.7672024572978141]
    )
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 13
    assert improper_axes[0][0] == 6
    assert improper_axes[1][0] == 6
    assert improper_axes[2][0] == 6
    assert improper_axes[3][0] == 6
    assert improper_axes[4][0] == 4
    assert improper_axes[5][0] == 4
    assert improper_axes[6][0] == 4
    assert improper_axes[7][0] == 2
    assert improper_axes[8][0] == 2
    assert improper_axes[9][0] == 2
    assert improper_axes[10][0] == 2
    assert improper_axes[11][0] == 2
    assert improper_axes[12][0] == 2
    assert improper_axes[0][1] == proper_axes[3][1]
    assert improper_axes[1][1] == proper_axes[4][1]
    assert improper_axes[2][1] == proper_axes[5][1]
    assert improper_axes[3][1] == proper_axes[6][1]
    assert improper_axes[4][1] == proper_axes[0][1]
    assert improper_axes[5][1] == proper_axes[1][1]
    assert improper_axes[6][1] == proper_axes[2][1]
    assert improper_axes[7][1] == proper_axes[7][1]
    assert improper_axes[8][1] == proper_axes[8][1]
    assert improper_axes[9][1] == proper_axes[9][1]
    assert improper_axes[10][1] == proper_axes[10][1]
    assert improper_axes[11][1] == proper_axes[11][1]
    assert improper_axes[12][1] == proper_axes[12][1]
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 9
    assert mirror_axes[0][0] == "h"
    assert mirror_axes[1][0] == "h"
    assert mirror_axes[2][0] == "h"
    assert mirror_axes[3][0] == "v"
    assert mirror_axes[4][0] == "v"
    assert mirror_axes[5][0] == "v"
    assert mirror_axes[6][0] == "v"
    assert mirror_axes[7][0] == "v"
    assert mirror_axes[8][0] == "v"
    assert mirror_axes[0][1] == pytest.approx(
        [0.1402494302759853, -0.4850199193326482, 0.863183511866285]
    )
    assert mirror_axes[1][1] == pytest.approx(
        [-0.20660962696401017, -0.867052560973589, -0.4533567232929642]
    )
    assert mirror_axes[2][1] == pytest.approx(
        [-0.9683115428116573, 0.11488480242450885, 0.22175265100915806]
    )
    assert mirror_axes[3][1] == pytest.approx(
        [0.7839107873403623, -0.424053445907144, 0.4535003335232401]
    )
    assert mirror_axes[4][1] == pytest.approx(
        [0.24529741116177126, 0.2702228702802404, 0.9310256604706667]
    )
    assert mirror_axes[5][1] == pytest.approx(
        [-0.046980250747360834, -0.9559757997415224, 0.28966036378536764]
    )
    assert mirror_axes[6][1] == pytest.approx(
        [-0.5386394026228776, 0.6941906525287401, 0.47745882742262874]
    )
    assert mirror_axes[7][1] == pytest.approx(
        [-0.5856652230138192, -0.26168375604832583, 0.7671491760880759]
    )
    assert mirror_axes[8][1] == pytest.approx(
        [-0.8309038705883419, -0.531787711389286, -0.16370885088063244]
    )
    assert coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "Oh"
    assert coords.symmetry_number(point_group) == 24

    data = logfiles["symmetries"]["SF6"]  # octahedron
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([195.62987814, 195.64569248, 195.66607271])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(
        np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, -3.53539420e-12], [0.0, 3.53528318e-12, -1.0]]
        )
    )
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 2
    assert len(groups[0]) == 1
    assert len(groups[1]) == 6
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("spheric", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 13
    assert proper_axes[0][0] == 4
    assert proper_axes[1][0] == 4
    assert proper_axes[2][0] == 4
    assert proper_axes[3][0] == 3
    assert proper_axes[4][0] == 3
    assert proper_axes[5][0] == 3
    assert proper_axes[6][0] == 3
    assert proper_axes[7][0] == 2
    assert proper_axes[8][0] == 2
    assert proper_axes[9][0] == 2
    assert proper_axes[10][0] == 2
    assert proper_axes[11][0] == 2
    assert proper_axes[12][0] == 2
    assert proper_axes[0][1] == pytest.approx(
        [0.20842728261025004, 0.9069406581468825, 0.36608292839711426]
    )
    assert proper_axes[1][1] == pytest.approx(
        [-0.6591913953359633, 0.40680141589916896, -0.6324391767889282]
    )
    assert proper_axes[2][1] == pytest.approx(
        [-0.7226336862477715, -0.10950265597784735, 0.6825025449284087]
    )
    assert proper_axes[3][1] == pytest.approx(
        [0.9180372654355696, 0.35199529029607535, 0.18250176678860658]
    )
    assert proper_axes[4][1] == pytest.approx(
        [0.6773692961070812, -0.695282947684529, -0.24031741374796345]
    )
    assert proper_axes[5][1] == pytest.approx(
        [0.15693255867682493, 0.8216792490527968, -0.547919139748902]
    )
    assert proper_axes[6][1] == pytest.approx(
        [-0.08368409490810279, -0.22550865619699081, -0.97064041654963]
    )
    assert proper_axes[7][1] == pytest.approx(
        [0.6583352159729611, 0.7187052894540965, -0.2237352236920646]
    )
    assert proper_axes[8][1] == pytest.approx(
        [0.6135077290343502, 0.3536546811989033, 0.7060712661489746]
    )
    assert proper_axes[9][1] == pytest.approx(
        [0.044864368068234406, 0.36511253176985675, -0.92988172776028]
    )
    assert proper_axes[10][1] == pytest.approx(
        [-0.3187354165588673, 0.9289431135238319, -0.18834124898092444]
    )
    assert proper_axes[11][1] == pytest.approx(
        [-0.3636141504484658, 0.5638938311603796, 0.7414907260194895]
    )
    assert proper_axes[12][1] == pytest.approx(
        [-0.9770164502779867, 0.21020438400853103, 0.03539735625433769]
    )
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 13
    assert improper_axes[0][0] == 6
    assert improper_axes[1][0] == 6
    assert improper_axes[2][0] == 6
    assert improper_axes[3][0] == 6
    assert improper_axes[4][0] == 4
    assert improper_axes[5][0] == 4
    assert improper_axes[6][0] == 4
    assert improper_axes[7][0] == 2
    assert improper_axes[8][0] == 2
    assert improper_axes[9][0] == 2
    assert improper_axes[10][0] == 2
    assert improper_axes[11][0] == 2
    assert improper_axes[12][0] == 2
    assert improper_axes[0][1] == proper_axes[3][1]
    assert improper_axes[1][1] == proper_axes[4][1]
    assert improper_axes[2][1] == proper_axes[5][1]
    assert improper_axes[3][1] == proper_axes[6][1]
    assert improper_axes[4][1] == proper_axes[0][1]
    assert improper_axes[5][1] == proper_axes[1][1]
    assert improper_axes[6][1] == proper_axes[2][1]
    assert improper_axes[7][1] == proper_axes[7][1]
    assert improper_axes[8][1] == proper_axes[8][1]
    assert improper_axes[9][1] == proper_axes[9][1]
    assert improper_axes[10][1] == proper_axes[10][1]
    assert improper_axes[11][1] == proper_axes[11][1]
    assert improper_axes[12][1] == proper_axes[12][1]
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 9
    assert mirror_axes[0][0] == "h"
    assert mirror_axes[1][0] == "h"
    assert mirror_axes[2][0] == "h"
    assert mirror_axes[3][0] == "v"
    assert mirror_axes[4][0] == "v"
    assert mirror_axes[5][0] == "v"
    assert mirror_axes[6][0] == "v"
    assert mirror_axes[7][0] == "v"
    assert mirror_axes[8][0] == "v"
    assert mirror_axes[0][1] == pytest.approx(
        [0.20842727455881338, 0.9069406295933482, 0.36608300372020325]
    )
    assert mirror_axes[1][1] == pytest.approx(
        [-0.6591914380472336, 0.4068014132082195, -0.6324391340018544]
    )
    assert mirror_axes[2][1] == pytest.approx(
        [-0.7226337899365044, -0.10950260471276833, 0.6825024433678778]
    )
    assert mirror_axes[3][1] == pytest.approx(
        [0.9770164393202924, -0.21020448690808052, -0.035397047639711646]
    )
    assert mirror_axes[4][1] == pytest.approx(
        [0.6583351398364776, 0.7187052740147074, -0.22373549731754638]
    )
    assert mirror_axes[5][1] == pytest.approx(
        [0.6135076831741578, 0.35365462575178835, 0.7060713337692159]
    )
    assert mirror_axes[6][1] == pytest.approx(
        [-0.044864382245252285, -0.36511251216163504, 0.9298817347753271]
    )
    assert mirror_axes[7][1] == pytest.approx(
        [-0.31873544248905106, 0.9289431247821802, -0.188341149569678]
    )
    assert mirror_axes[8][1] == pytest.approx(
        [-0.3636141885662752, 0.5638940339581285, 0.7414905531021403]
    )
    assert coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "Oh"
    assert coords.symmetry_number(point_group) == 24


def test_can_understand_Ih_symmetry():
    """Ensure values match regression logfiles for Ih symmetry."""
    data = logfiles["symmetries"]["B12H12-2"]  # icosahedron
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([323.38198873, 323.39397591, 323.41051849])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    # assert axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 2
    assert len(groups[0]) == 12
    assert len(groups[1]) == 12
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("spheric", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 1
    assert proper_axes[0][0] == 5
    assert proper_axes[0][1] == pytest.approx(
        [0.32276287512969803, -0.8506628423519896, 0.41496608907192073]
    )
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 1
    assert improper_axes[0][0] == 10
    assert improper_axes[0][1] == proper_axes[0][1]
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 15
    assert mirror_axes[0][0] == ""
    assert mirror_axes[1][0] == ""
    assert mirror_axes[2][0] == ""
    assert mirror_axes[3][0] == ""
    assert mirror_axes[4][0] == ""
    assert mirror_axes[5][0] == ""
    assert mirror_axes[6][0] == ""
    assert mirror_axes[7][0] == ""
    assert mirror_axes[8][0] == ""
    assert mirror_axes[9][0] == ""
    assert mirror_axes[10][0] == "v"
    assert mirror_axes[11][0] == "v"
    assert mirror_axes[12][0] == "v"
    assert mirror_axes[13][0] == "v"
    assert mirror_axes[14][0] == "v"
    assert mirror_axes[0][1] == pytest.approx(
        [0.7406252312928405, -0.5000080663393951, 0.44884986395003595]
    )
    assert mirror_axes[1][1] == pytest.approx(
        [0.5844072246729994, 0.8090067396306037, -0.06305783838875868]
    )
    assert mirror_axes[2][1] == pytest.approx(
        [0.5843682562650157, -0.8090320546656534, -0.06309418034380233]
    )
    assert mirror_axes[3][1] == pytest.approx(
        [0.3315804589223854, 0.30902105108328864, -0.8913811694489624]
    )
    assert mirror_axes[4][1] == pytest.approx(
        [0.2528103591750081, -0.5000190334881637, 0.8282921516248757]
    )
    assert mirror_axes[5][1] == pytest.approx(
        [-1.7851852306723233e-13, -1.0, -6.642652751092994e-14]
    )
    assert mirror_axes[6][1] == pytest.approx(
        [-0.20489043911377544, 0.8090477535862791, 0.5508735248464567]
    )
    assert mirror_axes[7][1] == pytest.approx(
        [-0.20492966629436482, -0.8090141005350259, 0.5509083562698821]
    )
    assert mirror_axes[8][1] == pytest.approx(
        [-0.6139542311602695, 1.620534225899669e-13, -0.7893416256858639]
    )
    assert mirror_axes[9][1] == pytest.approx(
        [-0.9455539364150436, 0.30903900618790064, 0.10209136096854846]
    )
    assert mirror_axes[10][1] == pytest.approx(
        [0.7406572278221821, 0.5000169176578358, 0.44878720228045393]
    )
    assert mirror_axes[11][1] == pytest.approx(
        [0.2528103591751866, 0.5000190334879634, 0.828292151624942]
    )
    assert mirror_axes[12][1] == pytest.approx(
        [-0.3315804589222751, 0.3090210510832886, 0.8913811694490034]
    )
    assert mirror_axes[13][1] == pytest.approx(
        [-0.7892975299967038, 1.001511141455943e-13, 0.6140109193989163]
    )
    assert mirror_axes[14][1] == pytest.approx(
        [-0.945553936415154, -0.30903900618757657, 0.10209136096850731]
    )
    assert coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "Ih"
    assert coords.symmetry_number(point_group) == 60

    data = logfiles["symmetries"]["dodecahedrane"]  # dodecahedron
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([913.24407956, 913.29418754, 913.31698587])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    assert axes == pytest.approx(
        np.array(
            [
                [1.0, 1.70162761e-12, -9.25867193e-13],
                [-1.70162761e-12, 1.0, -1.20725652e-11],
                [9.25867193e-13, 1.20723431e-11, 1.0],
            ]
        ),
        abs=1e-6,
    )
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 2
    assert len(groups[0]) == 20
    assert len(groups[1]) == 20
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("spheric", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 26
    assert proper_axes[0][0] == 5
    assert proper_axes[1][0] == 3
    assert proper_axes[2][0] == 3
    assert proper_axes[3][0] == 3
    assert proper_axes[4][0] == 3
    assert proper_axes[5][0] == 3
    assert proper_axes[6][0] == 3
    assert proper_axes[7][0] == 3
    assert proper_axes[8][0] == 3
    assert proper_axes[9][0] == 3
    assert proper_axes[10][0] == 3
    assert proper_axes[11][0] == 2
    assert proper_axes[12][0] == 2
    assert proper_axes[13][0] == 2
    assert proper_axes[14][0] == 2
    assert proper_axes[15][0] == 2
    assert proper_axes[16][0] == 2
    assert proper_axes[17][0] == 2
    assert proper_axes[18][0] == 2
    assert proper_axes[19][0] == 2
    assert proper_axes[20][0] == 2
    assert proper_axes[21][0] == 2
    assert proper_axes[22][0] == 2
    assert proper_axes[23][0] == 2
    assert proper_axes[24][0] == 2
    assert proper_axes[25][0] == 2
    assert proper_axes[0][1] == pytest.approx(
        [0.30812755902834976, -0.8903567199966111, -0.3351452200526021]
    )
    assert proper_axes[1][1] == pytest.approx(
        [0.8726443120221634, 0.48109350478653806, 0.08391033516540448]
    )
    assert proper_axes[2][1] == pytest.approx(
        [0.8614074576994745, -0.10001871102175852, 0.49796932562686447]
    )
    assert proper_axes[3][1] == pytest.approx(
        [0.7596892134828486, 0.27409255736379445, -0.5896995581782184]
    )
    assert proper_axes[4][1] == pytest.approx(
        [0.7414824163231822, -0.666144750744675, 0.0803429980698362]
    )
    assert proper_axes[5][1] == pytest.approx(
        [0.3302346609505096, 0.9016241192927089, 0.27931884328943724]
    )
    assert proper_axes[6][1] == pytest.approx(
        [0.3120109859687541, -0.03865184151950474, 0.9492919360143943]
    )
    assert proper_axes[7][1] == pytest.approx(
        [-0.016260703551204576, 0.5804485493590502, 0.8141345534166905]
    )
    assert proper_axes[8][1] == pytest.approx(
        [-0.1179699843211733, 0.9546064692510375, -0.27351338479739845]
    )
    assert proper_axes[9][1] == pytest.approx(
        [-0.1474452224156102, -0.5667983189879612, 0.8105551011370126]
    )
    assert proper_axes[10][1] == pytest.approx(
        [-0.6786441697763973, 0.4349077820354078, 0.5918591994330631]
    )
    assert proper_axes[11][1] == pytest.approx(
        [0.928118614537316, 0.20396514164540264, 0.31143869114627337]
    )
    assert proper_axes[12][1] == pytest.approx(
        [0.8736884535306767, 0.4042058565781007, -0.27071407734940317]
    )
    assert proper_axes[13][1] == pytest.approx(
        [0.8579208244577288, -0.41007394027335836, 0.3095338793580648]
    )
    assert proper_axes[14][1] == pytest.approx(
        [0.6438263259775652, 0.7400631620921857, 0.19440724804478193]
    )
    assert proper_axes[15][1] == pytest.approx(
        [0.6280638876411975, -0.07422235787999441, 0.7746139649088178]
    )
    assert proper_axes[16][1] == pytest.approx(
        [0.46001318415311654, -0.8674778123468325, 0.18939407459386004]
    )
    assert proper_axes[17][1] == pytest.approx(
        [0.1680489404647018, 0.7932497720982239, 0.585247257725149]
    )
    assert proper_axes[18][1] == pytest.approx(
        [0.15829526011504383, 0.2899900056203697, 0.943847661047803]
    )
    assert proper_axes[19][1] == pytest.approx(
        [0.113610536873139, 0.9935205074099875, 0.003106005022207571]
    )
    assert proper_axes[20][1] == pytest.approx(
        [0.08808142660688456, -0.32405646870648736, 0.9419283769885995]
    )
    assert proper_axes[21][1] == pytest.approx(
        [-0.015762043382657284, -0.8143123041659879, 0.5802129171884083]
    )
    assert proper_axes[22][1] == pytest.approx(
        [-0.3719270798702417, 0.5434681154659787, 0.7525374772933587]
    )
    assert proper_axes[23][1] == pytest.approx(
        [-0.48556796883698194, -0.4501297827597275, 0.7494043810335033]
    )
    assert proper_axes[24][1] == pytest.approx(
        [-0.7601520620354372, 0.5892327485753275, 0.27381309426256056]
    )
    assert proper_axes[25][1] == pytest.approx(
        [-0.7698674324966721, 0.08601045747669914, 0.6323814810584122]
    )
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 26
    assert improper_axes[0][0] == 10
    assert improper_axes[1][0] == 6
    assert improper_axes[2][0] == 6
    assert improper_axes[3][0] == 6
    assert improper_axes[4][0] == 6
    assert improper_axes[5][0] == 6
    assert improper_axes[6][0] == 6
    assert improper_axes[7][0] == 6
    assert improper_axes[8][0] == 6
    assert improper_axes[9][0] == 6
    assert improper_axes[10][0] == 6
    assert improper_axes[11][0] == 2
    assert improper_axes[12][0] == 2
    assert improper_axes[13][0] == 2
    assert improper_axes[14][0] == 2
    assert improper_axes[15][0] == 2
    assert improper_axes[16][0] == 2
    assert improper_axes[17][0] == 2
    assert improper_axes[18][0] == 2
    assert improper_axes[19][0] == 2
    assert improper_axes[20][0] == 2
    assert improper_axes[21][0] == 2
    assert improper_axes[22][0] == 2
    assert improper_axes[23][0] == 2
    assert improper_axes[24][0] == 2
    assert improper_axes[25][0] == 2
    assert improper_axes[0][1] == proper_axes[0][1]
    assert improper_axes[1][1] == proper_axes[1][1]
    assert improper_axes[2][1] == proper_axes[2][1]
    assert improper_axes[3][1] == proper_axes[3][1]
    assert improper_axes[4][1] == proper_axes[4][1]
    assert improper_axes[5][1] == proper_axes[5][1]
    assert improper_axes[6][1] == proper_axes[6][1]
    assert improper_axes[7][1] == proper_axes[7][1]
    assert improper_axes[8][1] == proper_axes[8][1]
    assert improper_axes[9][1] == proper_axes[9][1]
    assert improper_axes[10][1] == proper_axes[10][1]
    assert improper_axes[11][1] == proper_axes[11][1]
    assert improper_axes[12][1] == proper_axes[12][1]
    assert improper_axes[13][1] == proper_axes[13][1]
    assert improper_axes[14][1] == proper_axes[14][1]
    assert improper_axes[15][1] == proper_axes[15][1]
    assert improper_axes[16][1] == proper_axes[16][1]
    assert improper_axes[17][1] == proper_axes[17][1]
    assert improper_axes[18][1] == proper_axes[18][1]
    assert improper_axes[19][1] == proper_axes[19][1]
    assert improper_axes[20][1] == proper_axes[20][1]
    assert improper_axes[21][1] == proper_axes[21][1]
    assert improper_axes[22][1] == proper_axes[22][1]
    assert improper_axes[23][1] == proper_axes[23][1]
    assert improper_axes[24][1] == proper_axes[24][1]
    assert improper_axes[25][1] == proper_axes[25][1]
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 15
    assert mirror_axes[0][0] == ""
    assert mirror_axes[1][0] == ""
    assert mirror_axes[2][0] == ""
    assert mirror_axes[3][0] == ""
    assert mirror_axes[4][0] == ""
    assert mirror_axes[5][0] == ""
    assert mirror_axes[6][0] == ""
    assert mirror_axes[7][0] == ""
    assert mirror_axes[8][0] == ""
    assert mirror_axes[9][0] == ""
    assert mirror_axes[10][0] == "v"
    assert mirror_axes[11][0] == "v"
    assert mirror_axes[12][0] == "v"
    assert mirror_axes[13][0] == "v"
    assert mirror_axes[14][0] == "v"
    assert mirror_axes[0][1] == pytest.approx(
        [0.8579334690639634, -0.41004375526345205, 0.3095388205529633]
    )
    assert mirror_axes[1][1] == pytest.approx(
        [0.7601521051800151, -0.5892326301810166, -0.2738132292646812]
    )
    assert mirror_axes[2][1] == pytest.approx(
        [0.6438181879254126, 0.7400694139687545, 0.19441039942447158]
    )
    assert mirror_axes[3][1] == pytest.approx(
        [0.4600545615068519, -0.8674583597729, 0.18938266683318256]
    )
    assert mirror_axes[4][1] == pytest.approx(
        [0.16806298579330683, 0.7932770116125302, 0.5852063017888072]
    )
    assert mirror_axes[5][1] == pytest.approx(
        [0.1582924990882815, 0.2900529423545288, 0.9438287849837315]
    )
    assert mirror_axes[6][1] == pytest.approx(
        [0.11360805419593054, 0.9935208368212058, 0.0030914113129076734]
    )
    assert mirror_axes[7][1] == pytest.approx(
        [-0.015761991718039978, -0.8143123203369762, 0.580212895896403]
    )
    assert mirror_axes[8][1] == pytest.approx(
        [-0.37192961468555336, 0.5434156210090533, 0.7525741322708354]
    )
    assert mirror_axes[9][1] == pytest.approx(
        [-0.7699024189242863, 0.0859974901764, 0.632340649506176]
    )
    assert mirror_axes[10][1] == pytest.approx(
        [0.9281410528012134, 0.20390316557611038, 0.3114124036917868]
    )
    assert mirror_axes[11][1] == pytest.approx(
        [0.8737016660226733, 0.40421412598335454, -0.2706590828786609]
    )
    assert mirror_axes[12][1] == pytest.approx(
        [0.6280308059506817, -0.0742510540729961, 0.7746380365990211]
    )
    assert mirror_axes[13][1] == pytest.approx(
        [0.08809637894462627, -0.3239897263884152, 0.9419499377417064]
    )
    assert mirror_axes[14][1] == pytest.approx(
        [-0.4855678974971889, -0.4501298612905931, 0.7494043800877273]
    )
    assert coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "Ih"
    assert coords.symmetry_number(point_group) == 60

    data = logfiles["symmetries"]["C60"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([6133.59929944, 6133.81659269, 6134.15217423])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    # assert axes == pytest.approx(
    #     np.array(
    #         [
    #             [1.0, -5.34640329e-12, -1.79924976e-13],
    #             [5.34640329e-12, 1.0, -6.49480469e-14],
    #             [1.79924976e-13, 6.49480469e-14, 1.0],
    #         ]
    #     )
    # )
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 1
    assert len(groups[0]) == 60
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("spheric", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 26
    assert proper_axes[0][0] == 5
    assert proper_axes[1][0] == 3
    assert proper_axes[2][0] == 3
    assert proper_axes[3][0] == 3
    assert proper_axes[4][0] == 3
    assert proper_axes[5][0] == 3
    assert proper_axes[6][0] == 3
    assert proper_axes[7][0] == 3
    assert proper_axes[8][0] == 3
    assert proper_axes[9][0] == 3
    assert proper_axes[10][0] == 3
    assert proper_axes[11][0] == 2
    assert proper_axes[12][0] == 2
    assert proper_axes[13][0] == 2
    assert proper_axes[14][0] == 2
    assert proper_axes[15][0] == 2
    assert proper_axes[16][0] == 2
    assert proper_axes[17][0] == 2
    assert proper_axes[18][0] == 2
    assert proper_axes[19][0] == 2
    assert proper_axes[20][0] == 2
    assert proper_axes[21][0] == 2
    assert proper_axes[22][0] == 2
    assert proper_axes[23][0] == 2
    assert proper_axes[24][0] == 2
    assert proper_axes[25][0] == 2
    assert proper_axes[0][1] == pytest.approx(
        [-0.43323728588212457, -0.29725084310715066, -0.8508509801331712]
    )
    assert proper_axes[1][1] == pytest.approx(
        [0.7730790204640104, 0.5256477128224002, 0.3550257879689668]
    )
    assert proper_axes[2][1] == pytest.approx(
        [0.15158168357811375, 0.8000219937686652, 0.5805065052951898]
    )
    assert proper_axes[3][1] == pytest.approx(
        [0.1450907559831708, 0.8033231148089758, -0.5775990354405356]
    )
    assert proper_axes[4][1] == pytest.approx(
        [-0.19771726502363846, 0.2945110090568182, 0.934971202046302]
    )
    assert proper_axes[5][1] == pytest.approx(
        [-0.20401689339532977, 0.29794469238410615, -0.9325267114080213]
    )
    assert proper_axes[6][1] == pytest.approx(
        [-0.23651459756576956, 0.9716193328591106, 0.004088661482529412]
    )
    assert proper_axes[7][1] == pytest.approx(
        [-0.7688459299583754, -0.5322916638748053, 0.3543183886786462]
    )
    assert proper_axes[8][1] == pytest.approx(
        [-0.8013828638531268, 0.15391944650120618, -0.578008918193589]
    )
    assert proper_axes[9][1] == pytest.approx(
        [-0.8017166254761412, 0.153653414451914, 0.5776167247080171]
    )
    assert proper_axes[10][1] == pytest.approx(
        [-0.8214456016789993, 0.5702740207003627, 0.0038294642535675236]
    )
    assert proper_axes[11][1] == pytest.approx(
        [0.8421379703632436, 0.20303867473786846, -0.49957875798833457]
    )
    assert proper_axes[12][1] == pytest.approx(
        [0.8245933142510103, 0.565725826628097, 0.0003939270120550409]
    )
    assert proper_axes[13][1] == pytest.approx(
        [0.4924099305028999, 0.7126314811226139, -0.4996887355695732]
    )
    assert proper_axes[14][1] == pytest.approx(
        [0.4921145718059886, 0.7124170096771929, 0.5002851712161925]
    )
    assert proper_axes[15][1] == pytest.approx(
        [-0.0002887993033004395, -0.00023006122270818622, 0.9999999318333957]
    )
    assert proper_axes[16][1] == pytest.approx(
        [-0.027850303016893885, 0.5873084880802376, -0.8088838609162458]
    )
    assert proper_axes[17][1] == pytest.approx(
        [-0.028317541371115862, 0.5869360099803028, 0.809137959212826]
    )
    assert proper_axes[18][1] == pytest.approx(
        [-0.04534656462060827, 0.9500408685141216, -0.3088139200716869]
    )
    assert proper_axes[19][1] == pytest.approx(
        [-0.045559902513684546, 0.9499032477386635, 0.3092056196424684]
    )
    assert proper_axes[20][1] == pytest.approx(
        [-0.5374746977379148, 0.23765954826552577, -0.8090975765689595]
    )
    assert proper_axes[21][1] == pytest.approx(
        [-0.5379419182424852, 0.23728720736695694, 0.8088963306985552]
    )
    assert proper_axes[22][1] == pytest.approx(
        [-0.5657701507551198, 0.8245629968765673, 2.6390585100327075e-5]
    )
    assert proper_axes[23][1] == pytest.approx(
        [-0.8417666877910532, -0.20282071665898357, -0.5002925146543927]
    )
    assert proper_axes[24][1] == pytest.approx(
        [-0.8699069667952585, 0.38427951382181996, -0.309178143435041]
    )
    assert proper_axes[25][1] == pytest.approx(
        [-0.8700854547086772, 0.38413725024405565, 0.30885251250286183]
    )
    improper_axes = coords._get_improper_axes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(improper_axes) == 26
    assert improper_axes[0][0] == 10
    assert improper_axes[1][0] == 6
    assert improper_axes[2][0] == 6
    assert improper_axes[3][0] == 6
    assert improper_axes[4][0] == 6
    assert improper_axes[5][0] == 6
    assert improper_axes[6][0] == 6
    assert improper_axes[7][0] == 6
    assert improper_axes[8][0] == 6
    assert improper_axes[9][0] == 6
    assert improper_axes[10][0] == 6
    assert improper_axes[11][0] == 2
    assert improper_axes[12][0] == 2
    assert improper_axes[13][0] == 2
    assert improper_axes[14][0] == 2
    assert improper_axes[15][0] == 2
    assert improper_axes[16][0] == 2
    assert improper_axes[17][0] == 2
    assert improper_axes[18][0] == 2
    assert improper_axes[19][0] == 2
    assert improper_axes[20][0] == 2
    assert improper_axes[21][0] == 2
    assert improper_axes[22][0] == 2
    assert improper_axes[23][0] == 2
    assert improper_axes[24][0] == 2
    assert improper_axes[25][0] == 2
    assert improper_axes[0][1] == proper_axes[0][1]
    assert improper_axes[1][1] == proper_axes[1][1]
    assert improper_axes[2][1] == proper_axes[2][1]
    assert improper_axes[3][1] == proper_axes[3][1]
    assert improper_axes[4][1] == proper_axes[4][1]
    assert improper_axes[5][1] == proper_axes[5][1]
    assert improper_axes[6][1] == proper_axes[6][1]
    assert improper_axes[7][1] == proper_axes[7][1]
    assert improper_axes[8][1] == proper_axes[8][1]
    assert improper_axes[9][1] == proper_axes[9][1]
    assert improper_axes[10][1] == proper_axes[10][1]
    assert improper_axes[11][1] == proper_axes[11][1]
    assert improper_axes[12][1] == proper_axes[12][1]
    assert improper_axes[13][1] == proper_axes[13][1]
    assert improper_axes[14][1] == proper_axes[14][1]
    assert improper_axes[15][1] == proper_axes[15][1]
    assert improper_axes[16][1] == proper_axes[16][1]
    assert improper_axes[17][1] == proper_axes[17][1]
    assert improper_axes[18][1] == proper_axes[18][1]
    assert improper_axes[19][1] == proper_axes[19][1]
    assert improper_axes[20][1] == proper_axes[20][1]
    assert improper_axes[21][1] == proper_axes[21][1]
    assert improper_axes[22][1] == proper_axes[22][1]
    assert improper_axes[23][1] == proper_axes[23][1]
    assert improper_axes[24][1] == proper_axes[24][1]
    assert improper_axes[25][1] == proper_axes[25][1]
    mirror_axes = coords._get_mirror_planes(
        atomcoords, groups, axes, rotor_class, proper_axes
    )
    assert len(mirror_axes) == 15
    assert mirror_axes[0][0] == ""
    assert mirror_axes[1][0] == ""
    assert mirror_axes[2][0] == ""
    assert mirror_axes[3][0] == ""
    assert mirror_axes[4][0] == ""
    assert mirror_axes[5][0] == ""
    assert mirror_axes[6][0] == ""
    assert mirror_axes[7][0] == ""
    assert mirror_axes[8][0] == ""
    assert mirror_axes[9][0] == ""
    assert mirror_axes[10][0] == "v"
    assert mirror_axes[11][0] == "v"
    assert mirror_axes[12][0] == "v"
    assert mirror_axes[13][0] == "v"
    assert mirror_axes[14][0] == "v"
    assert mirror_axes[0][1] == pytest.approx(
        [0.8698296452143535, -0.38448665353327943, 0.3091381593415023]
    )
    assert mirror_axes[1][1] == pytest.approx(
        [0.8417946825484951, 0.2028096944676698, 0.5002498778240818]
    )
    assert mirror_axes[2][1] == pytest.approx(
        [0.8246198528746818, 0.5656871593325626, 0.00036879695507299417]
    )
    assert mirror_axes[3][1] == pytest.approx(
        [0.5380548456059006, -0.23704700989966598, -0.8088916480083338]
    )
    assert mirror_axes[4][1] == pytest.approx(
        [0.5374259789281473, -0.23777671430427477, 0.8090955143292947]
    )
    assert mirror_axes[5][1] == pytest.approx(
        [0.4920627193088096, 0.7124240356381789, 0.5003261673263055]
    )
    assert mirror_axes[6][1] == pytest.approx(
        [-0.00028905577051272966, -0.0002296065494731591, 0.9999999318637947]
    )
    assert mirror_axes[7][1] == pytest.approx(
        [-0.027796383613830928, 0.5871366696835493, -0.809010440087706]
    )
    assert mirror_axes[8][1] == pytest.approx(
        [-0.028438480152180923, 0.5868774909077528, 0.8091761634603769]
    )
    assert mirror_axes[9][1] == pytest.approx(
        [-0.045497246073414814, 0.9499165164627902, 0.30917408098178745]
    )
    assert mirror_axes[10][1] == pytest.approx(
        [0.8700983968716308, -0.3841192000885663, -0.30883850129920337]
    )
    assert mirror_axes[11][1] == pytest.approx(
        [0.8421437863381498, 0.2028798699994556, -0.49963346713470747]
    )
    assert mirror_axes[12][1] == pytest.approx(
        [0.492642249244334, 0.7126481211852062, -0.499435951479946]
    )
    assert mirror_axes[13][1] == pytest.approx(
        [-0.04530773468376337, 0.9500692347171875, -0.30873234106864056]
    )
    assert mirror_axes[14][1] == pytest.approx(
        [-0.565929883314664, 0.8244533722328923, -6.469360643961685e-5]
    )
    assert coords._has_inversion_center(atomcoords, groups)
    point_group = coords.find_point_group(data.atommasses, atomcoords, proper_axes)
    assert point_group == "Ih"
    assert coords.symmetry_number(point_group) == 60


# TODO(schneiderfelipe): allocate the function below to others and delete it.
def test_match_regression_logfiles():
    """Ensure calculated values minimally match regression logfiles."""
    # borane
    data = logfiles["symmetries"]["BH3"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([2.24732283, 2.2473566, 4.4946784])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 2
    assert len(groups[0]) == 1
    assert len(groups[1]) == 3
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric oblate", "regular planar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 4
    assert proper_axes[0][0] == 3
    assert proper_axes[1][0] == 2
    assert proper_axes[2][0] == 2
    assert proper_axes[3][0] == 2
    assert proper_axes[0][1] == pytest.approx([0.0, 0.0, 1.0])
    assert proper_axes[1][1] == pytest.approx(
        [-0.11625570934711942, -0.9932192710921326, 0.00029929151507175205]
    )
    assert proper_axes[2][1] == pytest.approx(
        [-0.8020297868218433, 0.5972839621766384, 0.0002992943104832451]
    )
    assert proper_axes[3][1] == pytest.approx(
        [-0.9182796196256745, -0.3959320419970198, 0.0005985818211691101]
    )

    # chloromethane
    data = logfiles["symmetries"]["chloromethane"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([3.32736206, 39.32328864, 39.32328864])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 3
    assert len(groups[0]) == 1
    assert len(groups[1]) == 1
    assert len(groups[2]) == 3
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("symmetric prolate", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 1
    assert proper_axes[0][0] == 3
    # assert proper_axes[0][1] == pytest.approx([-1.0, 0.0, 0.0])

    # dichloromethane
    data = logfiles["symmetries"]["dichloromethane"]
    moments, axes, atomcoords = coords.inertia(data.atommasses, data.atomcoords)
    assert moments == pytest.approx([16.0098287, 161.95533621, 174.59725576])
    assert axes.T @ axes == pytest.approx(np.eye(3))
    groups = coords._equivalent_atoms(data.atommasses, atomcoords)
    assert len(groups) == 3
    assert len(groups[0]) == 1
    assert len(groups[1]) == 2
    assert len(groups[2]) == 2
    rotor_class = coords._classify_rotor(moments)
    assert rotor_class == ("asymmetric", "nonplanar")
    proper_axes = coords._get_proper_axes(atomcoords, groups, axes, rotor_class)
    assert len(proper_axes) == 1
    assert proper_axes[0][0] == 2
    assert proper_axes[0][1] == pytest.approx([0.0, 1.0, 0.0])


def test_can_rotate_to_principal_axes():
    """Ensure we are able to rotate molecules to their principal axes."""
    data = logfiles["symmetries"]["water"]

    old_moments, old_axes, old_atomcoords = coords.inertia(
        data.atommasses, data.atomcoords, align=False
    )
    moments, axes, atomcoords = coords.inertia(
        data.atommasses, data.atomcoords, align=True
    )
    new_moments, new_axes, new_atomcoords = coords.inertia(
        data.atommasses, atomcoords, align=False
    )

    assert old_moments == pytest.approx(moments)
    assert moments == pytest.approx([0.6768072475, 1.1582103375, 1.835017585])
    assert new_moments == pytest.approx(moments)

    assert old_axes.T @ old_axes == pytest.approx(np.eye(3))
    assert old_axes == pytest.approx(
        np.array(
            [
                [-0.3929704, 0.53914983, -0.74491055],
                [0.20700549, 0.84115548, 0.49960603],
                [0.8959481, 0.04212981, -0.44215618],
            ]
        )
    )
    assert axes == pytest.approx(np.eye(3))
    assert new_axes == pytest.approx(axes)

    assert old_atomcoords == pytest.approx(
        np.array(
            [
                [0.03709541, 0.05787509, 0.00289939],
                [-0.59224959, -0.30239891, 0.65608639],
                [0.00347041, -0.61619591, -0.70210561],
            ]
        )
    )
    assert atomcoords == pytest.approx(
        np.array(
            [
                [7.69264498e-7, 6.88040830e-2, 0.0],
                [7.57957678e-1, -5.46034974e-1, 0.0],
                [-7.57969888e-1, -5.46025070e-1, 0.0],
            ]
        )
    )
    assert new_atomcoords == pytest.approx(atomcoords)
