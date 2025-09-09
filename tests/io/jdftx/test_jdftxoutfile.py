from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pymatgen.io.jdftx._output_utils import read_outfile_slices
from pymatgen.io.jdftx.inputs import JDFTXInfile
from pymatgen.io.jdftx.jdftxoutfileslice import JDFTXOutfileSlice
from pymatgen.io.jdftx.outputs import JDFTXOutfile, _jof_atr_from_last_slice

from .inputs_test_utils import aimd_infile_fname
from .outputs_test_utils import (
    etot_etype_outfile_known_simple,
    etot_etype_outfile_path,
    example_aimd_outfile_known,
    example_aimd_outfile_path,
    example_ionmin_outfile_known,
    example_ionmin_outfile_known_simple,
    example_ionmin_outfile_path,
    example_latmin_outfile_known,
    example_latmin_outfile_known_simple,
    example_latmin_outfile_path,
    example_sp_outfile_known,
    example_sp_outfile_known_simple,
    example_sp_outfile_path,
    example_vib_modes_known,
    example_vib_nrg_components,
    example_vib_outfile_path,
    jdftxoutfile_fromfile_matches_known,
    jdftxoutfile_fromfile_matches_known_simple,
    noeigstats_outfile_known_simple,
    noeigstats_outfile_path,
    partial_lattice_init_outfile_known_lattice,
    partial_lattice_init_outfile_path,
    problem2_outfile_known_simple,
    problem2_outfile_path,
)
from .shared_test_utils import assert_same_value

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("filename", "known"),
    [
        (example_sp_outfile_path, example_sp_outfile_known),
        (example_latmin_outfile_path, example_latmin_outfile_known),
        (example_ionmin_outfile_path, example_ionmin_outfile_known),
    ],
)
def test_JDFTXOutfile_fromfile(filename: Path, known: dict):
    jdftxoutfile_fromfile_matches_known(filename, known)


@pytest.mark.parametrize(
    ("filename", "known"),
    [
        (example_sp_outfile_path, example_sp_outfile_known_simple),
        (example_latmin_outfile_path, example_latmin_outfile_known_simple),
        (example_ionmin_outfile_path, example_ionmin_outfile_known_simple),
        (noeigstats_outfile_path, noeigstats_outfile_known_simple),
        (problem2_outfile_path, problem2_outfile_known_simple),
        (etot_etype_outfile_path, etot_etype_outfile_known_simple),
    ],
)
def test_JDFTXOutfile_fromfile_simple(filename: Path, known: dict):
    jdftxoutfile_fromfile_matches_known_simple(filename, known)


@pytest.mark.parametrize(
    ("filename", "latknown"),
    [
        (partial_lattice_init_outfile_path, partial_lattice_init_outfile_known_lattice),
    ],
)
def test_JDFTXOutfile_default_struc_inheritance(filename: Path, latknown: dict):
    jof = JDFTXOutfile.from_file(filename)
    lattice = jof.lattice
    for i in range(3):
        for j in range(3):
            assert pytest.approx(lattice[i][j]) == latknown[f"{i}{j}"]


# Make sure all possible exceptions are caught when none_on_error is True
@pytest.mark.parametrize(("ex_outfile_path"), [(partial_lattice_init_outfile_path)])
def test_none_on_partial(ex_outfile_path: Path):
    texts = read_outfile_slices(str(ex_outfile_path))
    texts0 = texts[:-1]

    slices = [
        JDFTXOutfileSlice._from_out_slice(text, is_bgw=False, none_on_error=False) for i, text in enumerate(texts0)
    ]
    outfile1 = JDFTXOutfile(slices=slices)
    slices.append(None)
    outfile2 = JDFTXOutfile(slices=slices)
    assert isinstance(outfile2, JDFTXOutfile)
    for var in _jof_atr_from_last_slice:
        assert_same_value(
            getattr(outfile1, var),
            getattr(outfile2, var),
        )


@pytest.mark.parametrize(
    ("example_vib_outfile_path", "example_vib_modes_known", "example_vib_nrg_components"),
    [
        (example_vib_outfile_path, example_vib_modes_known, example_vib_nrg_components),
    ],
)
def test_vib_parse(
    example_vib_outfile_path: Path, example_vib_modes_known: list[dict], example_vib_nrg_components: dict
):
    """
    Test that the vibration modes are parsed correctly from the outfile.
    """
    jdftxoutfile = JDFTXOutfile.from_file(example_vib_outfile_path)
    assert_same_value(jdftxoutfile.slices[-1].vibrational_modes, example_vib_modes_known)
    assert_same_value(jdftxoutfile.slices[-1].vibrational_energy_components, example_vib_nrg_components)


@pytest.mark.parametrize(
    ("aimd_outfile_path", "aimd_outfile_known"),
    [
        (example_aimd_outfile_path, example_aimd_outfile_known),
    ],
)
def test_aimd_parse(aimd_outfile_path: Path, aimd_outfile_known: dict):
    """
    Test that the AIMD properties are parsed correctly from the outfile.
    """
    jdftxoutfile = JDFTXOutfile.from_file(aimd_outfile_path)
    assert jdftxoutfile.slices[-1].is_md
    for key, val in aimd_outfile_known.items():
        assert hasattr(jdftxoutfile.slices[-1].jstrucs, key)
        assert_same_value(getattr(jdftxoutfile.slices[-1].jstrucs, key), val)
        assert hasattr(jdftxoutfile.slices[-1], key)
        assert_same_value(getattr(jdftxoutfile.slices[-1], key), val)
        assert hasattr(jdftxoutfile, key)
        assert_same_value(getattr(jdftxoutfile, key), val)


@pytest.mark.parametrize(
    ("outfile_path", "infile_path", "outfile_known"),
    [
        (example_aimd_outfile_path, aimd_infile_fname, example_aimd_outfile_known),
    ],
)
def test_outfile_to_infile(outfile_path: Path, infile_path: Path, outfile_known: dict):
    """
    Test that the JDFTXOutfileSlice.to_infile() method works correctly.
    """
    jdftxoutfile: JDFTXOutfile = JDFTXOutfile.from_file(outfile_path)
    JDFTXInfile.from_file(infile_path)
    new_infile: JDFTXInfile = jdftxoutfile.to_jdftxinfile()
    old_struc = jdftxoutfile.structure
    new_struc = new_infile.to_pmg_structure(new_infile, fill_site_properties=True)
    old_coords = old_struc.cart_coords
    new_coords = new_struc.cart_coords
    assert_same_value(old_coords, new_coords)
    old_velocities = old_struc.site_properties.get("velocities", None)
    new_velocities = new_struc.site_properties.get("velocities", None)
    [ov / nv for ov, nv in zip(old_velocities, new_velocities, strict=False)]
    assert_same_value(old_velocities, new_velocities)
    old_thermostat_velocity = jdftxoutfile.slices[-1].jstrucs.thermostat_velocity
    new_thermostat_velocity = np.array([new_infile["thermostat-velocity"][f"v{i + 1}"] for i in range(3)])
    assert_same_value(old_thermostat_velocity, new_thermostat_velocity)

    # TODO: Do the old TODO related to filling out default input tags so the robust comparison below
    # can be used instead of the above.
    # assert new_infile.is_comparable_to(jdftxinfile, exclude_tags=["ion", "thermostat-velocity", "dump"])
