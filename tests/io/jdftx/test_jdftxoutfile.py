from __future__ import annotations

from itertools import combinations
from time import time
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pymatgen.core.units import Ha_to_eV
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
    jdftxoutfile_matches_known,
    jdftxoutfile_matches_known_simple,
    noeigstats_outfile_known_simple,
    noeigstats_outfile_path,
    partial_lattice_init_outfile_known_lattice,
    partial_lattice_init_outfile_path,
    problem2_outfile_known_simple,
    problem2_outfile_path,
    problem3_outfile_path,
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
    ("filename", "known", "known_simple"),
    [
        (example_sp_outfile_path, example_sp_outfile_known, example_sp_outfile_known_simple),
        (example_latmin_outfile_path, example_latmin_outfile_known, example_latmin_outfile_known_simple),
        (example_ionmin_outfile_path, example_ionmin_outfile_known, example_ionmin_outfile_known_simple),
    ],
)
def test_JDFTXOutfile_skimming(filename: Path, known: dict, known_simple: dict):
    # Parse multiple times to get average time
    n_iters = 3
    # Time parsing the out file in regular mode
    time_fulls: list[float] = []
    for _ in range(n_iters):
        start1 = time()
        jof_full = JDFTXOutfile.from_file(filename)
        time_full = time() - start1
        time_fulls.append(time_full)
    time_full = sum(time_fulls) / n_iters
    # Count how minimum fewer objects can be read at each depth level
    nless_outslices = len(jof_full.slices) - 1
    nless_last_geom = len(jof_full.slices[-1].jstrucs.slices) - 1
    nless_last_elec = len(jof_full.slices[-1].jstrucs.slices[-1].elecmindata.slices) - 1
    skim_level_options = ["outfile", "geom", "elec"]
    # Don't run outfile depth skimming test if only one slice
    if not nless_outslices > 0:
        skim_level_options.remove("outfile")
    # Generate all combinations of skim level depths to test
    skim_levelss: list[tuple[str, ...]] = []
    for r in range(1, len(skim_level_options) + 1):
        skim_levelss.extend(combinations(skim_level_options, r))
    levels_time_tested = 0
    time_save_logs = []

    for skim_levels in skim_levelss:
        # Time parsing the out file in skimmed mode
        skim_times = []
        for _ in range(n_iters):
            skim_start = time()
            jof_skim = JDFTXOutfile.from_file(filename, skim_levels=skim_levels)
            skim_time = time() - skim_start
            skim_times.append(skim_time)
        skim_time = sum(skim_times) / n_iters
        # Check that the skimmed version matches the full version where it should
        jdftxoutfile_matches_known(jof_skim, known)
        jdftxoutfile_matches_known_simple(jof_skim, known_simple)
        # Make sure skim levels are correctly applied
        if "outfile" in skim_levels:
            assert len(jof_skim.slices) == 1
        if "geom" in skim_levels:
            for oslice in jof_skim.slices:
                assert len(oslice.jstrucs.slices) == 1
        if "elec" in skim_levels:
            for oslice in jof_skim.slices:
                for gslice in oslice.jstrucs.slices:
                    assert len(gslice.elecmindata.slices) == 1
        # Make sure less objects are being read before testing speedup
        less_outfile_slices_read = nless_outslices if "outfile" in skim_levels else 0
        less_geom_slices_read = nless_last_geom if "geom" in skim_levels else 0
        less_elec_slices_read = nless_last_elec if "elec" in skim_levels else 0
        less_sum = less_outfile_slices_read + less_geom_slices_read + less_elec_slices_read
        if less_sum > 0:
            time_save_info = [
                filename.name,
                skim_levels,
                time_full,
                skim_time,
                less_outfile_slices_read,
                less_geom_slices_read,
                less_elec_slices_read,
            ]
            assert skim_time < time_full, time_save_info
            levels_time_tested += 1
            time_save_logs.append(time_save_info)
            time_save_info = [
                filename.name,
                skim_levels,
                time_full,
                skim_time,
                less_outfile_slices_read,
                less_geom_slices_read,
                less_elec_slices_read,
            ]
            assert skim_time < time_full
            levels_time_tested += 1
            time_save_logs.append(time_save_info)
    assert levels_time_tested > 0, "No skim levels tested - likely an error in test code."
    print(time_save_logs)


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


def test_etot_bug_handling():
    outfile = JDFTXOutfile.from_file(problem3_outfile_path)
    assert len(outfile.slices) == 1
    assert len(outfile.slices[-1].jstrucs.slices) == 7
    # The first ionic step in this outfile has the etot bug
    assert outfile.slices[-1].jstrucs.slices[0].etype == "F"
    assert "F" not in outfile.slices[-1].jstrucs.slices[0].ecomponents
    assert_same_value(outfile.slices[-1].jstrucs.slices[0].e, -3.018425965201787 * Ha_to_eV)
