from __future__ import annotations

from typing import Any

import pytest
from pytest import approx

from pymatgen.core.units import Ha_to_eV
from pymatgen.io.jdftx.jelstep import JElStep, JElSteps

from .conftest import assert_slices_1layer_attribute_error, assert_slices_2layer_attribute_error

ex_fillings_line1 = "FillingsUpdate:  mu: +0.714406772  \
    nElectrons: 64.000000  magneticMoment: [ Abs: 0.00578  Tot: -0.00141 ]"
ex_fillings_line1_known = {
    "mu": 0.714406772 * Ha_to_eV,
    "nelectrons": 64.0,
    "abs_magneticmoment": 0.00578,
    "tot_magneticmoment": -0.00141,
}

ex_fillings_line2 = "FillingsUpdate:  mu: +0.814406772  \
    nElectrons: 60.000000  magneticMoment: [ Abs: 0.0578  Tot: -0.0141 ]"
ex_fillings_line2_known = {
    "mu": 0.814406772 * Ha_to_eV,
    "nelectrons": 60.0,
    "abs_magneticmoment": 0.0578,
    "tot_magneticmoment": -0.0141,
}

ex_subspace_line1 = "SubspaceRotationAdjust: set factor to 0.229"
ex_subspace_line1_known = {"subspacerotationadjust": 0.229}

ex_subspace_line2 = "SubspaceRotationAdjust: set factor to 0.329"
ex_subspace_line2_known = {"subspacerotationadjust": 0.329}

ex_iter_line1 = "ElecMinimize: Iter:   6  F: -246.531038317370076\
        |grad|_K:  6.157e-08  alpha:  5.534e-01  linmin: -4.478e-06\
              t[s]:    248.68"
ex_iter_line1_known = {
    "niter": 6,
    "e": -246.531038317370076 * Ha_to_eV,
    "grad_k": 6.157e-08,
    "alpha": 5.534e-01,
    "linmin": -4.478e-06,
    "t_s": 248.68,
}

ex_iter_line2 = "ElecMinimize: Iter:   7  F: -240.531038317370076\
        |grad|_K:  6.157e-07  alpha:  5.534e-02  linmin: -5.478e-06\
                t[s]:    48.68"
ex_iter_line2_known = {
    "niter": 7,
    "e": -240.531038317370076 * Ha_to_eV,
    "grad_k": 6.157e-07,
    "alpha": 5.534e-02,
    "linmin": -5.478e-06,
    "t_s": 48.68,
}


ex_lines1 = [ex_fillings_line1, ex_subspace_line1, ex_iter_line1]
ex_lines2 = [ex_fillings_line2, ex_subspace_line2, ex_iter_line2]
ex_known1 = {}
for known1 in [ex_fillings_line1_known, ex_iter_line1_known, ex_subspace_line1_known]:
    ex_known1.update(known1)
ex_known2 = {}
for known2 in [ex_fillings_line2_known, ex_iter_line2_known, ex_subspace_line2_known]:
    ex_known2.update(known2)


def is_right_known(val: Any, ex_known_val: Any):
    if isinstance(val, type(ex_known_val)):
        result = ex_known_val == approx(val) if isinstance(val, float) else ex_known_val == val
    else:
        result = False
    return result


@pytest.mark.parametrize(
    ("exfill_line", "exfill_known", "exiter_line", "exiter_known", "exsubspace_line", "exsubspace_known"),
    [
        (
            ex_fillings_line1,
            ex_fillings_line1_known,
            ex_iter_line1,
            ex_iter_line1_known,
            ex_subspace_line1,
            ex_subspace_line1_known,
        )
    ],
)
def test_JElStep_known(
    exfill_line: str,
    exfill_known: dict[str, float],
    exiter_line: str,
    exiter_known: dict[str, float],
    exsubspace_line: str,
    exsubspace_known: dict[str, float],
    etype: str = "F",
    eitertype="ElecMinimize",
):
    ex_lines_collect = [exiter_line, exfill_line, exsubspace_line, ""]  # Empty line added for coverage
    jei = JElStep.from_lines_collect(ex_lines_collect, eitertype, etype)
    ex_known = {}
    for dictlike in [exfill_known, exiter_known, exsubspace_known]:
        ex_known.update(dictlike)
    for var in [
        "mu",
        "nelectrons",
        "abs_magneticmoment",
        "tot_magneticmoment",
        "niter",
        "e",
        "grad_k",
        "alpha",
        "linmin",
        "t_s",
        "subspacerotationadjust",
    ]:
        val = getattr(jei, var)
        assert is_right_known(val, ex_known[var])


@pytest.mark.parametrize(("ex_lines", "ex_knowns"), [([ex_lines1, ex_lines2], [ex_known1, ex_known2])])
def test_JElSteps_known(
    ex_lines: list[list[str]],
    ex_knowns: list[dict],
    etype: str = "F",
    eitertype="ElecMinimize",
):
    text_slice = []
    for exl in ex_lines:
        text_slice += exl
    jeis = JElSteps.from_text_slice(text_slice, iter_type=eitertype, etype=etype)
    for var in [
        "mu",
        "nelectrons",
        "abs_magneticmoment",
        "tot_magneticmoment",
        "niter",
        "e",
        "grad_k",
        "alpha",
        "linmin",
        "t_s",
        "subspacerotationadjust",
    ]:
        val = getattr(jeis, var)
        assert is_right_known(val, ex_knowns[-1][var])
        for i in range(len(ex_lines)):
            val2 = getattr(jeis[i], var)
            assert is_right_known(val2, ex_knowns[i][var])


ex_text_slice = [ex_fillings_line1, ex_subspace_line1, ex_iter_line1]


@pytest.mark.parametrize(
    ("text_slice", "varname"),
    [
        (ex_text_slice, "niter"),
        (ex_text_slice, "grad_k"),
        (ex_text_slice, "alpha"),
        (ex_text_slice, "linmin"),
        (ex_text_slice, "abs_magneticmoment"),
        (ex_text_slice, "tot_magneticmoment"),
    ],
)
def test_JElSteps_has_1layer_slice_freakout(text_slice: list[str], varname: str):
    assert_slices_1layer_attribute_error(JElSteps.from_text_slice, text_slice, varname, "slices")


@pytest.mark.parametrize(
    ("text_slice", "varname"),
    [
        (ex_text_slice, "e"),
        (ex_text_slice, "t_s"),
        (ex_text_slice, "mu"),
        (ex_text_slice, "nelectrons"),
        (ex_text_slice, "subspacerotationadjust"),
    ],
)
def test_JElSteps_has_2layer_slice_freakout(text_slice: list[str], varname: str):
    assert_slices_2layer_attribute_error(JElSteps.from_text_slice, text_slice, varname, "slices")
