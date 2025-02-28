"""
This module is intended to match crystal structures against known crystallographic "prototype"
structures.

In this module, the AflowPrototypeMatcher uses the AFLOW LIBRARY OF CRYSTALLOGRAPHIC PROTOTYPES.
If using this particular class, please cite their publication appropriately:

Mehl, M. J., Hicks, D., Toher, C., Levy, O., Hanson, R. M., Hart, G., & Curtarolo, S. (2017).
The AFLOW library of crystallographic prototypes: part 1.
Computational Materials Science, 136, S1-S828.
https://doi.org/10.1016/j.commatsci.2017.01.017
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from monty.serialization import loadfn

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.util.due import Doi, due

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
AFLOW_PROTOTYPE_LIBRARY = loadfn(f"{MODULE_DIR}/aflow_prototypes.json")


@due.dcite(
    Doi("10.1016/j.commatsci.2017.01.017"),
    description="The AFLOW library of crystallographic prototypes: part 1.",
)
class AflowPrototypeMatcher:
    """
    This class will match structures to their crystal prototypes, and will
    attempt to group species together to match structures derived from
    prototypes (e.g. an A_xB_1-x_C from a binary prototype), and will
    give these the names the "-like" suffix.

    This class uses data from the AFLOW LIBRARY OF CRYSTALLOGRAPHIC PROTOTYPES.
    If using this class, please cite their publication appropriately:

    Mehl, M. J., Hicks, D., Toher, C., Levy, O., Hanson, R. M., Hart, G., & Curtarolo, S. (2017).
    The AFLOW library of crystallographic prototypes: part 1.
    Computational Materials Science, 136, S1-S828.
    https://doi.org/10.1016/j.commatsci.2017.01.017
    """

    def __init__(
        self,
        initial_ltol: float = 0.2,
        initial_stol: float = 0.3,
        initial_angle_tol: float = 5,
    ) -> None:
        """
        Tolerances as defined in StructureMatcher. Tolerances will be
        gradually decreased until only a single match is found (if possible).

        Args:
            initial_ltol (float): fractional length tolerance.
            initial_stol (float): site tolerance.
            initial_angle_tol (float): angle tolerance.
        """
        self.initial_ltol = initial_ltol
        self.initial_stol = initial_stol
        self.initial_angle_tol = initial_angle_tol

        # Preprocess AFLOW prototypes
        self._aflow_prototype_library: list[tuple[Structure, dict]] = []
        for dct in AFLOW_PROTOTYPE_LIBRARY:
            structure: Structure = dct["snl"].structure
            reduced_structure = self._preprocess_structure(structure)
            self._aflow_prototype_library.append((reduced_structure, dct))

    @staticmethod
    def _preprocess_structure(structure: Structure) -> Structure:
        return structure.get_reduced_structure(reduction_algo="niggli").get_primitive_structure()

    def _match_prototype(
        self,
        structure_matcher: StructureMatcher,
        reduced_structure: Structure,
    ) -> list[dict]:
        tags = []
        for aflow_reduced_structure, dct in self._aflow_prototype_library:
            # Since both structures are already reduced, we can skip the structure reduction step
            match = structure_matcher.fit_anonymous(
                aflow_reduced_structure, reduced_structure, skip_structure_reduction=True
            )
            if match:
                tags.append(dct)
        return tags

    def _match_single_prototype(self, structure: Structure) -> list[dict]:
        sm = StructureMatcher(
            ltol=self.initial_ltol,
            stol=self.initial_stol,
            angle_tol=self.initial_angle_tol,
            primitive_cell=True,
        )
        reduced_structure = self._preprocess_structure(structure)
        tags = self._match_prototype(sm, reduced_structure)
        while len(tags) > 1:
            sm.ltol *= 0.8
            sm.stol *= 0.8
            sm.angle_tol *= 0.8
            tags = self._match_prototype(sm, reduced_structure)
            if sm.ltol < 0.01:
                break
        return tags

    def get_prototypes(self, structure: Structure) -> list[dict] | None:
        """Get prototype(s) structures for a given input structure. If you use this method in
        your work, please cite the appropriate AFLOW publication:

            Mehl, M. J., Hicks, D., Toher, C., Levy, O., Hanson, R. M., Hart, G., & Curtarolo,
            S. (2017). The AFLOW library of crystallographic prototypes: part 1. Computational
            Materials Science, 136, S1-S828. https://doi.org/10.1016/j.commatsci.2017.01.017

        Args:
            structure (Structure): structure to match

        Returns:
            list[dict] | None: A list of dicts with keys "snl" for the matched prototype and
                "tags", a dict of tags ("mineral", "strukturbericht" and "aflow") of that
                prototype. This should be a list containing just a single entry, but it is
                possible a material can match multiple prototypes.
        """
        tags: list[dict] = self._match_single_prototype(structure)

        return tags or None
