r"""
This module contains classes useful for analyzing ferroelectric candidates.
The Polarization class can recover the spontaneous polarization using
multiple calculations along a nonpolar to polar ferroelectric distortion.
The EnergyTrend class is useful for assessing the trend in energy across
the distortion.

See Nicola Spaldin's "A beginner's guide to the modern theory of polarization"
(https://arxiv.org/abs/1202.1831) for an introduction to crystal polarization.

VASP reports dipole moment values (used to derive polarization) along Cartesian
directions (see pead.F around line 970 in the VASP source to confirm this).
However, it is most convenient to perform the adjustments necessary to recover
a same branch polarization by expressing the polarization along lattice directions.
For this reason, calc_ionic calculates ionic contributions to the polarization
along lattice directions. We provide the means to convert Cartesian direction
polarizations to lattice direction polarizations in the Polarization class.

We recommend using our calc_ionic function for calculating the ionic
polarization rather than the values from OUTCAR. We find that the ionic
dipole moment reported in OUTCAR differ from the naive calculation of
\\sum_i Z_i r_i where i is the index of the atom, Z_i is the ZVAL from the
pseudopotential file, and r is the distance in Angstroms along the lattice vectors.
Note, this difference is not simply due to VASP using Cartesian directions and
calc_ionic using lattice direction but rather how the ionic polarization is
computed. Compare calc_ionic to VASP SUBROUTINE POINT_CHARGE_DIPOL in dipol.F in
the VASP source to see the differences. We are able to recover a smooth same
branch polarization more frequently using the naive calculation in calc_ionic
than using the ionic dipole moment reported in the OUTCAR.

Some definitions of terms used in the comments below:

A polar structure belongs to a polar space group. A polar space group has a
one of the 10 polar point group:
        (1, 2, m, mm2, 4, 4mm, 3, 3m, 6, 6m)

Being nonpolar is not equivalent to being centrosymmetric (having inversion
symmetry). For example, any space group with point group 222 is nonpolar but
not centrosymmetric.

By symmetry the polarization of a nonpolar material modulo the quantum
of polarization can only be zero or 1/2. We use a nonpolar structure to help
determine the spontaneous polarization because it serves as a reference point.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import UnivariateSpline

from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self

    from pymatgen.core.sites import PeriodicSite


__author__ = "Tess Smidt"
__copyright__ = "Copyright 2017, The Materials Project"
__version__ = "1.0"
__email__ = "tsmidt@berkeley.edu"
__status__ = "Development"
__date__ = "April 15, 2017"


def zval_dict_from_potcar(potcar) -> dict[str, float]:
    """Create zval_dictionary for calculating the ionic polarization from
    Potcar object.

    potcar: Potcar object
    """
    return {p.element: p.ZVAL for p in potcar}


def calc_ionic(site: PeriodicSite, structure: Structure, zval: float) -> np.ndarray:
    """
    Calculate the ionic dipole moment using ZVAL from pseudopotential.

    site: PeriodicSite
    structure: Structure
    zval: Charge value for ion (ZVAL for VASP pseudopotential)

    Returns polarization in electron Angstroms.
    """
    norms = structure.lattice.lengths
    return np.multiply(norms, -site.frac_coords * zval)


def get_total_ionic_dipole(structure, zval_dict):
    """Get the total ionic dipole moment for a structure.

    structure: pymatgen Structure
    zval_dict: specie, zval dictionary pairs
    center (np.array with shape [3,1]) : dipole center used by VASP
    tiny (float) : tolerance for determining boundary of calculation.
    """
    tot_ionic = []
    for site in structure:
        zval = zval_dict[str(site.specie)]
        tot_ionic.append(calc_ionic(site, structure, zval))
    return np.sum(tot_ionic, axis=0)


def get_nearest_site(
    struct: Structure,
    coords: Sequence[float],
    site: PeriodicSite,
    r: float | None = None,
):
    """
    Given coords and a site, find closet site to coords.

    Args:
        coords (3x1 array): Cartesian coords of center of sphere
        site: site to find closest to coords
        r (float): radius of sphere. Defaults to diagonal of unit cell

    Returns:
        Closest site and distance.
    """
    index = struct.index(site)
    radius = r or np.linalg.norm(np.sum(struct.lattice.matrix, axis=0))
    ns = struct.get_sites_in_sphere(coords, radius, include_index=True)
    # Get sites with identical index to site
    ns = [n for n in ns if n[2] == index]
    # Sort by distance to coords
    ns.sort(key=lambda x: x[1])
    # Return PeriodicSite and distance of closest image
    return ns[0][:2]


class Polarization:
    """Recover the same branch polarization for a set of polarization
    calculations along the nonpolar - polar distortion path of a ferroelectric.

    p_elecs, p_ions, and structures lists should be given in order
    of nonpolar to polar! For example, the structures returned from:
        nonpolar.interpolate(polar,interpolate_lattices=True)
    if nonpolar is the nonpolar Structure and polar is the polar structure.

    It is assumed that the electronic and ionic dipole moment values are given in
    electron Angstroms along the three lattice directions (a,b,c).
    """

    def __init__(
        self,
        p_elecs,
        p_ions,
        structures: Sequence[Structure],
        p_elecs_in_cartesian=True,
        p_ions_in_cartesian=False,
    ):
        """
        p_elecs (np.ndarray): electronic contribution to the polarization with shape [N, 3]
        p_ions (np.ndarray): ionic contribution to the polarization with shape [N, 3]
        p_elecs_in_cartesian: whether p_elecs is along Cartesian directions (rather than lattice directions).
            Default is True because that is the convention for VASP.
        p_ions_in_cartesian: whether p_ions is along Cartesian directions (rather than lattice directions).
            Default is False because calc_ionic (which we recommend using for calculating the ionic
            contribution to the polarization) uses lattice directions.
        """
        if len(p_elecs) != len(p_ions) or len(p_elecs) != len(structures):
            raise ValueError("The number of electronic polarization and ionic polarization values must be equal.")
        if p_elecs_in_cartesian:
            p_elecs = [
                struct.lattice.get_vector_along_lattice_directions(p_elecs[idx])
                for idx, struct in enumerate(structures)
            ]
        if p_ions_in_cartesian:
            p_ions = [
                struct.lattice.get_vector_along_lattice_directions(p_ions[idx]) for idx, struct in enumerate(structures)
            ]
        self.p_elecs = np.array(p_elecs)
        self.p_ions = np.array(p_ions)
        self.structures = structures

    @classmethod
    def from_outcars_and_structures(cls, outcars, structures, calc_ionic_from_zval=False) -> Self:
        """
        Create Polarization object from list of Outcars and Structures in order
        of nonpolar to polar.

        Note, we recommend calculating the ionic dipole moment using calc_ionic
        than using the values in Outcar (see module comments). To do this set
        calc_ionic_from_zval = True
        """
        p_elecs = []
        p_ions = []

        for idx, outcar in enumerate(outcars):
            p_elecs.append(outcar.p_elec)
            if calc_ionic_from_zval:
                p_ions.append(get_total_ionic_dipole(structures[idx], outcar.zval_dict))
            else:
                p_ions.append(outcar.p_ion)
        return cls(p_elecs, p_ions, structures)

    def get_pelecs_and_pions(self, convert_to_muC_per_cm2=False):
        """Get the electronic and ionic dipole moments / polarizations.

        convert_to_muC_per_cm2: Convert from electron * Angstroms to microCoulomb
            per centimeter**2
        """
        if not convert_to_muC_per_cm2:
            return self.p_elecs, self.p_ions

        if convert_to_muC_per_cm2:
            p_elecs = self.p_elecs.T
            p_ions = self.p_ions.T

            volumes = [struct.volume for struct in self.structures]
            e_to_muC = -1.6021766e-13
            cm2_to_A2 = 1e16
            units = 1 / np.array(volumes)
            units *= e_to_muC * cm2_to_A2

            p_elecs = np.matmul(units, p_elecs)
            p_ions = np.matmul(units, p_ions)

            p_elecs, p_ions = p_elecs.T, p_ions.T

            return p_elecs, p_ions

        return None

    def get_same_branch_polarization_data(self, convert_to_muC_per_cm2=True, all_in_polar=True):
        r"""Get same branch dipole moment (convert_to_muC_per_cm2=False)
        or polarization for given polarization data (convert_to_muC_per_cm2=True).

        Polarization is a lattice vector, meaning it is only defined modulo the
        quantum of polarization:

            P = P_0 + \\sum_i \\frac{n_i e R_i}{\\Omega}

        where n_i is an integer, e is the charge of the electron in microCoulombs,
        R_i is a lattice vector, and \\Omega is the unit cell volume in cm**3
        (giving polarization units of microCoulomb per centimeter**2).

        The quantum of the dipole moment in electron Angstroms (as given by VASP) is:

            \\sum_i n_i e R_i

        where e, the electron charge, is 1 and R_i is a lattice vector, and n_i is an integer.

        Given N polarization calculations in order from nonpolar to polar, this algorithm
        minimizes the distance between adjacent polarization images. To do this, it
        constructs a polarization lattice for each polarization calculation using the
        pymatgen.core.structure class and calls the get_nearest_site method to find the
        image of a given polarization lattice vector that is closest to the previous polarization
        lattice vector image.

        Note, using convert_to_muC_per_cm2=True and all_in_polar=True calculates the "proper
        polarization" (meaning the change in polarization does not depend on the choice of
        polarization branch) while convert_to_muC_per_cm2=True and all_in_polar=False calculates
        the "improper polarization" (meaning the change in polarization does depend on the choice
        of branch). As one might guess from the names. We recommend calculating the "proper
        polarization".

        convert_to_muC_per_cm2: convert polarization from electron * Angstroms to
            microCoulomb per centimeter**2
        all_in_polar: convert polarization to be in polar (final structure) polarization lattice
        """
        p_elec, p_ion = self.get_pelecs_and_pions()
        p_tot = p_elec + p_ion
        p_tot = np.array(p_tot)

        lattices = [struct.lattice for struct in self.structures]
        volumes = np.array([latt.volume for latt in lattices])

        n_elecs = len(p_elec)

        e_to_muC = -1.6021766e-13
        cm2_to_A2 = 1e16
        units = 1 / np.array(volumes)
        units *= e_to_muC * cm2_to_A2

        # convert polarizations and lattice lengths prior to adjustment
        if convert_to_muC_per_cm2 and not all_in_polar:
            # Convert the total polarization
            p_tot = np.multiply(units.T[:, None], p_tot)
            # adjust lattices
            for idx in range(n_elecs):
                lattice = lattices[idx]
                lattices[idx] = Lattice.from_parameters(
                    *(np.array(lattice.lengths) * units.ravel()[idx]), *lattice.angles
                )
        #  convert polarizations to polar lattice
        elif convert_to_muC_per_cm2 and all_in_polar:
            abc = [lattice.abc for lattice in lattices]
            abc = np.array(abc)  # [N, 3]
            p_tot /= abc  # e * Angstroms to e
            p_tot *= abc[-1] / volumes[-1] * e_to_muC * cm2_to_A2  # to muC / cm^2
            for idx in range(n_elecs):
                lattice = lattices[-1]  # Use polar lattice
                # Use polar units (volume)
                lattices[idx] = Lattice.from_parameters(
                    *(np.array(lattice.lengths) * units.ravel()[-1]), *lattice.angles
                )

        d_structs = []
        sites = []
        for idx in range(n_elecs):
            lattice = lattices[idx]
            frac_coord = np.divide(np.array([p_tot[idx]]), np.array(lattice.lengths))
            struct = Structure(lattice, ["C"], [np.array(frac_coord).ravel()])
            d_structs.append(struct)
            site = struct[0]
            # Adjust nonpolar polarization to be closest to zero.
            # This is compatible with both a polarization of zero or a half quantum.
            prev_site = [0, 0, 0] if idx == 0 else sites[-1].coords
            new_site = get_nearest_site(struct, prev_site, site)
            sites.append(new_site[0])

        adjust_pol = []
        for site, struct in zip(sites, d_structs, strict=True):
            adjust_pol.append(np.multiply(site.frac_coords, np.array(struct.lattice.lengths)).ravel())
        return np.array(adjust_pol)

    def get_lattice_quanta(self, convert_to_muC_per_cm2=True, all_in_polar=True):
        """Get the dipole / polarization quanta along a, b, and c for
        all structures.
        """
        lattices = [struct.lattice for struct in self.structures]
        volumes = np.array([struct.volume for struct in self.structures])

        n_structs = len(self.structures)

        e_to_muC = -1.6021766e-13
        cm2_to_A2 = 1e16
        units = 1 / np.array(volumes)
        units *= e_to_muC * cm2_to_A2

        # convert polarizations and lattice lengths prior to adjustment
        if convert_to_muC_per_cm2 and not all_in_polar:
            # adjust lattices
            for idx in range(n_structs):
                lattice = lattices[idx]
                lattices[idx] = Lattice.from_parameters(
                    *(np.array(lattice.lengths) * units.ravel()[idx]), *lattice.angles
                )
        elif convert_to_muC_per_cm2 and all_in_polar:
            for idx in range(n_structs):
                lattice = lattices[-1]
                lattices[idx] = Lattice.from_parameters(
                    *(np.array(lattice.lengths) * units.ravel()[-1]), *lattice.angles
                )

        return np.array([np.array(latt.lengths) for latt in lattices])

    def get_polarization_change(self, convert_to_muC_per_cm2=True, all_in_polar=True):
        """Get difference between nonpolar and polar same branch polarization."""
        tot = self.get_same_branch_polarization_data(
            convert_to_muC_per_cm2=convert_to_muC_per_cm2, all_in_polar=all_in_polar
        )
        # reshape to preserve backwards compatibility due to changes
        # when switching from np.matrix to np.array
        return (tot[-1] - tot[0]).reshape((1, 3))

    def get_polarization_change_norm(self, convert_to_muC_per_cm2=True, all_in_polar=True):
        """Get magnitude of difference between nonpolar and polar same branch
        polarization.
        """
        polar = self.structures[-1]
        a, b, c = polar.lattice.matrix
        a, b, c = a / np.linalg.norm(a), b / np.linalg.norm(b), c / np.linalg.norm(c)
        P = self.get_polarization_change(
            convert_to_muC_per_cm2=convert_to_muC_per_cm2, all_in_polar=all_in_polar
        ).ravel()
        return np.linalg.norm(a * P[0] + b * P[1] + c * P[2])

    def same_branch_splines(self, convert_to_muC_per_cm2=True, all_in_polar=True):
        """Fit splines to same branch polarization. This is used to assess any jumps
        in the same branch polarization.
        """
        tot = self.get_same_branch_polarization_data(
            convert_to_muC_per_cm2=convert_to_muC_per_cm2, all_in_polar=all_in_polar
        )
        L = tot.shape[0]
        try:
            sp_a = UnivariateSpline(range(L), tot[:, 0].ravel())
        except Exception:
            sp_a = None
        try:
            sp_b = UnivariateSpline(range(L), tot[:, 1].ravel())
        except Exception:
            sp_b = None
        try:
            sp_c = UnivariateSpline(range(L), tot[:, 2].ravel())
        except Exception:
            sp_c = None
        return sp_a, sp_b, sp_c

    def max_spline_jumps(self, convert_to_muC_per_cm2=True, all_in_polar=True):
        """Get maximum difference between spline and same branch polarization data."""
        tot = self.get_same_branch_polarization_data(
            convert_to_muC_per_cm2=convert_to_muC_per_cm2, all_in_polar=all_in_polar
        )
        sps = self.same_branch_splines(convert_to_muC_per_cm2=convert_to_muC_per_cm2, all_in_polar=all_in_polar)
        max_jumps = [None, None, None]
        for idx, sp in enumerate(sps):
            if sp is not None:
                max_jumps[idx] = max(tot[:, idx].ravel() - sp(range(len(tot[:, idx].ravel()))))
        return max_jumps

    def smoothness(self, convert_to_muC_per_cm2=True, all_in_polar=True):
        """Get rms average difference between spline and same branch polarization data."""
        tot = self.get_same_branch_polarization_data(
            convert_to_muC_per_cm2=convert_to_muC_per_cm2, all_in_polar=all_in_polar
        )
        L = tot.shape[0]
        try:
            sp = self.same_branch_splines(convert_to_muC_per_cm2=convert_to_muC_per_cm2, all_in_polar=all_in_polar)
        except Exception:
            print("Something went wrong.")
            return None
        sp_latt = [sp[i](range(L)) for i in range(3)]
        diff = [sp_latt[i] - tot[:, i].ravel() for i in range(3)]
        return [np.sqrt(np.sum(np.square(diff[i])) / L) for i in range(3)]


class EnergyTrend:
    """Analyze the trend in energy across a distortion path."""

    def __init__(self, energies):
        """
        Args:
            energies: Energies.
        """
        self.energies = energies

    def spline(self):
        """Fit spline to energy trend data."""
        return UnivariateSpline(range(len(self.energies)), self.energies, k=4)

    def smoothness(self):
        """Get rms average difference between spline and energy trend."""
        energies = self.energies
        try:
            sp = self.spline()
        except Exception:
            print("Energy spline failed.")
            return None
        spline_energies = sp(range(len(energies)))
        diff = spline_energies - energies
        return np.sqrt(np.sum(np.square(diff)) / len(energies))

    def max_spline_jump(self):
        """Get maximum difference between spline and energy trend."""
        sp = self.spline()
        return max(self.energies - sp(range(len(self.energies))))

    def endpoints_minima(self, slope_cutoff=5e-3):
        """Test if spline endpoints are at minima for a given slope cutoff."""
        energies = self.energies
        try:
            sp = self.spline()
        except Exception:
            print("Energy spline failed.")
            return None
        der = sp.derivative()
        der_energies = der(range(len(energies)))
        return {
            "polar": abs(der_energies[-1]) <= slope_cutoff,
            "nonpolar": abs(der_energies[0]) <= slope_cutoff,
        }
