"""JDFTx Outfile Slice Class.

This module defines the JDFTxOutfileSlice class, which is used to read and
process a JDFTx out file.

"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from pymatgen.io.jdftx.jelstep import JElSteps
from pymatgen.core.periodic_table import Element
from pymatgen.core.trajectory import Trajectory
from pymatgen.core.units import Ha_to_eV, ang_to_bohr
from pymatgen.io.jdftx.jminsettings import (
    JMinSettings,
    JMinSettingsElectronic,
    JMinSettingsFluid,
    JMinSettingsIonic,
    JMinSettingsLattice,
)
from pymatgen.io.jdftx.joutstructures import JOutStructures
from pymatgen.io.jdftx.utils import (
    ClassPrintFormatter,
    find_all_key,
    find_first_range_key,
    find_key,
    find_key_first,
    get_pseudo_read_section_bounds,
    key_exists,
)

__author__ = "Ben Rich"


@dataclass
class JDFTXOutfileSlice(ClassPrintFormatter):
    """A class to read and process a JDFTx out file.

    A class to read and process a JDFTx out file.

    Attributes
    ----------
        see JDFTx documentation for tag info and typing
    """

    prefix: str | None = None

    jstrucs: JOutStructures | None = None
    jsettings_fluid: JMinSettings | None = None
    jsettings_electronic: JMinSettings | None = None
    jsettings_lattice: JMinSettings | None = None
    jsettings_ionic: JMinSettings | None = None

    xc_func: str | None = None

    lattice_initial: np.ndarray | None = None
    lattice_final: np.ndarray | None = None
    lattice: np.ndarray | None = None
    a: float | None = None
    b: float | None = None
    c: float | None = None

    fftgrid: list[int] | None = None
    geom_opt: bool | None = None
    geom_opt_type: str | None = None

    # grouping fields related to electronic parameters.
    # Used by the get_electronic_output() method
    _electronic_output: ClassVar[list[str]] = [
        "efermi",
        "egap",
        "emin",
        "emax",
        "homo",
        "lumo",
        "homo_filling",
        "lumo_filling",
        "is_metal",
    ]
    efermi: float | None = None
    egap: float | None = None
    emin: float | None = None
    emax: float | None = None
    homo: float | None = None
    lumo: float | None = None
    homo_filling: float | None = None
    lumo_filling: float | None = None
    is_metal: bool | None = None
    etype: str | None = None

    broadening_type: str | None = None
    broadening: float | None = None
    kgrid: list | None = None
    truncation_type: str | None = None
    truncation_radius: float | None = None
    pwcut: float | None = None
    rhocut: float | None = None

    pp_type: str | None = None
    total_electrons: float | None = None
    semicore_electrons: int | None = None
    valence_electrons: float | None = None
    total_electrons_uncharged: int | None = None
    semicore_electrons_uncharged: int | None = None
    valence_electrons_uncharged: int | None = None
    nbands: int | None = None

    atom_elements: list | None = None
    atom_elements_int: list | None = None
    atom_types: list | None = None
    spintype: str | None = None
    nspin: int | None = None
    nat: int | None = None
    atom_coords_initial: list[list[float]] | None = None
    atom_coords_final: list[list[float]] | None = None
    atom_coords: list[list[float]] | None = None

    has_solvation: bool = False
    fluid: str | None = None
    is_gc: bool | None = None
    is_bgw: bool = False
    has_eigstats: bool = False

    @property
    def t_s(self) -> float | None:
        """Return the total time in seconds for the calculation.

        Return the total time in seconds for the calculation.

        Returns
        -------
        t_s: float
            The total time in seconds for the calculation
        """
        t_s = None
        if self.jstrucs:
            t_s = self.jstrucs.t_s
        return t_s

    @property
    def is_converged(self) -> bool | None:
        """Return True if calculation converged.

        Return True if the electronic and geometric optimization have converged
        (or only the former if a single-point calculation)

        Returns
        -------
        converged: bool
            True if calculation converged
        """
        if self.jstrucs is None:
            return None
        converged = self.jstrucs.elec_converged
        if self.geom_opt:
            converged = converged and self.jstrucs.geom_converged
        return converged

    @property
    def trajectory(self) -> Trajectory:
        """Return pymatgen trajectory object.

        Return pymatgen trajectory object containing intermediate Structure's
        of outfile slice calculation.

        Returns
        -------
        traj: Trajectory
            pymatgen Trajectory object
        """
        constant_lattice = False
        if self.jsettings_lattice is not None:
            if "niterations" in self.jsettings_lattice.params:
                constant_lattice = int(self.jsettings_lattice.params["niterations"]) == 0
            else:
                raise ValueError("Unknown issue due to partial initialization of settings objects.")
        return Trajectory.from_structures(structures=self.jstrucs, constant_lattice=constant_lattice)

    @property
    def electronic_output(self) -> dict:
        """Return a dictionary with all relevant electronic information.

        Return dict with values corresponding to these keys in _electronic_output
        field.
        """
        dct = {}
        for field in self.__dataclass_fields__:
            if field in self._electronic_output:
                value = getattr(self, field)
                dct[field] = value
        return dct

    @property
    def structure(self) -> Structure:
        """Return calculation result as pymatgen Structure.

        Return calculation result as pymatgen Structure.

        Returns
        -------
        structure: Structure
            pymatgen Structure object
        """
        if self.jstrucs is not None:
            return self.jstrucs[-1]
        raise AttributeError("Property structure inaccessible due to empty jstrucs class field")

    ###########################################################################
    # Properties inherited directly from jstrucs
    ###########################################################################

    @property
    def eiter_type(self) -> str:
        """
        Return eiter_type from most recent JOutStructure.

        Return eiter_type from most recent JOutStructure.
        """
        if self.jstrucs is not None:
            return self.jstrucs.eiter_type
        raise AttributeError("Property eiter_type inaccessible due to empty jstrucs class field")

    @property
    def elecmindata(self) -> JElSteps:
        """
        Return elecmindata from most recent JOutStructure.

        Return elecmindata from most recent JOutStructure.
        """
        if self.jstrucs is not None:
            return self.jstrucs.elecmindata
        raise AttributeError("Property elecmindata inaccessible due to empty jstrucs class field")

    @property
    def stress(self) -> np.ndarray:
        """
        Return stress from most recent JOutStructure.

        Return stress from most recent JOutStructure.
        """
        if self.jstrucs is not None:
            return self.jstrucs.stress
        raise AttributeError("Property stress inaccessible due to empty jstrucs class field")

    @property
    def strain(self) -> np.ndarray:
        """
        Return strain from most recent JOutStructure.

        Return strain from most recent JOutStructure.
        """
        if self.jstrucs is not None:
            return self.jstrucs.strain
        raise AttributeError("Property strain inaccessible due to empty jstrucs class field")

    @property
    def niter(self) -> int:
        """
        Return (geometric) niter from most recent JOutStructure.

        Return (geometric) niter from most recent JOutStructure.
        """
        if self.jstrucs is not None:
            return self.jstrucs.niter
        raise AttributeError("Property niter inaccessible due to empty jstrucs class field")

    @property
    def e(self) -> float:
        """
        Return E from most recent JOutStructure.

        Return E from most recent JOutStructure.
        """
        if self.jstrucs is not None:
            return self.jstrucs.e
        raise AttributeError("Property e inaccessible due to empty jstrucs class field")

    @property
    def grad_k(self) -> float:
        """
        Return (geometric) grad_k from most recent JOutStructure.

        Return (geometric) grad_k from most recent JOutStructure.
        """
        if self.jstrucs is not None:
            return self.jstrucs.grad_k
        raise AttributeError("Property grad_k inaccessible due to empty jstrucs class field")

    @property
    def alpha(self) -> float:
        """
        Return (geometric) alpha from most recent JOutStructure.

        Return (geometric) alpha from most recent JOutStructure.
        """
        if self.jstrucs is not None:
            return self.jstrucs.alpha
        raise AttributeError("Property alpha inaccessible due to empty jstrucs class field")

    @property
    def linmin(self) -> float:
        """
        Return (geometric) linmin from most recent JOutStructure.

        Return (geometric) linmin from most recent JOutStructure.
        """
        if self.jstrucs is not None:
            return self.jstrucs.linmin
        raise AttributeError("Property linmin inaccessible due to empty jstrucs class field")

    @property
    def nelectrons(self) -> float:
        """
        Return nelectrons from most recent JOutStructure.

        Return nelectrons from most recent JOutStructure.
        """
        if self.jstrucs is not None:
            return self.jstrucs.nelectrons
        raise AttributeError("Property nelectrons inaccessible due to empty jstrucs class field")

    @property
    def abs_magneticmoment(self) -> float | None:
        """
        Return abs_magneticmoment from most recent JOutStructure.

        Return abs_magneticmoment from most recent JOutStructure.
        """
        if self.jstrucs is not None:
            return self.jstrucs.abs_magneticmoment
        raise AttributeError("Property abs_magneticmoment inaccessible due to empty jstrucs class field")

    @property
    def tot_magneticmoment(self) -> float | None:
        """
        Return tot_magneticmoment from most recent JOutStructure.

        Return tot_magneticmoment from most recent JOutStructure.
        """
        if self.jstrucs is not None:
            return self.jstrucs.tot_magneticmoment
        raise AttributeError("Property tot_magneticmoment inaccessible due to empty jstrucs class field")

    @property
    def mu(self) -> float:
        """
        Return mu from most recent JOutStructure.

        Return mu from most recent JOutStructure. (Equivalent to efermi)
        """
        if self.jstrucs is not None:
            return self.jstrucs.mu
        raise AttributeError("Property mu inaccessible due to empty jstrucs class field")

    ###########################################################################
    # Electronic properties inherited from most recent JElSteps with symbol
    # disambiguation.
    ###########################################################################

    @property
    def elec_niter(self) -> int:
        """Return the most recent electronic iteration.

        Return the most recent electronic iteration.

        Returns
        -------
        elec_niter: int
        """
        if self.jstrucs is not None:
            return self.jstrucs.elec_niter
        raise AttributeError("Property elec_niter inaccessible due to empty jstrucs class field")

    @property
    def elec_e(self) -> float:
        """Return the most recent electronic energy.

        Return the most recent electronic energy.

        Returns
        -------
        elec_e: float
        """
        if self.jstrucs is not None:
            return self.jstrucs.elec_e
        raise AttributeError("Property elec_e inaccessible due to empty jstrucs class field")

    @property
    def elec_grad_k(self) -> float | None:
        """Return the most recent electronic grad_k.

        Return the most recent electronic grad_k.

        Returns
        -------
        grad_k: float
        """
        if self.jstrucs is not None:
            return self.jstrucs.elec_grad_k
        raise AttributeError("Property elec_grad_k inaccessible due to empty jstrucs class field")

    @property
    def elec_alpha(self) -> float | None:
        """Return the most recent electronic alpha.

        Return the most recent electronic alpha.

        Returns
        -------
        alpha: float
        """
        if self.jstrucs is not None:
            return self.jstrucs.elec_alpha
        raise AttributeError("Property elec_alpha inaccessible due to empty jstrucs class field")

    @property
    def elec_linmin(self) -> float | None:
        """Return the most recent electronic linmin.

        Return the most recent electronic linmin.

        Returns
        -------
        linmin: float
        """
        if self.jstrucs is not None:
            return self.jstrucs.elec_linmin
        raise AttributeError("Property elec_linmin inaccessible due to empty jstrucs class field")

    ###########################################################################
    # Creation methods
    ###########################################################################

    @classmethod
    def from_out_slice(cls, text: list[str]) -> JDFTXOutfileSlice:
        """Read slice of out file into a JDFTXOutfileSlice instance.

        Read slice of out file into a JDFTXOutfileSlice instance.

        Parameters
        ----------
        text: list[str]
            file to read
        """
        instance = cls()

        instance.set_min_settings(text)
        instance.set_geomopt_vars(text)
        instance.set_jstrucs(text)
        instance.prefix = instance.get_prefix(text)
        spintype, nspin = instance.get_spinvars(text)
        instance.xc_func = instance.get_xc_func(text)
        instance.spintype = spintype
        instance.nspin = nspin
        broadening_type, broadening = instance.get_broadeningvars(text)
        instance.broadening_type = broadening_type
        instance.broadening = broadening
        instance.kgrid = instance.get_kgrid(text)
        truncation_type, truncation_radius = instance.get_truncationvars(text)
        instance.truncation_type = truncation_type
        instance.truncation_radius = truncation_radius
        instance.pwcut = instance.get_pw_cutoff(text)
        instance.rhocut = instance.get_rho_cutoff(text)
        instance.fftgrid = instance.get_fftgrid(text)
        instance.set_eigvars(text)
        instance.set_orb_fillings()
        instance.is_metal = instance.determine_is_metal()
        instance.set_fluid(text)
        instance.set_total_electrons(text)
        instance.set_nbands(text)
        instance.set_atom_vars(text)
        instance.set_pseudo_vars(text)
        instance.set_lattice_vars(text)
        instance.has_solvation = instance.check_solvation()

        # @ Cooper added @#
        instance.is_gc = key_exists("target-mu", text)
        instance.set_ecomponents(text)
        # instance._build_trajectory(templines)

        return instance

    def get_xc_func(self, text: list[str]) -> str:
        """Get the exchange-correlation functional used in the calculation.

        Get the exchange-correlation functional used in the calculation.

        Parameters
        ----------
        text: list[str]
            output of read_file for out file

        Returns
        -------
        xc_func: str
            exchange-correlation functional used
        """
        line = find_key("elec-ex-corr", text)
        return text[line].strip().split()[-1].strip()

    def get_prefix(self, text: list[str]) -> str | None:
        """Get output prefix from the out file.

        Get output prefix from the out file.

        Parameters
        ----------
            text: list[str]
                output of read_file for out file

        Returns
        -------
            prefix: str
                prefix of dump files for JDFTx calculation
        """
        # prefix = None
        line = find_key("dump-name", text)
        dumpname = text[line].split()[1]
        return dumpname.split(".")[0] if "." in dumpname else dumpname

    def get_spinvars(self, text: list[str]) -> tuple[str, int]:
        """Set spintype and nspin from out file text for instance.

        Set spintype and nspin from out file text for instance.

        Parameters
        ----------
        text: list[str]
            output of read_file for out file

        Returns
        -------
        spintype: str
            type of spin in calculation
        nspin: int
            number of spin types in calculation
        """
        line = find_key("spintype ", text)
        spintype = text[line].split()[1]
        if spintype == "no-spin":
            # vvv This causes many problems vvv
            # spintype = None
            nspin = 1
        elif spintype == "z-spin":
            nspin = 2
        else:
            raise NotImplementedError("have not considered this spin yet")
        return spintype, nspin

    def get_broadeningvars(self, text: list[str]) -> tuple[str, float]:
        """Get broadening type and value from out file text.

        Get broadening type and value from out file text.

        Parameters
        ----------
        text: list[str]
            output of read_file for out file

        Returns
        -------
        broadening_type: str
            type of electronic smearing
        broadening: float
            parameter for electronic smearing
        """
        line = find_key("elec-smearing ", text)
        if line is not None:
            broadening_type = text[line].split()[1]
            broadening = float(text[line].split()[2])
        else:
            broadening_type = None
            broadening = 0
        return broadening_type, broadening

    def get_truncationvars(self, text: list[str]) -> tuple[str, float] | tuple[None, None]:
        """Get truncation type and value from out file text.

        Get truncation type and value from out file text.

        Parameters
        ----------
        text: list[str]
            output of read_file for out file

        Returns
        -------
        truncation_type: str
            type of coulomb truncation
        truncation_radius: float | None
            radius of truncation (if truncation_type is spherical)
        """
        # TODO: Write tests to catch ValueErrors here
        maptypes = {
            # Technically periodic implied no truncation at all, but I'm the Olsen
            # twin that thinks the opposite of fire is 'water', not 'no fire'.
            "Periodic": "periodic",
            "Slab": "slab",
            "Cylindrical": "wire",
            "Wire": "wire",
            "Spherical": "spherical",
            "Isolated": "box",
        }
        line = find_key("coulomb-interaction", text)
        truncation_type = None
        truncation_radius = None
        if line is not None:
            truncation_type = text[line].split()[1]
            if truncation_type not in maptypes:
                raise ValueError("Problem with this truncation!")
            truncation_type = maptypes[truncation_type]
            direc = None
            if len(text[line].split()) == 3:
                direc = text[line].split()[2]
            if self.is_bgw:
                if truncation_type == "slab" and direc != "001":
                    raise ValueError("BGW slab Coulomb truncation must be along z!")
                if truncation_type == "wire" and direc != "001":
                    raise ValueError("BGW wire Coulomb truncation must be periodic in z!")
            if truncation_type == "spherical":
                line = find_key("Initialized spherical truncation of radius", text)
                truncation_radius = float(text[line].split()[5]) / ang_to_bohr
        else:
            # coulomb-interaction tag is always present in out file, so red flag if we can't find it
            raise ValueError("No truncation type found in out file.")
        return truncation_type, truncation_radius

    def get_pw_cutoff(self, text: list[str]) -> float:
        """Get the electron cutoff from the out file text.

        Get the electron cutoff from the out file text.

        Parameters
        ----------
        text: list[str]
            output of read_file for out file

        Returns
        -------
        pwcut: float
            plane wave cutoff used in calculation
        """
        line = find_key("elec-cutoff ", text)
        return float(text[line].split()[1]) * Ha_to_eV

    def get_rho_cutoff(self, text: list[str]) -> float:
        """Get the electron cutoff from the out file text.

        Get the electron cutoff from the out file text.

        Parameters
        ----------
        text: list[str]
            output of read_file for out file

        Returns
        -------
        rhocut: float
            electron density cutoff used in calculation
        """
        line = find_key("elec-cutoff ", text)
        lsplit = text[line].split()
        if len(lsplit) == 3:
            rhocut = float(lsplit[2]) * Ha_to_eV
        else:
            if self.pwcut is None:
                self.pwcut = self.get_pw_cutoff(text)
            rhocut = float(self.pwcut * 4)
        return rhocut

    def get_fftgrid(self, text: list[str]) -> list[int]:
        """Get the FFT grid from the out file text.

        Get the FFT grid from the out file text.

        Parameters
        ----------
        text: list[str]
            output of read_file for out file

        Returns
        -------
        fftgrid: list[int]
            FFT grid used in calculation
        """
        line = find_key_first("Chosen fftbox size", text)
        return [int(x) for x in text[line].split()[6:9]]

    def get_kgrid(self, text: list[str]) -> list[int]:
        """Get the kpoint grid from the out file text.

        Get the kpoint grid from the out file text.

        Parameters
        ----------
        text: list[str]
            output of read_file for out file

        Returns
        -------
        kgrid: list[int]
            kpoint grid used in calculation
        """
        line = find_key("kpoint-folding ", text)
        return [int(x) for x in text[line].split()[1:4]]

    def get_eigstats_varsdict(self, text: list[str], prefix: str | None) -> dict[str, float | None]:
        """Get the eigenvalue statistics from the out file text.

        Get the eigenvalue statistics from the out file text.

        Parameters
        ----------
        text: list[str]
            output of read_file for out file
        prefix: str
            prefix for the eigStats section in the out file

        Returns
        -------
        varsdict: dict[str, float]
            dictionary of eigenvalue statistics
        """
        varsdict: dict[str, float | None] = {}
        lines1 = find_all_key("Dumping ", text)
        lines2 = find_all_key("eigStats' ...", text)
        lines3 = [lines1[i] for i in range(len(lines1)) if lines1[i] in lines2]
        if not len(lines3):
            varsdict["emin"] = None
            varsdict["homo"] = None
            varsdict["efermi"] = None
            varsdict["lumo"] = None
            varsdict["emax"] = None
            varsdict["egap"] = None
            self.has_eigstats = False
        else:
            line = lines3[-1]
            varsdict["emin"] = float(text[line + 1].split()[1]) * Ha_to_eV
            varsdict["homo"] = float(text[line + 2].split()[1]) * Ha_to_eV
            varsdict["efermi"] = float(text[line + 3].split()[2]) * Ha_to_eV
            varsdict["lumo"] = float(text[line + 4].split()[1]) * Ha_to_eV
            varsdict["emax"] = float(text[line + 5].split()[1]) * Ha_to_eV
            varsdict["egap"] = float(text[line + 6].split()[2]) * Ha_to_eV
            self.has_eigstats = True
        return varsdict

    def set_eigvars(self, text: list[str]) -> None:
        """Set the eigenvalue statistics variables.

        Set the eigenvalue statistics variables.

        Parameters
        ----------
        text: list[str]
            output of read_file for out file
        """
        eigstats = self.get_eigstats_varsdict(text, self.prefix)
        self.emin = eigstats["emin"]
        self.homo = eigstats["homo"]
        self.efermi = eigstats["efermi"]
        self.lumo = eigstats["lumo"]
        self.emax = eigstats["emax"]
        self.egap = eigstats["egap"]
        if not self.has_eigstats:
            # TODO: Check if any other variables need to be set
            if self.mu is not None:
                self.efermi = self.mu
            else:
                raise RuntimeError("Variable mu not found to replace variable efermi")

    def get_pp_type(self, text: list[str]) -> str:
        """Get the pseudopotential type used in calculation.

        Get the pseudopotential type used in calculation.

        Parameters
        ----------
        text: list[str]
            output of read_file for out file

        Returns
        -------
        pptype: str
            Pseudopotential library used
        """
        skey = "Reading pseudopotential file"
        line = find_key(skey, text)
        ppfile_example = text[line].split(skey)[1].split(":")[0].strip("'").strip()
        pptype = None
        readable = ["GBRV", "SG15"]
        for _pptype in readable:
            if _pptype in ppfile_example:
                if pptype is not None:
                    if ppfile_example.index(pptype) < ppfile_example.index(_pptype):
                        pptype = _pptype
                    else:
                        pass
                else:
                    pptype = _pptype
        if pptype is None:
            raise ValueError(f"Could not determine pseudopotential type from file name {ppfile_example}")
        return pptype

    def set_pseudo_vars(self, text: list[str]) -> None:
        """Set the pseudopotential variables.

        Set the pseudopotential variables.

        Parameters
        ----------
        text: list[str]
            output of read_file for out file
        """
        self.pp_type = self.get_pp_type(text)
        if self.pp_type in ["SG15", "GBRV"]:
            self.set_pseudo_vars_t1(text)
        else:
            raise NotImplementedError(
                "Outfile parsing requires SG15 or\
                                       GBRV pseudos"
            )

    def set_pseudo_vars_t1(self, text: list[str]) -> None:
        """Set the pseudopotential variables for SG15 pseudopotentials.

        Set the pseudopotential variables for SG15 pseudopotentials.

        Parameters
        ----------
        text: list[str]
            output of read_file for out file
        """
        all_val_lines = find_all_key("valence electrons", text)
        atom_total_elec = []
        bounds_list = get_pseudo_read_section_bounds(text)
        for bounds in bounds_list:
            startline = bounds[0]
            endline = bounds[1]
            val_lines = [x for x in all_val_lines if x < endline and x > startline]
            val_line = val_lines[0]
            val_elec = int(text[val_line].split("valence electrons")[0].strip().split()[-1])
            atom_total_elec.append(val_elec)
        total_elec_dict = {}
        if self.atom_types is not None:
            for i, atom in enumerate(self.atom_types):
                total_elec_dict[atom] = atom_total_elec[i]
        else:
            raise ValueError("Pseuopotential data cannot be allocated without atom types.")
        if self.atom_elements is None:
            raise ValueError("Atom elements not set yet.")
        # total_elec_dict = dict(zip(self.atom_types, atom_total_elec, strict=False))
        # Explicit zipping due to pre-commit in three lines below
        element_total_electrons = np.array([total_elec_dict[x] for x in self.atom_elements])
        pmg_elements = [Element(x) for x in self.atom_elements]
        # element_valence_electrons = np.array([get_atom_valence_electrons(x) for x in self.atom_elements])
        element_valence_electrons = np.array([np.sum(np.array([v[1] for v in el.valences])) for el in pmg_elements])
        # element_valence_electrons = np.array([atom_valence_electrons[x] for x in self.atom_elements])
        element_semicore_electrons = element_total_electrons - element_valence_electrons
        self.total_electrons_uncharged = np.sum(element_total_electrons)
        self.valence_electrons_uncharged = np.sum(element_valence_electrons)
        self.semicore_electrons_uncharged = np.sum(element_semicore_electrons)
        self.semicore_electrons = self.semicore_electrons_uncharged
        if (self.total_electrons is not None) and (self.semicore_electrons is not None):
            self.valence_electrons = self.total_electrons - self.semicore_electrons  # accounts for if system is charged
        else:
            raise ValueError("Total electrons and semicore electrons must be set.")

    def _collect_settings_lines(self, text: list[str], start_flag: str) -> list[int]:
        """Collect the lines of settings from the out file text.

        Collect the lines of settings from the out file text.

        Parameters
        ----------
        text: list[str]
            output of read_file for out file
        start_flag: str
            key to start collecting settings lines

        Returns
        -------
        line_texts: list[int]
            list of line numbers where settings occur
        """
        started = False
        line_texts = []
        for i, line_text in enumerate(text):
            if started:
                if line_text.strip().split()[-1].strip() == "\\":
                    line_texts.append(i)
                else:
                    started = False
            elif start_flag in line_text:
                started = True
            elif len(line_texts):
                break
        return line_texts

    def _create_settings_dict(self, text: list[str], start_flag: str) -> dict:
        """Get a dictionary of settings from the out file text.

        Create a dictionary of settings from the out file text

        Parameters
        ----------
        text: list[str]
            output of read_file for out file
        start_flag: str
            key to start collecting settings lines

        Returns
        -------
        settings_dict: dict
            dictionary of settings
        """
        line_texts = self._collect_settings_lines(text, start_flag)
        settings_dict = {}
        for line_text in line_texts:
            line_text_list = text[line_text].strip().split()
            key = line_text_list[0].lower()
            value = line_text_list[1]
            settings_dict[key] = value
        return settings_dict

    def get_settings_object(
        self,
        text: list[str],
        settings_class: type[JMinSettingsElectronic | JMinSettingsFluid | JMinSettingsIonic | JMinSettingsLattice],
    ) -> JMinSettingsElectronic | JMinSettingsFluid | JMinSettingsIonic | JMinSettingsLattice:
        """Get appropriate JMinSettings mutant.

        Get the settings object from the out file text

        Parameters
        ----------
        text: list[str]
            output of read_file for out file
        settings_class: Type[JMinSettings]
            settings class to create object from

        Returns
        -------
        settings_obj: JMinSettings
            settings object
        """
        settings_dict = self._create_settings_dict(text, settings_class.start_flag)
        return settings_class(params=settings_dict) if len(settings_dict) else None

    def set_min_settings(self, text: list[str]) -> None:
        """Set the settings objects from the out file text.

        Set the settings objects from the out file text.

        Parameters
        ----------
        text: list[str]
            output of read_file for out file
        """
        self.jsettings_fluid = self.get_settings_object(text, JMinSettingsFluid)
        self.jsettings_electronic = self.get_settings_object(text, JMinSettingsElectronic)
        self.jsettings_lattice = self.get_settings_object(text, JMinSettingsLattice)
        self.jsettings_ionic = self.get_settings_object(text, JMinSettingsIonic)

    def set_geomopt_vars(self, text: list[str]) -> None:
        """Set the geom_opt and geom_opt_type class variables.

        Set vars geom_opt and geom_opt_type for initializing self.jstrucs

        Parameters
        ----------
            text: list[str]
                output of read_file for out file
        """
        # Attempts to set all self.jsettings_x class variables
        self.set_min_settings(text)
        if self.jsettings_ionic is None or self.jsettings_lattice is None:
            raise ValueError("Unknown issue in setting settings objects")
        if int(self.jsettings_lattice.params["niterations"]) > 0:
            self.geom_opt = True
            self.geom_opt_type = "lattice"
        elif int(self.jsettings_ionic.params["niterations"]) > 0:
            # elif self.jsettings_ionic.niterations > 0:
            self.geom_opt = True
            self.geom_opt_type = "ionic"
        else:
            self.geom_opt = False
            self.geom_opt_type = "single point"

    def set_jstrucs(self, text: list[str]) -> None:
        """Set the jstrucs class variable.

        Set the JStructures object to jstrucs from the out file text.

        Parameters
        ----------
            text: list[str]
                output of read_file for out file
        """
        self.jstrucs = JOutStructures.from_out_slice(text, iter_type=self.geom_opt_type)
        if self.etype is None:
            self.etype = self.jstrucs[-1].etype

    def set_orb_fillings_nobroad(self, nspin: float) -> None:
        """Set the orbital fillings without broadening.

        Set the orbital fillings without broadening.

        Parameters
        ----------
        nspin: float
            number of spins in calculation
        """
        self.homo_filling = 2 / nspin
        self.lumo_filling = 0

    def set_orb_fillings_broad(
        self, nspin: float, ehomo: float, elumo: float, efermi: float, broadening_type: str, broadening: float
    ):
        self.homo_filling = (2 / nspin) * self.calculate_filling(broadening_type, broadening, ehomo, efermi)
        self.lumo_filling = (2 / nspin) * self.calculate_filling(broadening_type, broadening, elumo, efermi)

    def set_orb_fillings(self) -> None:
        """Set the orbital fillings.

        Calculate and set homo and lumo fillings
        """
        if self.has_eigstats:
            if self.nspin is not None:
                if self.broadening_type is not None:
                    if self.broadening is not None:
                        if self.efermi is not None:
                            if self.homo is not None:
                                if self.lumo is not None:
                                    self.set_orb_fillings_broad(
                                        self.nspin,
                                        self.homo,
                                        self.lumo,
                                        self.efermi,
                                        self.broadening_type,
                                        self.broadening,
                                    )
                                else:
                                    raise ValueError(
                                        "Cannot set orbital fillings with broadening with self.lumo as None"
                                    )
                            else:
                                raise ValueError("Cannot set orbital fillings with broadening with self.homo as None")
                        else:
                            raise ValueError("Cannot set orbital fillings with broadening with self.efermi as None")
                    else:
                        raise ValueError("Cannot set orbital fillings with broadening with self.broadening as None")
                else:
                    self.set_orb_fillings_nobroad(self.nspin)
            else:
                raise ValueError("Cannot set homo/lumo filling with self.nspin as None")

    def set_fluid(self, text: list[str]) -> None:  # Is this redundant to the fluid settings?
        """Set the fluid class variable.

        Set the fluid class variable.

        Parameters
        ----------
        text: list[str]
            output of read_file for out file
        """
        line = find_first_range_key("fluid ", text)
        self.fluid = text[line[0]].split()[
            1
        ]  # This allows self.fluid to be set to the string "None", which is distinct
        # from the None built-in, as it signifies the fluid line was properly read but there is no fluid.

    def set_total_electrons(self, text: list[str]) -> None:
        """Set the total_Electrons class variable.

        Set the total_electrons class variable.

        Parameters
        ----------
        text: list[str]
            output of read_file for out file
        """
        if self.jstrucs is not None:
            self.total_electrons = self.jstrucs.nelectrons

    def set_nbands(self, text: list[str]) -> None:
        """Set the Nbands class variable.

        Set the nbands class variable.
        Prioritizes finding nBands from the reiteration of the input parameters.
        If this line is not found, then it pulls it from "Setting up k-points,
        bands, fillings" section.

        Parameters
        ----------
        text: list[str]
            output of read_file for out file
        """
        lines = find_all_key("elec-n-bands", text)
        if len(lines):
            line = lines[0]
            nbands = int(text[line].strip().split()[-1].strip())
        else:
            lines = find_all_key("nBands:", text)
            line = lines[0]
            nbands = int(text[line].split("nBands:")[1].strip().split()[0].strip())
        self.nbands = nbands

    def set_atom_vars(self, text: list[str]) -> None:
        """Set the atom variables.

        Set the atom variables from the out file text.

        Parameters
        ----------
        text: list[str]
        output of read_file for out file
        """
        startline = find_key("Input parsed successfully", text)
        endline = find_key("---------- Initializing the Grid ----------", text)
        lines = find_first_range_key("ion ", text, startline=startline, endline=endline)
        atom_elements = [text[x].split()[1] for x in lines]
        self.nat = len(atom_elements)
        atom_coords = [text[x].split()[2:5] for x in lines]
        self.atom_coords_initial = np.array(atom_coords, dtype=float)
        atom_types = []
        for x in atom_elements:
            if x not in atom_types:
                atom_types.append(x)
        self.atom_elements = atom_elements
        # mapping_dict = dict(zip(atom_types, range(1, len(atom_types) + 1), strict=False))
        # Below sets mapping_dict identical to above line, but is allowed by pre-commit
        # mapping_dict = {}
        # for atom_type in atom_types:
        #     mapping_dict[atom_type] = atom_valence_electrons[atom_type]
        self.atom_elements_int = [Element(x).Z for x in self.atom_elements]
        self.atom_types = atom_types
        line = find_key("# Ionic positions in", text) + 1
        coords = np.array([text[i].split()[2:5] for i in range(line, line + self.nat)], dtype=float)
        self.atom_coords_final = coords
        self.atom_coords = coords.copy()

    def set_lattice_vars(self, text: list[str]) -> None:
        """Set the lattice variables.

        Set the lattice variables from the out file text.

        Parameters
        ----------
        text: list[str]
            output of read_file for out file
        """
        if self.jstrucs is not None:
            self.lattice_initial = self.jstrucs[0].lattice.matrix
            self.lattice_final = self.jstrucs[-1].lattice.matrix
            self.lattice = self.jstrucs[-1].lattice.matrix.copy()
            self.a, self.b, self.c = np.sum(self.jstrucs[-1].lattice.matrix ** 2, axis=1) ** 0.5
        else:
            raise ValueError("No structures found in out file.")

    def set_ecomponents(self, text: list[str]) -> None:
        """Set the energy components dictionary.

        Set the energy components dictionary from the out file text.

        Parameters
        ----------
        text: list[str]
            output of read_file for out file
        """
        if self.jstrucs is not None:
            ecomp = self.jstrucs[-1].ecomponents
            if self.etype not in ecomp:
                ecomp[self.etype] = self.jstrucs[-1].e
            self.ecomponents = ecomp
        else:
            raise ValueError("No structures found in out file.")

    def calculate_filling(self, broadening_type: str, broadening: float, eig: float, efermi: float) -> float:
        """Calculate the filling for a given eigenvalue.

        Use the broadening type, broadening value, eigenvalue, and fermi energy
        to calculate the filling at the eigenvalue.

        Parameters
        ----------
        broadening_type: str
            type of broadening to use
        broadening: float
            broadening parameter
        eig: float
            eigenvalue
        efermi: float
            fermi energy

        Returns
        -------
        filling: float
            filling at the eigenvalue
        """
        # most broadening implementations do not have the denominator factor
        # of 2, but JDFTx does currently.
        # Remove if use this for other code outfile reading
        x = (eig - efermi) / (2.0 * broadening)
        if broadening_type == "Fermi":
            filling = 0.5 * (1 - np.tanh(x))
        elif broadening_type == "Gauss":
            filling = 0.5 * (1 - math.erf(x))
        elif broadening_type == "MP1":
            filling = 0.5 * (1 - math.erf(x)) - x * np.exp(-1 * x**2) / (2 * np.pi**0.5)
        elif broadening_type == "Cold":
            filling = 0.5 * (1 - math.erf(x + 0.5**0.5)) + np.exp(-1 * (x + 0.5**0.5) ** 2) / (2 * np.pi) ** 0.5
        else:
            raise NotImplementedError("Have not added other broadening types")

        return filling

    def _determine_is_metal(self, tol_partial: float, nspin: int, homo_filling: float, lumo_filling: float) -> bool:
        return not (homo_filling / (2 / nspin) > (1 - tol_partial) and lumo_filling / (2 / nspin) < tol_partial)

    def determine_is_metal(self) -> bool | None:
        """Determine if the system is a metal based.

        Determine if the system is a metal based on the fillings of
        homo and lumo.

        Returns
        -------
        is_metal: bool
            True if system is metallic
        """
        tol_partial = 0.01
        if self.has_eigstats:
            if self.nspin is not None:
                if self.homo_filling is not None:
                    if self.lumo_filling is not None:
                        return self._determine_is_metal(tol_partial, self.nspin, self.homo_filling, self.lumo_filling)
                    raise ValueError("Cannot determine if system is metal - self.lumo_filling undefined")
                raise ValueError("Cannot determine if system is metal - self.homo_filling undefined")
            raise ValueError("Cannot determine if system is metal - self.nspin undefined")
        return None

    def check_solvation(self) -> bool:
        """Check for implicit solvation.

        Check if calculation used implicit solvation

        Returns
        -------
        has_solvation: bool
            True if calculation used implicit solvation
        """
        return self.fluid is not None

    def write(self) -> None:
        """Return an error.

        Return an error. (pre-commit needs a docustring here)
        """
        # don't need a write method since will never do that
        raise NotImplementedError("There is no need to write a JDFTx out file")

    def to_dict(self) -> dict:
        """Convert dataclass to dictionary representation.

        Convert dataclass to dictionary representation.

        Returns
        -------
        dict
            JDFTXOutfileSlice in dictionary format
        """
        # convert dataclass to dictionary representation
        dct = {}
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            dct[field] = value

        # Include properties in the dictionary representation
        for name, _obj in inspect.getmembers(type(self), lambda o: isinstance(o, property)):
            dct[name] = getattr(self, name)
        return dct

    # This method is likely never going to be called as all (currently existing)
    # attributes of the most recent slice are explicitly defined as a class
    # property. However, it is included to reduce the likelihood of errors
    # upon future changes to downstream code.
    def __getattr__(self, name: str) -> Any:
        """Return attribute value.

        Return the value of an attribute.

        Parameters
        ----------
        name: str
            The name of the attribute

        Returns
        -------
        value
            The value of the attribute
        """
        # The whole point of this is to be an unreached safety net, so expect hit
        # from coverage here
        # if name not in self.__dict__:
        # if name not in inspect.getmembers(self):
        #     if not hasattr(self.jstrucs, name):
        #         raise AttributeError(f"{self.__class__.__name__} not found: {name}")
        #     return getattr(self.jstrucs, name)
        # return self.__dict__[name]
        # Check if the attribute is in self.__dict__
        # Check if the attribute is in self.__dict__
        if name in self.__dict__:
            return self.__dict__[name]

        # Check if the attribute is a property of the class
        for cls in inspect.getmro(self.__class__):
            if name in cls.__dict__ and isinstance(cls.__dict__[name], property):
                return cls.__dict__[name].__get__(self)

        # Check if the attribute is in self.jstrucs
        if hasattr(self.jstrucs, name):
            return getattr(self.jstrucs, name)

        # If the attribute is not found in either, raise an AttributeError
        raise AttributeError(f"{self.__class__.__name__} not found: {name}")
