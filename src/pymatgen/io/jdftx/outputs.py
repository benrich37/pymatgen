"""JDFTx outputs parsing module.

Module for parsing outputs of JDFTx.

Note: JDFTXOutfile will be moved back to its own module once a more broad outputs
class is written.
"""

from __future__ import annotations

import pprint
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pymatgen.core.trajectory import Trajectory
from pymatgen.io.jdftx._output_utils import read_outfile_slices
from pymatgen.io.jdftx.jdftxoutfileslice import JDFTXOutfileSlice

if TYPE_CHECKING:
    import numpy as np

    from pymatgen.core.structure import Structure
    from pymatgen.io.jdftx.jelstep import JElSteps
    from pymatgen.io.jdftx.jminsettings import (
        JMinSettingsElectronic,
        JMinSettingsFluid,
        JMinSettingsIonic,
        JMinSettingsLattice,
    )
    from pymatgen.io.jdftx.joutstructures import JOutStructures

__author__ = "Ben Rich, Jacob Clary"


_jof_atr_from_last_slice = [
    "prefix",
    "jstrucs",
    "jsettings_fluid",
    "jsettings_electronic",
    "jsettings_lattice",
    "jsettings_ionic",
    "xc_func",
    "lattice_initial",
    "lattice_final",
    "lattice",
    "a",
    "b",
    "c",
    "fftgrid",
    "geom_opt",
    "geom_opt_type",
    "efermi",
    "egap",
    "emin",
    "emax",
    "homo",
    "lumo",
    "homo_filling",
    "lumo_filling",
    "is_metal",
    "converged",
    "etype",
    "broadening_type",
    "broadening",
    "kgrid",
    "truncation_type",
    "truncation_radius",
    "pwcut",
    "rhocut",
    "pp_type",
    "total_electrons",
    "semicore_electrons",
    "valence_electrons",
    "total_electrons_uncharged",
    "semicore_electrons_uncharged",
    "valence_electrons_uncharged",
    "nbands",
    "atom_elements",
    "atom_elements_int",
    "atom_types",
    "spintype",
    "nspin",
    "nat",
    "atom_coords_initial",
    "atom_coords_final",
    "atom_coords",
    "structure",
    "has_solvation",
    "fluid",
    "is_gc",
    "eopt_type",
    "elecmindata",
    "stress",
    "strain",
    "nstep",
    "e",
    "grad_k",
    "alpha",
    "linmin",
    "abs_magneticmoment",
    "tot_magneticmoment",
    "mu",
    "elec_nstep",
    "elec_e",
    "elec_grad_k",
    "elec_alpha",
    "elec_linmin",
    "electronic_output",
    "t_s",
    "ecomponents",
]


@dataclass
class JDFTXOutfile:
    """
    JDFTx out file parsing class.

    A class to read and process a JDFTx out file.

    Methods:
        from_file(file_path: str | Path) -> JDFTXOutfile:
            Return JDFTXOutfile object from the path to a JDFTx out file.

    Attributes:
        slices (list[JDFTXOutfileSlice]): A list of JDFTXOutfileSlice objects. Each slice corresponds to an individual
            call of the JDFTx executable. Subsequent JDFTx calls within the same directory and prefix will append
            outputs to the same out file. More than one slice may correspond to restarted calculations, geom + single
            point calculations, or optimizations done with 3rd-party wrappers like ASE.
        prefix (str): The prefix of the most recent JDFTx call.
        jstrucs (JOutStructures): The JOutStructures object from the most recent JDFTx call. This object contains a
            series of JOutStructure objects in its 'slices' attribute, each corresponding to a single structure
            (multiple iff performing a geometric optimization) as well as convergence data for the structures as a
            series.
        jsettings_fluid (JMinSettingsFluid): The JMinSettingsFluid object from the most recent JDFTx call. This object
            contains only a 'params' attribute, which is a dictionary of the input parameters for the fluid
            optimization.
        jsettings_electronic (JMinSettingsElectronic): The JMinSettingsElectronic object from the most recent JDFTx
            call. This object contains only a 'params' attribute, which is a dictionary of the input parameters for the
            electronic optimization.
        jsettings_lattice (JMinSettingsLattice): The JMinSettingsLattice object from the most recent JDFTx call. This
            object contains only a 'params' attribute, which is a dictionary of the input parameters for the lattice
            optimization.
        jsettings_ionic (JMinSettingsIonic): The JMinSettingsIonic object from the most recent JDFTx call. This object
            contains only a 'params' attribute, which is a dictionary of the input parameters for the ionic
            optimization.
        xc_func (str): The exchange-correlation functional used in the most recent JDFTx call. See documentation for
            JDFTx online for a list of available exchange-correlation functionals.
        lattice_initial (np.ndarray): The initial lattice vectors of the most recent JDFTx call as a 3x3 numpy array.
            In units of Angstroms.
        lattice_final (np.ndarray): The final lattice vectors of the most recent JDFTx call as a 3x3 numpy array. In
            units of Angstroms.
        lattice (np.ndarray): The lattice vectors of the most recent JDFTx call as a 3x3 numpy array (redundant to
            lattice_final).
        a (float): Length of the first lattice vector. In units of Angstroms.
        b (float): Length of the second lattice vector. In units of Angstroms.
        c (float): Length of the third lattice vector. In units of Angstroms.
        fftgrid (list[int]): The FFT grid shape used in the most recent JDFTx call. Can be used to properly shape
            densities dumped as binary files.
        geom_opt (bool): True if the most recent JDFTx call was a geometry optimization (lattice or ionic).
        geom_opt_type (str): The type of geometry optimization performed in the most recent JDFTx call. Options are
            'lattice' or 'ionic' if geom_opt, else "single point". ('lattice' optimizations perform ionic optimizations
            as well unless ion positions are given in direct coordinates).
        ecomponents (dict): The components of the total energy in eV of the most recent JDFTx call.
        efermi (float): The Fermi energy in eV of the most recent JDFTx call. Equivalent to "mu".
        egap (float): The band gap in eV of the most recent JDFTx call. (Only available if eigstats was dumped).
        emin (float): The minimum energy in eV (smallest Kohn-Sham eigenvalue) of the most recent JDFTx call. (Only
            available if eigstats was dumped).
        emax (float): The maximum energy in eV (largest Kohn-Sham eigenvalue) of the most recent JDFTx call. (Only
            available if eigstats was dumped).
        homo (float): The energy in eV of the band-gap lower bound (Highest Occupied Molecular Orbital) (Only available
            if eigstats was dumped).
        lumo (float): The energy in eV of the band-gap upper bound (Lowest Unoccupied Molecular Orbital) (Only
            available if eigstats was dumped).
        homo_filling (float): The electron filling at the homo band-state. (Only available if eigstats was dumped).
        lumo_filling (float): The electron filling at the lumo band-state. (Only available if eigstats was dumped).
        is_metal (bool): True if fillings of homo and lumo band-states are off-set by 1 and 0 by at least an arbitrary
            tolerance of 0.01 (ie 1 - 0.015 and 0.012 for homo/lumo fillings would be metallic, while 1-0.001 and 0
            would not be). (Only available if eigstats was dumped).
        converged (bool): True if most recent SCF cycle converged (and geom forces converged is calc is geom_opt)
        etype (str): String representation of total energy-type of system. Commonly "G" (grand-canonical potential) for
            GC calculations, and "F" for canonical (fixed electron count) calculations.
        broadening_type (str): Type of broadening for electronic filling about Fermi-level requested. Either "Fermi",
            "Cold", "MP1", or "Gauss".
        broadening (float): Magnitude of broadening for electronic filling.
        kgrid (list[int]): Shape of k-point grid used in calculation. (equivalent to k-point folding)
        truncation_type (str): Type of coulomb truncation used to prevent interaction between periodic images along
            certain directions. "periodic" means no coulomb truncation was used.
        truncation_radius (float | None): If spherical truncation_type, this is the radius of the coulomb truncation
            sphere.
        pwcut (float): The plane-wave cutoff energy in Hartrees used in the most recent JDFTx call.
        rhocut (float): The density cutoff energy in Hartrees used in the most recent JDFTx call.
        pp_type (str): The pseudopotential library used in the most recent JDFTx call. Currently only "GBRV" and "SG15"
            are supported by this output parser.
        total_electrons (float): The total number of electrons in the most recent JDFTx call (redundant to nelectrons).
        semicore_electrons (int): The number of semicore electrons in the most recent JDFTx call.
        valence_electrons (float): The number of valence electrons in the most recent JDFTx call.
        total_electrons_uncharged (int): The total number of electrons in the most recent JDFTx call, uncorrected for
            charge. (ie total_electrons + charge)
        semicore_electrons_uncharged (int): The number of semicore electrons in the most recent JDFTx call, uncorrected
            for charge. (ie semicore_electrons + charge)
        valence_electrons_uncharged (int): The number of valence electrons in the most recent JDFTx call, uncorrected
            for charge. (ie valence_electrons + charge)
        nbands (int): The number of bands used in the most recent JDFTx call.
        atom_elements (list[str]): The list of each ion's element symbol in the most recent JDFTx call.
        atom_elements_int (list[int]): The list of ion's atomic numbers in the most recent JDFTx call.
        atom_types (list[str]): Non-repeating list of each ion's element symbol in the most recent JDFTx call.
        spintype (str): The spin type used in the most recent JDFTx call. Options are "none", "collinear",
        nspin (int): The number of spins used in the most recent JDFTx call.
        nat (int): The number of atoms in the most recent JDFTx call.
        atom_coords_initial (list[list[float]]): The initial atomic coordinates of the most recent JDFTx call.
        atom_coords_final (list[list[float]]): The final atomic coordinates of the most recent JDFTx call.
        atom_coords (list[list[float]]): The atomic coordinates of the most recent JDFTx call.
        structure (Structure): The updated pymatgen Structure object of the most recent JDFTx call.
        trajectory (Trajectory): The Trajectory object of the most recent JDFTx call.
        has_solvation (bool): True if the most recent JDFTx call included a solvation calculation.
        fluid (str): The fluid used in the most recent JDFTx call.
        is_gc (bool): True if the most recent slice is a grand canonical calculation.
        eopt_type (str): The type of energy iteration used in the most recent JDFTx call.
        elecmindata (JElSteps): The JElSteps object from the most recent JDFTx call. This object contains a series of
            JElStep objects in its 'steps' attribute, each corresponding to a single energy iteration.
        stress (np.ndarray): The stress tensor of the most recent JDFTx call as a 3x3 numpy array. In units of
            eV/Angstrom^3.
        strain (np.ndarray): The strain tensor of the most recent JDFTx call as a 3x3 numpy array.
        nstep (int): The number of geometric optimization steps in the most recent JDFTx call.
        e (float): The final energy in eV of the most recent JDFTx call (equivalent to the call's etype).
        grad_k (float): The final norm of the preconditioned gradient for geometric optimization of the most recent
            JDFTx call (evaluated as dot(g, Kg), where g is the gradient and Kg is the preconditioned gradient).
            (written as "|grad|_K" in JDFTx output).
        alpha (float): The step size of the final geometric step in the most recent JDFTx call.
        linmin (float): The final normalized projection of the geometric step direction onto the gradient for the most
            recent JDFTx call.
        abs_magneticmoment (float | None): The absolute magnetic moment of the most recent JDFTx call.
        tot_magneticmoment (float | None): The total magnetic moment of the most recent JDFTx call.
        mu (float): The Fermi energy in eV of the most recent JDFTx call.
        elec_e (float): The final energy in eV of the most recent electronic optimization step.
        elec_nstep (int): The number of electronic optimization steps in the most recent JDFTx call.
        elec_grad_k (float): The final norm of the preconditioned gradient for electronic optimization of the most
            recent JDFTx call (evaluated as dot(g, Kg), where g is the gradient and Kg is the preconditioned gradient).
            (written as "|grad|_K" in JDFTx output).
        elec_alpha (float): The step size of the final electronic step in the most recent JDFTx call.
        elec_linmin (float): The final normalized projection of the electronic step direction onto the gradient for the
            most recent JDFTx call.

    Magic Methods:
        __getitem__(key: str | int) -> Any: Decides behavior of how JDFTXOutfile objects are indexed. If the key is a
            string, it will return the value of the property with the same name. If the key is an integer, it will
            return the slice of the JDFTXOutfile object at that index.
        __len__() -> int: Returns the number of slices in the JDFTXOutfile object.
        __getattr__(name: str) -> Any: Returns the value of the property with the same name as the input string.
        __str__() -> str: Returns a string representation of the JDFTXOutfile object.
    """

    slices: list[JDFTXOutfileSlice] = field(default_factory=list)
    prefix: str = field(init=False)
    jstrucs: JOutStructures = field(init=False)
    jsettings_fluid: JMinSettingsFluid = field(init=False)
    jsettings_electronic: JMinSettingsElectronic = field(init=False)
    jsettings_lattice: JMinSettingsLattice = field(init=False)
    jsettings_ionic: JMinSettingsIonic = field(init=False)
    xc_func: str = field(init=False)
    lattice_initial: np.ndarray = field(init=False)
    lattice_final: np.ndarray = field(init=False)
    lattice: np.ndarray = field(init=False)
    a: float = field(init=False)
    b: float = field(init=False)
    c: float = field(init=False)
    fftgrid: list[int] = field(init=False)
    geom_opt: bool = field(init=False)
    geom_opt_type: str = field(init=False)
    efermi: float = field(init=False)
    egap: float = field(init=False)
    emin: float = field(init=False)
    emax: float = field(init=False)
    homo: float = field(init=False)
    lumo: float = field(init=False)
    homo_filling: float = field(init=False)
    lumo_filling: float = field(init=False)
    is_metal: bool = field(init=False)
    converged: bool = field(init=False)
    etype: str = field(init=False)
    broadening_type: str = field(init=False)
    broadening: float = field(init=False)
    kgrid: list[int] = field(init=False)
    truncation_type: str = field(init=False)
    truncation_radius: float = field(init=False)
    pwcut: float = field(init=False)
    rhocut: float = field(init=False)
    pp_type: str = field(init=False)
    total_electrons: float = field(init=False)
    semicore_electrons: int = field(init=False)
    valence_electrons: float = field(init=False)
    total_electrons_uncharged: int = field(init=False)
    semicore_electrons_uncharged: int = field(init=False)
    valence_electrons_uncharged: int = field(init=False)
    nbands: int = field(init=False)
    atom_elements: list[str] = field(init=False)
    atom_elements_int: list[int] = field(init=False)
    atom_types: list[str] = field(init=False)
    spintype: str = field(init=False)
    nspin: int = field(init=False)
    nat: int = field(init=False)
    atom_coords_initial: list[list[float]] = field(init=False)
    atom_coords_final: list[list[float]] = field(init=False)
    atom_coords: list[list[float]] = field(init=False)
    structure: Structure = field(init=False)
    trajectory: Trajectory = field(init=False)
    has_solvation: bool = field(init=False)
    fluid: str = field(init=False)
    is_gc: bool = field(init=False)
    eopt_type: str = field(init=False)
    elecmindata: JElSteps = field(init=False)
    stress: np.ndarray = field(init=False)
    strain: np.ndarray = field(init=False)
    nstep: int = field(init=False)
    e: float = field(init=False)
    grad_k: float = field(init=False)
    alpha: float = field(init=False)
    linmin: float = field(init=False)
    abs_magneticmoment: float = field(init=False)
    tot_magneticmoment: float = field(init=False)
    mu: float = field(init=False)
    elec_nstep: int = field(init=False)
    elec_e: float = field(init=False)
    elec_grad_k: float = field(init=False)
    elec_alpha: float = field(init=False)
    elec_linmin: float = field(init=False)
    electronic_output: float = field(init=False)

    @classmethod
    def from_calc_dir(
        cls, calc_dir: str | Path, is_bgw: bool = False, none_slice_on_error: bool = False
    ) -> JDFTXOutfile:
        """
        Create a JDFTXOutfile object from a directory containing JDFTx out files.

        Args:
            calc_dir (str | Path): The path to the directory containing the JDFTx out files.
            is_bgw (bool): Mark True if data must be usable for BGW calculations. This will change the behavior of the
                parser to be stricter with certain criteria.
            none_slice_on_error (bool): If True, will return None if an error occurs while parsing a slice instead of
                halting the parsing process. This can be useful for parsing files with multiple slices where some slices
                may be incomplete or corrupted.

        Returns:
            JDFTXOutfile: The JDFTXOutfile object.
        """
        file_path = _find_jdftx_out_file(Path(calc_dir))
        texts = read_outfile_slices(file_path)
        slices = [
            JDFTXOutfileSlice._from_out_slice(text, is_bgw=is_bgw, none_on_error=none_slice_on_error) for text in texts
        ]
        return cls(slices=slices)

    @classmethod
    def from_file(cls, file_path: str | Path, is_bgw: bool = False, none_slice_on_error: bool = False) -> JDFTXOutfile:
        """
        Create a JDFTXOutfile object from a JDFTx out file.

        Args:
            file_path (str | Path): The path to the JDFTx out file.
            is_bgw (bool): Mark True if data must be usable for BGW calculations. This will change the behavior of the
                parser to be stricter with certain criteria.
            none_slice_on_error (bool): If True, will return None if an error occurs while parsing a slice instead of
                halting the parsing process. This can be useful for parsing files with multiple slices where some slices
                may be incomplete or corrupted.

        Returns:
            JDFTXOutfile: The JDFTXOutfile object.
        """
        texts = read_outfile_slices(file_path)
        slices = [
            JDFTXOutfileSlice._from_out_slice(text, is_bgw=is_bgw, none_on_error=none_slice_on_error) for text in texts
        ]
        return cls(slices=slices)

    def __post_init__(self):
        if len(self.slices):
            for var in _jof_atr_from_last_slice:
                setattr(self, var, getattr(self.slices[-1], var))
            self.trajectory = self._get_trajectory()

    def _get_trajectory(self) -> Trajectory:
        """Set the trajectory attribute of the JDFTXOutfile object."""
        constant_lattice = True
        structures = []
        for _i, slc in enumerate(self.slices):
            structures += slc.jstrucs.slices
            if constant_lattice and (slc.jsettings_lattice is not None):
                if "niterations" in slc.jsettings_lattice.params:
                    if int(slc.jsettings_lattice.params["niterations"]) > 1:
                        constant_lattice = False
                else:
                    constant_lattice = False

        return Trajectory.from_structures(structures=structures, constant_lattice=constant_lattice)

    def to_dict(self) -> dict:
        """
        Convert the JDFTXOutfile object to a dictionary.

        Returns:
            dict: A dictionary representation of the JDFTXOutfile object.
        """
        dct = {}
        for fld in self.__dataclass_fields__:
            if fld == "slices":
                dct[fld] = [slc.to_dict() for slc in self.slices]
                continue
            value = getattr(self, fld)
            dct[fld] = value
        return dct

    ###########################################################################
    # Magic methods
    ###########################################################################

    def __getitem__(self, key: int | str) -> JDFTXOutfileSlice | Any:
        """Return item.

        Args:
            key (int | str): The key of the item.

        Returns:
            JDFTXOutfileSlice | Any: The value of the item.

        Raises:
            TypeError: If the key type is invalid.
        """
        val = None
        if type(key) is int:
            val = self.slices[key]
        elif type(key) is str:
            val = getattr(self, key)
        else:
            raise TypeError(f"Invalid key type: {type(key)}")
        return val

    def __len__(self) -> int:
        """Return length of JDFTXOutfile object.

        Returns:
            int: The number of geometric optimization steps in the JDFTXOutfile object.
        """
        return len(self.slices)

    def __str__(self) -> str:
        """Return string representation of JDFTXOutfile object.

        Returns:
            str: The string representation of the JDFTXOutfile object.
        """
        return pprint.pformat(self)


def _find_jdftx_out_file(calc_dir: Path) -> Path:
    """
    Find the JDFTx out file in a directory.

    Args:
        calc_dir (Path): The directory containing the JDFTx out file.

    Returns:
        Path: The path to the JDFTx out file.
    """
    out_files = list(calc_dir.glob("*.out")) + list(calc_dir.glob("out"))
    if len(out_files) == 0:
        raise FileNotFoundError("No JDFTx out file found in directory.")
    if len(out_files) > 1:
        raise FileNotFoundError("Multiple JDFTx out files found in directory.")
    return out_files[0]
