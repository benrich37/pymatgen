"""Module for parsing single SCF step from JDFTx.

This module contains the JEiter class for parsing single SCF step from a JDFTx out file.
"""

from __future__ import annotations

from pymatgen.core.units import Ha_to_eV
from pymatgen.io.jdftx.utils import get_colon_var_t1

__author__ = "Ben Rich"


class JEiter:
    """Electronic minimization data for a single SCF step.

    Class object for storing logged electronic minimization data for a single
    SCF step.
    """

    iter_type: str | None = None
    etype: str | None = None
    niter: int | None = None
    e: float | None = None
    grad_k: float | None = None
    alpha: float | None = None
    linmin: float | None = None
    t_s: float | None = None
    mu: float | None = None
    nelectrons: float | None = None
    abs_magneticmoment: float | None = None
    tot_magneticmoment: float | None = None
    subspacerotationadjust: float | None = None
    converged: bool = False
    converged_reason: str | None = None

    @classmethod
    def from_lines_collect(cls, lines_collect: list[str], iter_type: str, etype: str) -> JEiter:
        """Return JEiter object.

        Create a JEiter object from a list of lines of text from a JDFTx out
        file corresponding to a single SCF step.

        Parameters
        ----------
        lines_collect: list[str]
            A list of lines of text from a JDFTx out file corresponding to a
            single SCF step
        iter_type: str
            The type of electronic minimization step
        etype: str
            The type of energy component
        """
        instance = cls()
        instance.iter_type = iter_type
        instance.etype = etype
        _iter_flag = f"{iter_type}: Iter: "
        for i, line_text in enumerate(lines_collect):
            if instance.is_iter_line(i, line_text, _iter_flag):
                instance.read_iter_line(line_text)
            elif instance.is_fillings_line(i, line_text):
                instance.read_fillings_line(line_text)
            elif instance.is_subspaceadjust_line(i, line_text):
                instance.read_subspaceadjust_line(line_text)
        return instance

    def is_iter_line(self, i: int, line_text: str, _iter_flag: str) -> bool:
        """Return True if opt iter line.

        Return True if the line_text is the start of a log message for a
        JDFTx optimization step.

        Parameters
        ----------
        i: int
            The index of the line in the text slice
        line_text: str
            A line of text from a JDFTx out file
        _iter_flag:  str
            The flag that indicates the start of a log message for a JDFTx
            optimization step

        Returns
        -------
        is_line: bool
            True if the line_text is the start of a log message for a JDFTx
            optimization step
        """
        return _iter_flag in line_text

    def read_iter_line(self, line_text: str) -> None:
        """Set class variables iter, E, grad_K, alpha, linmin, t_s.

        Parse the lines of text corresponding to the electronic minimization
        data of a JDFTx out file.

        Parameters
        ----------
        line_text: str
            A line of text from a JDFTx out file containing the electronic
            minimization data
        """
        niter_float = get_colon_var_t1(line_text, "Iter: ")
        if isinstance(niter_float, float):
            self.niter = int(niter_float)
        elif niter_float is None:
            raise ValueError("Could not find niter in line_text")
        self.e = get_colon_var_t1(line_text, f"{self.etype}: ") * Ha_to_eV
        self.grad_k = get_colon_var_t1(line_text, "|grad|_K: ")
        self.alpha = get_colon_var_t1(line_text, "alpha: ")
        self.linmin = get_colon_var_t1(line_text, "linmin: ")
        self.t_s = get_colon_var_t1(line_text, "t[s]: ")

    def is_fillings_line(self, i: int, line_text: str) -> bool:
        """Return True if fillings line.

        Return True if the line_text is the start of a log message for a
        JDFTx optimization step.

        Parameters
        ----------
        i: int
            The index of the line in the text slice
        line_text: str
            A line of text from a JDFTx out file

        Returns
        -------
        is_line: bool
            True if the line_text is the start of a log message for a JDFTx
            optimization step
        """
        return "FillingsUpdate" in line_text

    def read_fillings_line(self, fillings_line: str) -> None:
        """Set class variables mu, nelectrons, magneticmoment.

        Parse the lines of text corresponding to the electronic minimization
        data of a JDFTx out file.

        Parameters
        ----------
        fillings_line: str
            A line of text from a JDFTx out file containing the electronic
            minimization data
        """
        if "FillingsUpdate:" in fillings_line:
            self.set_mu(fillings_line)
            self.set_nelectrons(fillings_line)
            if "magneticMoment" in fillings_line:
                self.set_magdata(fillings_line)
        else:
            raise ValueError("FillingsUpdate string not found")

    def is_subspaceadjust_line(self, i: int, line_text: str) -> bool:
        """Return True if subspace adjust line.

        Return True if the line_text is the start of a log message for a
        JDFTx optimization step.

        Parameters
        ----------
        i: int
            The index of the line in the text slice
        line_text: str
            A line of text from a JDFTx out file

        Returns
        -------
        is_line: bool
            True if the line_text is the start of a log message for a JDFTx
            optimization step
        """
        return "SubspaceRotationAdjust" in line_text

    def read_subspaceadjust_line(self, line_text: str) -> None:
        """Set class variable subspaceRotationAdjust.

        Parse the lines of text corresponding to the electronic minimization
        data of a JDFTx out file.

        Parameters
        ----------
        line_text: str
            A line of text from a JDFTx out file containing the electronic
            minimization data
        """
        self.subspacerotationadjust = get_colon_var_t1(line_text, "SubspaceRotationAdjust: set factor to")

    def set_magdata(self, fillings_line: str) -> None:
        """Set class variables abs_magneticMoment, tot_magneticMoment.

        Parse the lines of text corresponding to the electronic minimization
        data of a JDFTx out file.

        Parameters
        ----------
        fillings_line: str
            A line of text from a JDFTx out file containing the electronic
            minimization data
        """
        _fillings_line = fillings_line.split("magneticMoment: [ ")[1].split(" ]")[0].strip()
        self.abs_magneticmoment = get_colon_var_t1(_fillings_line, "Abs: ")
        self.tot_magneticmoment = get_colon_var_t1(_fillings_line, "Tot: ")

    def set_mu(self, fillings_line: str) -> None:
        """Set mu class variable.

        Parse the lines of text corresponding to the electronic minimization
        data of a JDFTx out file.

        Parameters
        ----------
        fillings_line: str
            A line of text from a JDFTx out file containing the electronic
            minimization data
        """
        self.mu = get_colon_var_t1(fillings_line, "mu: ") * Ha_to_eV

    def set_nelectrons(self, fillings_line: str) -> None:
        """Set nelectrons class variable.

        Parse the lines of text corresponding to the electronic minimization
        data of a JDFTx out file.

        Parameters
        ----------
        fillings_line: str
            A line of text from a JDFTx out file containing the electronic
            minimization data
        """
        self.nelectrons = get_colon_var_t1(fillings_line, "nElectrons: ")
