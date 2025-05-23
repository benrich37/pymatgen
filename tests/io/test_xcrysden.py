from __future__ import annotations

import numpy as np

from pymatgen.core.structure import Structure
from pymatgen.io.xcrysden import XSF
from pymatgen.util.testing import MatSciTest


class TestXSF(MatSciTest):
    def setup_method(self):
        self.coords = [[0, 0, 0], [0.75, 0.5, 0.75]]
        self.lattice = [
            [3.8401979337, 0.00, 0.00],
            [1.9200989668, 3.3257101909, 0.00],
            [0.00, -2.2171384943, 3.1355090603],
        ]
        self.struct = Structure(self.lattice, ["Si", "Si"], self.coords)

    def test_xsf(self):
        xsf = XSF(self.struct)
        assert self.struct, XSF.from_str(xsf.to_str())
        xsf = XSF(self.struct)
        assert self.struct, XSF.from_str(xsf.to_str())

    def test_append_vect(self):
        self.struct.add_site_property("vect", np.eye(2, 3))
        xsf_str = XSF(self.struct).to_str()
        last_line_split = xsf_str.split("\n")[-1].split()
        assert len(last_line_split) == 7
        assert last_line_split[-1] == "0.00000000000000"
        assert last_line_split[-2] == "1.00000000000000"
        assert last_line_split[-3] == "0.00000000000000"

    def test_to_str(self):
        structure = self.get_structure("Li2O")
        xsf = XSF(structure)
        assert (
            xsf.to_str()
            == """CRYSTAL
# Primitive lattice vectors in Angstrom
PRIMVEC
 2.91738857000000 0.09789437000000 1.52000466000000
 0.96463406000000 2.75503561000000 1.52000466000000
 0.13320635000000 0.09789443000000 3.28691771000000
# Cartesian coordinates in Angstrom.
PRIMCOORD
 3 1
O     0.00000000000000     0.00000000000000     0.00000000000000
Li     3.01213761017484     2.21364440998406     4.74632330032018
Li     1.00309136982516     0.73718000001594     1.58060372967982"""
        )

        assert (
            xsf.to_str(atom_symbol=False)
            == """CRYSTAL
# Primitive lattice vectors in Angstrom
PRIMVEC
 2.91738857000000 0.09789437000000 1.52000466000000
 0.96463406000000 2.75503561000000 1.52000466000000
 0.13320635000000 0.09789443000000 3.28691771000000
# Cartesian coordinates in Angstrom.
PRIMCOORD
 3 1
8     0.00000000000000     0.00000000000000     0.00000000000000
3     3.01213761017484     2.21364440998406     4.74632330032018
3     1.00309136982516     0.73718000001594     1.58060372967982"""
        )

    def test_xsf_symbol_parse(self):
        """Ensure that the same structure is parsed
        even if the atomic symbol / number convention
        is different.
        """
        test_str = """
CRYSTAL
PRIMVEC
       11.45191956     0.00000000     0.00000000
        5.72596044     9.91765288     0.00000000
      -14.31490370    -8.26471287    23.37613199
PRIMCOORD
1 1
H     -0.71644986    -0.41364333     1.19898200     0.00181803     0.00084718     0.00804832
"""
        structure = XSF.from_str(test_str).structure
        assert str(structure.species[0]) == "H"
        test_string2 = """
CRYSTAL
PRIMVEC
       11.45191956     0.00000000     0.00000000
        5.72596044     9.91765288     0.00000000
      -14.31490370    -8.26471287    23.37613199
PRIMCOORD
1 1
1     -0.71644986    -0.41364333     1.19898200     0.00181803     0.00084718     0.00804832
"""

        structure2 = XSF.from_str(test_string2).structure
        assert structure == structure2
