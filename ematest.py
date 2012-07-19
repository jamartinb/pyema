#!/usr/bin/env python
# coding=utf-8
##
# This file is part of pyema.
#
# pyema is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyema is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pyema.  If not, see <http://www.gnu.org/licenses/>.
##



"""\
Tests the python implementation of EMA
"""


import unittest;
import logging;

import numpy;

from ema import process_dataset, file2dataset;

log = logging.getLogger('ematest');



class EmaTest(unittest.TestCase):


    def test_scientist_17(self):
        with file("scientist-17.sparse",'r') as inp:
            true_result = numpy.array([0.3058, 0.544815]);
            result_matrix = process_dataset(file2dataset(inp));
            result = numpy.mean(numpy.array(result_matrix)[:,2:],0);
            self.assertTrue(numpy.allclose(result, true_result));


    def test_scientist_4(self):
        with file("scientist-4.sparse",'r') as inp:
            true_result = numpy.array([0.154426, 0.344575]);
            result_matrix = process_dataset(file2dataset(inp));
            result = numpy.mean(numpy.array(result_matrix)[:,2:],0);
            self.assertTrue(numpy.allclose(result, true_result));


    def test_scientist_36(self):
        with file("scientist-36.sparse",'r') as inp:
            true_result = numpy.array([0.197495, 0.370023]);
            result_matrix = process_dataset(file2dataset(inp));
            result = numpy.mean(numpy.array(result_matrix)[:,2:],0);
            self.assertTrue(numpy.allclose(result, true_result));



if __name__ == "__main__":
    #logging.basicConfig(level=logging.DEBUG);
    logging.basicConfig();
    unittest.main();

