#!/usr/bin/env python
# coding=utf-8

"""\
Tests the Greenberg's dataset featurization
"""


import unittest;
import logging;

from greenberg import FeaturizeGreenberg;

log = logging.getLogger('greenbergtest');



class GreenbergTest(unittest.TestCase):


    def setUp(self):
        self.enc = FeaturizeGreenberg();
        self.input = """\
S Thu Feb 26 17:23:27 1987
E NIL

C cd workspace
D /home/jamartin
A NIL
H NIL
X NIL

C cd umc
D /home/jamartin/workspace
A NIL
H NIL
X NIL

C ls
D /home/jamartin/workspace/umc
A NIL
H NIL
X NIL

S Thu Feb 26 17:23:27 1987
E NIL

C cd workspace
D /home/jamartin
A NIL
H NIL
X NIL

C cd umc
D /home/jamartin/workspace
A NIL
H NIL
X NIL

C ls
D /home/jamartin/workspace/umc
A NIL
H NIL
X NIL

S Thu Feb 26 17:23:27 1987
E NIL

C cd workspace
D /home/jamartin
A NIL
H NIL
X NIL

C cd umc
D /home/jamartin/workspace
A NIL
H NIL
X NIL
""";
        self.lines = [x+'\n' for x in self.input.split('\n')];


    def tearDown(self):
        self.enc = None;
        self.input = None;


    def test_encode_mockup(self):
        gen = self.enc.get_encoding(self.lines);
        clss, fs = gen.next();
        result = self.enc.encode(session_start=True,current_dir='D /home/jamartin');
        fs.sort(); result.sort();
        self.assertEqual(fs,result);
        result = self.enc.encode("C cd workspace", "D /home/jamartin/workspace",
                "X NIL");
        clss, fs = gen.next();
        fs.sort(); result.sort();
        self.assertEqual(fs,result);
        result = self.enc.encode("C cd umc", "D /home/jamartin/workspace/umc",
                "X NIL", "C cd workspace");
        clss, fs = gen.next();
        fs.sort(); result.sort();
        self.assertEqual(fs,result);
        result = self.enc.encode(session_start=True,current_dir="D /home/jamartin");
        clss, fs = gen.next();
        fs.sort(); result.sort();
        self.assertEqual(fs,result);
        result = self.enc.encode("C cd workspace", "D /home/jamartin/workspace",
                "X NIL");
        clss, fs = gen.next();
        fs.sort(); result.sort();
        self.assertEqual(fs,result);
        result = self.enc.encode("C cd umc", "D /home/jamartin/workspace/umc",
                "X NIL", "C cd workspace");
        clss, fs = gen.next();
        fs.sort(); result.sort();
        self.assertEqual(fs,result);
        




if __name__ == "__main__":
    #logging.basicConfig(level=logging.DEBUG);
    logging.basicConfig();
    unittest.main();

