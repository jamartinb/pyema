#!/usr/bin/env python
# coding=utf-8
""""""
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

import os

from setuptools import setup, find_packages
#from distutils.core import setup

#print list(os.walk('itaca/samples'))

__version__ = '0.1'

packages = find_packages()
# - The line above replaced the four below
#packages = [root for (root, dirs, files) in os.walk('itacalib')]
#packages = list(set(packages))
#packages.remove('itacalib/adaptor/gnuplot')
#packages.append('itaca')

#files = ['/'.join((d+'/'+f).split('/')[1:])
#        for (d, subdirs, files) in os.walk("hello/samples")
#        for f in files
#        if os.path.isfile(d+'/'+f)];
#files = list(set(files))

#@TODO: ./setuy.py --provides still doesn't work :-( 

setup(name='pyema',
        version=__version__,
        description='Python implementation of EMA: an efficient online '+\
                    'multiclass classifier, based on Exponential Moving '+\
                    'Average updating, which is specially suitable for '+\
                    'non-stationary learning problems',
        author='José Antonio Martín Baena',
        author_email='jose.antonio.martin.baena@gmail.com',
        url='https://github.com/jamartinb/pyema',
        download_url='https://github.com/jamartinb/pyema/tarball/master',
        install_requires=['distribute'],
        provides=["pyema ("+__version__+")"],
        requires=["scipy (>=0.9)"],
        packages = packages,
        package_data = {'tests.pyema':['*.sparse']},
        license="GPLv3",
        test_suite="tests.test_all",
        scripts=["scripts/ema"],
        );
