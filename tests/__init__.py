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

__all__ = ["greenberg","pyema"]



def test_all():
    import os
    import unittest
    # Get the current folder
    here = os.path.dirname(__file__)

    # Get the path to tests files
    testfiles = [os.path.join(d,f) for (d, subdirs, files) in os.walk(here)
                                     for f in files
                                     if os.path.isfile(os.path.join(d,f))
                                     and f.endswith(".py")
                                     and "__init__.py" != f
                                     # This is automatically created, why?
                                     and "dststest.py" != f]

    # Translate paths to dotted modules
    # @TODO: I'm sure this can be done better
    modules = ['.'.join(f.split('.py')[0].split(os.path.sep))
                for f in testfiles]

    # travis-ci.org does not set __package__
    package = __package__+'.' if __package__ else 'tests.'
    testmodules = None
    if modules and package in modules[0]:
        testmodules = [package+(m.split(package)[1]) for m in modules]
    else:
        raise Exception("Couldn't load test files")

    suite = unittest.TestSuite()

    for t in testmodules:
        try:
            # If the module defines a suite() function, call it to get the suite.
            mod = __import__(t, globals(), locals(), ['suite'])
            suitefn = getattr(mod, 'suite')
            suite.addTest(suitefn())
        except (ImportError, AttributeError):
            # else, just load all the test cases from the module.
            # It only allows the "dotted" name
            try:
                # Try to load from name
                suite.addTest(unittest.defaultTestLoader.loadTestsFromName(t))
            except Exception:
                try:
                    # Try to load from module
                    mod = __import__(t, globals(), locals())
                    suite.addTest(unittest.defaultTestLoader.loadTestsFromModule(mod))
                except Exception as e:
                    # Check what went wrong
                    # Let's show the content of the module
                    module_content = "UNKNOWN"
                    try:
                        mod = __import__('.'.join(t.split('.')[:-1]),
                                         globals(), locals())
                        module_content = getattr(
                                         getattr(mod,t.split('.')[-2]),'__all__')
                    except:
                        pass

                    raise Exception("Couldn't load test module: "+repr(t),
                                    module_content, e)

    return suite

