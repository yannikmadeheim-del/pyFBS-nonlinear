import unittest
import subprocess
from pathlib import Path
import os

# Testing for the notebooks - use nbconvert to execute all cells of the
# notebook

# For testing on TravisCI, be sure to include a requirements.txt that
# includes jupyter so that you run on the most up-to-date version.


TESTDIR = os.path.abspath(__file__) # absolutepath of this file
NBDIR = str(Path(__file__).parents[1] ) + os.sep + "examples" + os.sep # where are the notebooks?

def setUp():
    # list of notebooks, with file paths
    nbpaths = []
    # list of notebook names (for making the tests)
    nbnames = []
    # walk the test directory and find all notebooks
    for dirname, dirnames, filenames in os.walk(NBDIR):
        for filename in filenames:
            if filename.endswith('.ipynb') and not filename.endswith('-checkpoint.ipynb'):
                nbpaths.append(os.path.abspath(dirname) + os.path.sep + filename) # get abspath of notebook
                nbnames.append(''.join(filename[:-6])) # strip off the file extension
    return nbpaths, nbnames


def get(nbname, nbpath):
    # use nbconvert to execute the notebook
    def test_func(self):
        print('\n--------------- Testing {0} ---------------'.format(nbname))
        print('   {0}'.format(nbpath))
        # execute the notebook using nbconvert to generate html
        #print("gg", nbpath)
        #nbexe = subprocess.Popen(['jupyter', 'nbconvert', '{0}'.format(nbpath),
        #                          '--execute',
        #                          '--ExecutePreprocessor.timeout=120'],
        #                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        #                         stderr=subprocess.PIPE)

        nbexe = subprocess.Popen(['jupyter', 'nbconvert','--to','notebook',
                                  '--execute', '{0}'.format(nbpath),'--ExecutePreprocessor.timeout=120'],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)

        output, err = nbexe.communicate()
        check = nbexe.returncode
        if check == 0:
            print('\n ..... {0} Passed ..... \n'.format(nbname))

            # if passed remove the generated  file
            os.remove(nbpath[:-6] + '.nbconvert.ipynb')
        else:
            print('\n <<<<< {0} FAILED >>>>> \n'.format(nbname))
            print('Captured Output: \n {0}'.format(err))

        self.assertTrue(check == 0)

    return test_func



attrs = dict()
nbpaths, nbnames = setUp()

# build test for each notebook
for i, nb in enumerate(nbnames):
    attrs['test_'+nb] = get(nb, nbpaths[i])

# create class to unit test notebooks
TestNotebooks = type('TestNotebooks', (unittest.TestCase,), attrs)


if __name__ == '__main__':
    unittest.main()