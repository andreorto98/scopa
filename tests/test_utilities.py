'''Test for the utilities.py file
'''

import unittest

from scopa.utilities import ssquare
#export PYTHONPATH=/Users/andrea/Desktop/computing_methods/smartsquare/  per farlo funzionare...


class TestCore(unittest.TestCase):      # classes con CamelCase
    def test_float(self):                   # Metodi/Funzioni con '_'
        self.assertAlmostEqual(ssquare(2.), 4.)



if __name__ == '__main__':
    unittest.main()
