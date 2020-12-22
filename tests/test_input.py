'''
test for the core.py file

'''

import unittest

from scopa.input import ssquare
#export PYTHONPATH=/Users/andrea/Desktop/computing_methods/smartsquare/  per farlo funzionare...


class TestCore(unittest.TestCase):      # classes con CamelCase
    ''' Unittest for core module '''
    def test_float(self):                   # Metodi/Funzioni con '_'
        '''test'''
        self.assertAlmostEqual(ssquare(2.), 4.)

if __name__ == '__main__':
    unittest.main()
