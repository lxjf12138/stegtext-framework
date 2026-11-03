import math, unittest
from stegtext.starter.adaptive import select_kernel
from stegtext.starter.provenance import derive_domain_seed, stable_json_sha256

class StarterContracts(unittest.TestCase):
    def test_gate_endpoints(self):
        self.assertEqual(select_kernel(0.0, 0.0), 'las')
        self.assertEqual(select_kernel(100.0, math.inf), 'fullsync')
        self.assertEqual(select_kernel(0.5, 0.5), 'las')
        self.assertEqual(select_kernel(0.499, 0.5), 'fullsync')
    def test_domain_separation(self):
        k=b'case-key'
        self.assertNotEqual(derive_domain_seed(k,'coder',1), derive_domain_seed(k,'advance',1))
        self.assertEqual(derive_domain_seed(k,'coder',1), derive_domain_seed(k,'coder',1))
    def test_stable_hash(self):
        self.assertEqual(stable_json_sha256({'b':2,'a':1}), stable_json_sha256({'a':1,'b':2}))
if __name__ == '__main__': unittest.main()
