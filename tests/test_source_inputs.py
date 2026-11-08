"""Regression checks for filtered 10x matrix extraction."""
import gzip
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('source_inputs', ROOT / 'scripts/source_pipeline/source_inputs.py')
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


class FilteredMatrixExtraction(unittest.TestCase):
    def test_zip_retains_filtered_matrix_when_raw_members_share_names(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            package = base / 'counts.zip'
            files = {name: gzip.compress(('filtered-' + name).encode(), mtime=0)
                     for name in ('matrix.mtx.gz', 'barcodes.tsv.gz', 'features.tsv.gz')}
            with zipfile.ZipFile(package, 'w') as archive:
                for name, value in files.items():
                    archive.writestr('filtered_feature_bc_matrix/' + name, value)
                    archive.writestr('raw_feature_bc_matrix/' + name, gzip.compress(('raw-' + name).encode(), mtime=0))
            expected = [{'name': 'sample_' + name, 'size_bytes': len(value),
                         'sha256': hashlib.sha256(value).hexdigest()} for name, value in files.items()]
            source.extract(package, {'sample_id': 'sample', 'source_files': expected}, base / 'output')
            for name, value in files.items():
                self.assertEqual((base / 'output' / ('sample_' + name)).read_bytes(), value)

    def test_conflicting_filtered_members_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            package = base / 'counts.zip'
            with zipfile.ZipFile(package, 'w') as archive:
                archive.writestr('one/filtered_feature_bc_matrix/matrix.mtx.gz', b'first')
                archive.writestr('two/filtered_feature_bc_matrix/matrix.mtx.gz', b'second')
            with self.assertRaisesRegex(ValueError, 'Multiple different archive members'):
                source.extract(package, {'sample_id': 'sample', 'source_files': []}, base / 'output')


if __name__ == '__main__':
    unittest.main()
