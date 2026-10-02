"""Check submission manifest, clean-execution source identity, and blind identifiers."""
import gzip,hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent;S=ROOT/'outputs/submission'
def run():
    blocked=['pwqppt','joo-hyun','C:/Users','C:\\Users','@gmail.com','@naver.com']
    checked=0;source=[]
    with zipfile.ZipFile(S/'research_source.zip') as z:
        manifest=json.loads(z.read('FILE_MANIFEST.json'));assert z.testzip() is None
        assert set(z.namelist())==set(manifest)|{'FILE_MANIFEST.json'}
        for name,sha in manifest.items():
            b=z.read(name);assert hashlib.sha256(b).hexdigest()==sha
            if name.endswith('.gz'):b=gzip.decompress(b)
            if Path(name).suffix in ['.csv','.json','.py','.md','.ipynb','.txt','.gz']:
                t=b.decode('utf8');assert not any(term.lower() in t.lower() for term in blocked),name
                checked+=1
            if '/' not in name and name.endswith('.py'):
                assert z.read(name)==(ROOT/'work/reproduce-check'/name).read_bytes(),name
                assert z.read(name)==(ROOT/name).read_bytes(),name
                source.append(name)
        nb=json.loads(z.read('reproduce_executed.ipynb'))
        assert all(c['execution_count'] is not None for c in nb['cells'] if c['cell_type']=='code')
        assert not any(o['output_type']=='error' for c in nb['cells'] if c['cell_type']=='code' for o in c['outputs'])
    result={'manifest_file_count':len(manifest),'text_files_blind_scanned':checked,
      'executed_source_files_identical_to_final':source,'executed_notebook_included':True,
      'zip_integrity_passed':True,'blind_identifier_scan_passed':True,
      'zip_sha256':hashlib.sha256((S/'research_source.zip').read_bytes()).hexdigest(),
      'scope':'Known account names/local-user paths/email identifiers; official public institutions are source citations, not participant affiliations.'}
    (S/'package_verification.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
if __name__=='__main__':run()
