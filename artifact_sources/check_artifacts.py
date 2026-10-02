from pathlib import Path
import json,hashlib,zipfile
import numpy as np
from PIL import Image
from pypdf import PdfReader
root=Path(__file__).resolve().parents[1];s=root/'outputs/submission'
receipt=json.loads((root/'work/deck/validation-reviewed.json').read_text())
assert receipt['packageIntegrity']['findingCount']==0
assert receipt['presentationLayout']['findingCount']==0
unchanged=[];diff=[]
slide=PdfReader(s/'presentation_reviewed.pdf')
for i in range(1,14):
    name=f'slide-{i:02}.png';a=root/'work/deck/rendered'/name;b=root/'work/deck/reviewed'/name
    if a.read_bytes()==b.read_bytes():unchanged.append(i)
    x=np.asarray(Image.open(b).convert('RGB')).astype(float)
    y=np.asarray(Image.open(root/'work/slide-pdf-review'/name).convert('RGB')).astype(float)
    assert x.shape==y.shape
    mae=float(np.abs(x-y).mean());diff.append(mae)
    embedded=np.asarray(slide.pages[i-1].images[0].image.convert('RGB')).astype(float)
    assert np.array_equal(x,embedded)  # PDF image data unchanged; rasterizer antialiasing differs.
report=PdfReader(s/'result_report_reviewed.pdf');slide=PdfReader(s/'presentation_reviewed.pdf')
assert len(report.pages)==18 and len(slide.pages)==13
texts='\n'.join(p.extract_text() for p in report.pages)
assert '제6장' in texts and '176' in texts and '17.483' in texts
blocked=['pwqppt','joo-hyun','C:/Users','C:\\Users','@gmail.com','@naver.com']
for t in [texts,str(report.metadata),str(slide.metadata)]:
    assert not any(b.lower() in t.lower() for b in blocked)
with zipfile.ZipFile(s/'presentation_reviewed.pptx') as z:
    xml='\n'.join(z.read(n).decode('utf8') for n in z.namelist() if n.endswith('.xml'))
    assert not any(b.lower() in xml.lower() for b in blocked)
q={'report_pages':18,'slides':13,'all_report_pages_visually_reviewed':True,'all_deck_slides_visually_reviewed':True,
   'unchanged_slide_renders_from_first_review':unchanged,'changed_slide_6_reviewed':True,
   'pdf_slide_embedded_pixels_exactly_equal':True,'pdf_slide_render_pixel_mae_max':max(diff),
   'rasterizer_difference':'Poppler resampling/antialiasing; source PDF image pixels exactly match PPTX renders.',
   'report_korean_legible':True,'no_observed_overlap_or_clipping':True,
   'presentation_package_findings':0,'presentation_geometry_findings':0,'native_chart_count':2,
   'native_chart_workbook_cache_verified':True,'native_powerpoint_runtime_verified':False,
   'blind_identifiers_scan_passed':True,'font_exception':'Official Human Myeongjo absent; report uses HY Sin Myeongjo 14pt,160% leading.',
   'remaining_formal_requirements':['Real participant information/signatures','Actual survey completion screenshot','Official font finalization'],
   'files':{n:hashlib.sha256((s/n).read_bytes()).hexdigest() for n in ['result_report_reviewed.pdf','presentation_reviewed.pptx','presentation_reviewed.pdf']}}
(s/'quality_verification.json').write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(q,ensure_ascii=False,indent=2))
