"""Export the verified final slide renders as a lossless visual PDF."""
import argparse
from pathlib import Path
from reportlab.pdfgen.canvas import Canvas
from PIL import Image

def run(images,output):
    files=sorted(images.glob('slide-*.png'));assert len(files)==13
    c=Canvas(str(output),pagesize=(960,540));c.setTitle('전력 예측과 일최대 위험 분석 발표자료');c.setAuthor('');c.setCreator('')
    for f in files:
        w,h=Image.open(f).size;assert w/h==16/9
        c.drawImage(str(f),0,0,width=960,height=540);c.showPage()
    c.save()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('images',type=Path);p.add_argument('output',type=Path);a=p.parse_args();run(a.images,a.output)
