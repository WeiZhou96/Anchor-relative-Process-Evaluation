"""Check table completeness and create a PNG contact sheet for visual review."""
from pathlib import Path
import csv,json
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/r2b'
rows=[]
for h in ['H4p00','H10p00','H21p50']:
 for prefix,n in [('table1_audit',184),('table1b_audit_by_arm_rule',56),('table2_cohort_definitions',184),('table5_phenomena',184)]:
  p=OUT/'tables'/f'{prefix}_{h}.csv'
  if not p.exists():
   candidates=list((OUT/'tables').glob(prefix+'*'+h+'*.csv'))
   raise AssertionError((str(p),candidates))
  raw=list(csv.reader(p.open()))
  assert all(len(x)==len(raw[0]) for x in raw),p
  data=list(csv.DictReader(p.open()))
  assert len(data)==n,(p,len(data))
  if prefix=='table1_audit':
   assert all(k in data[0] for k in ['arm_rule','backbone','train_data_unknown'])
  if prefix=='table2_cohort_definitions':
   assert all(k in data[0] for k in ['G3 original consequence','G3 revised consequence'])
  rows.append(dict(path=str(p.relative_to(OUT)),n=len(data)))
figs=sorted((OUT/'figs').glob('*.png'));assert len(figs)==8
assert len(list((OUT/'figs').glob('*.pdf')))==8
assert len(list((OUT/'figs').glob('*.svg')))==4
canvas=Image.new('RGB',(2200,2400),'white');draw=ImageDraw.Draw(canvas)
images=[]
for i,p in enumerate(figs):
 im=Image.open(p).convert('RGB');assert min(im.size)>500
 images.append(dict(file=p.name,width=im.width,height=im.height))
 thumb=ImageOps.contain(im,(1080,550))
 x=(i%2)*1100+(1100-thumb.width)//2;y=(i//2)*600+30
 canvas.paste(thumb,(x,y));draw.text(((i%2)*1100+15,(i//2)*600+8),p.stem,fill='black')
qa=OUT/'qa';qa.mkdir(exist_ok=True);canvas.save(qa/'contact.png')
(qa/'structure.json').write_text(json.dumps(dict(tables=rows,figures=images,visual_review='pending'),indent=2))
print('table structure and 8 PNG / 8 PDF / 4 SVG verified; contact sheet ready')
