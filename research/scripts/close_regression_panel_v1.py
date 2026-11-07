"""Second terminal-log recount, fixed-pair comparison and representative video sheets."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[1]
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):path.write_text(json.dumps(value,indent=2)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();run=Path(a.run).absolute();began=time.monotonic()
    p=json.loads((run/'plan.json').read_text());panel=json.loads((R/p['panel']).read_text());accepted=json.loads((run/'review/independent_acceptance.json').read_text())
    assert accepted['status']=='PASS' and accepted['episodes_used']==600 and accepted['full_decoded_videos']==120
    rows=[];by={};video_count=0;physical=policycalls=0
    for case in panel['cases']:
        key='%s/task%d_init%d'%(case['suite'],case['task_id'],case['initial_state_index'])
        for arm in p['arms']:
            directory=run/arm/key;assert directory.is_symlink();record=json.loads((directory/'episode.json').read_text())
            assert record['plan_sha256']==digest(run/'plan.json') and record['checkpoint_sha256']==p['arms'][arm]['checkpoint_sha256']
            assert record['case']==case and record['pair_id']==key
            journal=directory/'trajectory.jsonl';assert digest(journal)==record['journal_sha256'];steps=calls=0;last=None
            with journal.open() as stream:
                for raw in stream:
                    item=json.loads(raw)
                    if item['kind']=='step':
                        assert item['step']==steps and (last is None or not last['done']);steps+=1;last=item
                    else:assert item['kind']=='call' and item['step']==steps and steps%5==0;calls+=1
            success=bool(last['done']);assert success==record['task_success'] and steps==record['steps'] and calls==record['policy_calls']
            assert success or steps==panel['max_action_steps'][case['suite']]
            with np.load(directory/'trace.npz') as trace:assert bool(trace['done'][-1])==success and len(trace['done'])==steps
            if record['video_selected']:assert digest(directory/'external.mp4')==record['video_sha256'];video_count+=1
            rows.append(dict(arm=arm,suite=case['suite'],task=case['task_id'],pair_id=key,success=success,steps=steps,policy_calls=calls,video=record['video_selected']))
            by[(arm,key)]=rows[-1];physical+=steps;policycalls+=calls
    assert len(rows)==600 and video_count==120 and physical==accepted['total_physical_steps'] and policycalls==accepted['total_policy_calls']
    totals={}
    for arm in p['arms']:
        totals[arm]={suite:sum(row['success'] for row in rows if row['arm']==arm and row['suite']==suite) for suite in panel['suites']}
        assert sum(totals[arm].values())==accepted['results'][arm]['successes']
        for suite,n in totals[arm].items():assert n==accepted['results'][arm]['by_suite'][suite]['successes']
    comparisons={}
    for candidate,control in [('depth2','depth1'),('depth1','existing'),('depth2','existing')]:
        rescued=lost=0
        for case in panel['cases']:
            key='%s/task%d_init%d'%(case['suite'],case['task_id'],case['initial_state_index']);x=by[(candidate,key)]['success'];y=by[(control,key)]['success'];rescued+=x and not y;lost+=y and not x
        comparisons[candidate+'_vs_'+control]=dict(rescued=rescued,new_failures=lost)
        assert rescued==accepted['paired'][candidate+'_vs_'+control]['rescued'] and lost==accepted['paired'][candidate+'_vs_'+control]['new_failures']
    # Outcome-based selection is for qualitative review only, after all results are frozen.
    selected=[];sheets=[]
    for suite in panel['suites']:
        keys=[row['pair_id'] for row in rows if row['arm']=='existing' and row['suite']==suite and row['video']]
        choices=[]
        for direction in [(True,False),(False,True),(True,True),(False,False)]:
            matches=[key for key in keys if key not in choices and (by[('existing',key)]['success'],by[('depth2',key)]['success'])==direction]
            if matches:choices.append(matches[0])
            if len(choices)==2:break
        assert len(choices)==2
        canvas=Image.new('RGB',(6*160,42+6*184),'white');draw=ImageDraw.Draw(canvas);draw.text((8,8),suite+' | diagnostic frames; not model selection',fill='black')
        for i,(key,arm) in enumerate((key,arm) for key in choices for arm in ['existing','depth1','depth2']):
            row=by[(arm,key)];video=run/arm/key/'external.mp4';reader=imageio.get_reader(video);indices=np.linspace(0,row['steps'],6,dtype=int)
            images=[]
            try:
                for index in indices:images.append(Image.fromarray(reader.get_data(int(index))).resize((160,160)))
            finally:reader.close()
            y=42+i*184;draw.text((8,y),key.split('/')[-1]+' / '+arm+' / '+('SUCCESS' if row['success'] else 'FAIL')+' / '+str(row['steps'])+' steps',fill='black')
            for j,image in enumerate(images):canvas.paste(image,(j*160,y+20))
            selected.append(dict(arm=arm,pair_id=key,success=row['success'],video_sha256=digest(video),frame_indices=indices.tolist()))
        file=run/'review'/('video_sheet_'+suite+'.jpg');canvas.save(file,quality=88);sheets.append(str(file.relative_to(R)))
    result=dict(status='PASS',raw_terminal_outcomes_recounted=600,full_suite_task_coverage=40,totals=totals,paired=comparisons,
                videos_rehashed=video_count,full_video_decodes_already_verified=120,representative_clips=selected,sheets=sheets,
                physical_steps=physical,policy_calls=policycalls,new_simulations=0,new_training_updates=0,elapsed_seconds=time.monotonic()-began)
    dump(run/'review/terminal_recount_and_visual_manifest.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['representative_clips','sheets']}))

if __name__=='__main__':main()
