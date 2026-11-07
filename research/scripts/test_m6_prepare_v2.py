"""Small isolated CPU tests of integrity, resume and suite-ID cache mapping."""
import hashlib,io,json,tempfile,threading,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from mini_vla import prepare_m6_v2 as m

class PrepareTests(unittest.TestCase):
    def test_existing_integrity_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp') as td:
            base=Path(td);data=b'exact';(base/'x.parquet').write_bytes(data)
            row=dict(rfilename='x.parquet',size=len(data),lfs={'sha256':hashlib.sha256(data).hexdigest()})
            with patch.object(m.urllib.request,'urlopen',side_effect=AssertionError('No network expected')):
                self.assertFalse(m.download_one(row,base,'rev',threading.Event())['downloaded'])
                row['lfs']['sha256']='bad'
                with self.assertRaises(ValueError):m.download_one(row,base,'rev',threading.Event())
                self.assertEqual((base/'x.parquet').read_bytes(),data)

    def test_http_resume_exact_hash(self):
        class Response(io.BytesIO):
            status=206
            headers={'Content-Range':'bytes 3-5/6'}
        with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp') as td:
            base=Path(td);(base/'x.parquet.part').write_bytes(b'abc');row=dict(rfilename='x.parquet',size=6,lfs={'sha256':hashlib.sha256(b'abcdef').hexdigest()})
            with patch.object(m.urllib.request,'urlopen',return_value=Response(b'def')):
                result=m.download_one(row,base,'rev',threading.Event())
            self.assertTrue(result['downloaded']);self.assertEqual((base/'x.parquet').read_bytes(),b'abcdef');self.assertFalse((base/'x.parquet.part').exists())

    def test_cache_resume_and_global_task_mapping(self):
        with tempfile.TemporaryDirectory(dir=m.ROOT/'tmp') as td,patch.object(m,'ROOT',Path(td)):
            root=Path(td);(root/'source/data/chunk-000').mkdir(parents=True);(root/'out').mkdir()
            pre={'vocabulary':{'<PAD>':0,'<UNK>':1,'put':2,'bowl':3},'normalization':{k:{'mean':[0.]*n,'scale':[1.]*n} for k,n in [('state',8),('actions',7)]}}
            split={'train':[dict(episode_id=0,length=2,instruction='put bowl',suite='libero_spatial',task_id=0)],'validation':[dict(episode_id=1,length=1,instruction='put bowl',suite='libero_10',task_id=1)]}
            (root/'split.json').write_text(json.dumps(split));(root/'pre.json').write_text(json.dumps(pre))
            for eid,n in [(0,2),(1,1)]:
                rows=[]
                for i in range(n):
                    buf=io.BytesIO();Image.fromarray(np.full((16,16,3),eid*80+i,dtype='uint8')).save(buf,format='PNG');cell={'bytes':buf.getvalue()}
                    rows.append(dict(image=cell,wrist_image=cell,state=[.1]*8,actions=[.2]*7,episode_index=eid,frame_index=i))
                pq.write_table(pa.Table.from_pylist(rows),root/f'source/data/chunk-000/episode_{eid:06d}.parquet')
            cfg=dict(dataset_root='source',cache='cache',split='split.json',preprocessing='pre.json')
            m.cache(cfg,root/'out',split,pre)
            self.assertEqual(np.load(root/'cache/validation/task_id.npy').tolist(),[31])
            before=m.sha(root/'cache/train/images.npy');(root/'cache/complete.json').unlink()
            # Completed episodes must be reused without decoding them again.
            with patch.object(m.pq,'read_table',side_effect=AssertionError('Unexpected redecode')):m.cache(cfg,root/'out',split,pre)
            self.assertEqual(m.sha(root/'cache/train/images.npy'),before)

if __name__=='__main__':unittest.main()
