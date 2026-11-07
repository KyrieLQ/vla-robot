"""Bounded formal work units. Parent owns the complete182250-update experiment.

Each child finishes its assigned commit, closes all files and synchronizes device
work before a recorded exit. It never ends the overall experiment early.
"""
import argparse,dataclasses,faulthandler,functools,hashlib,importlib.util,io,json,os,random,resource,signal,sys,time
from pathlib import Path
import flax.nnx as nnx
import jax
import numpy as np
import torch
from openpi.training import sharding
from .pi05_four_suite_v1 import R,make_config
from .pi05_full_data_v1 import FullDataset,epoch_order,validation_noise
from .pi05_compact_state_v1 import restore as restore_legacy,leaf_digest
from .pi05_compact_state_v2 import save_state,load_state,state_payload,write_archive,save_parameters,fsync_directory
from .pi05_formal_io_v1 import write,sha,digest_json,inspect_checkpoint,verify_log_segments

def configuration(plan):
    cfg=make_config(4,4)
    return dataclasses.replace(cfg,name='pi05_four_suite_formal_v1',num_train_steps=182250,save_interval=1000,policy_metadata=dict(scope='approvedthreeepochfour-suiteformal',formal_training=True))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--run',required=True);ap.add_argument('--job',required=True);a=ap.parse_args()
    p=json.loads(Path(a.plan).read_text());run=Path(a.run).absolute();jobfile=Path(a.job).absolute();job=json.loads(jobfile.read_text());out=jobfile.parent;started=time.monotonic();interrupted=[False]
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));faulthandler.enable(all_threads=True)
    def stop(signum,frame):interrupted[0]=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def progress(stage,**kw):write(out/'progress.json',dict(stage=stage,mode=job['mode'],elapsed_seconds=time.monotonic()-started,**kw))
    def finish(report,code=0):
        write(out/'report.json',report);progress('committed' if code==0 else 'interrupted',step=report.get('step'))
        print(json.dumps(report),flush=True);sys.stdout.flush();sys.stderr.flush();os._exit(code)
    try:
        assert jax.default_backend()=='gpu' and jax.device_count()==4
        assert job['mode'] in ['gate','train','validate','finalize'] and p['training']['total_updates']==182250
        jax.config.update('jax_enable_compilation_cache',False)
        np.random.seed(42);random.seed(42);torch.set_num_threads(1);cfg=configuration(p);binding=p['checkpoint_binding']
        spec=importlib.util.spec_from_file_location('formal_upstream_train',R/'repos/openpi/scripts/train.py');upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
        mesh=sharding.make_mesh(4);rep=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec());ds=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        train_rng,init_rng=jax.random.split(jax.random.key(42));progress('initializing_base')
        initial,layout=upstream.init_train_state(cfg,init_rng,mesh,resume=False);jax.block_until_ready(initial)
        assert sum(x.size for x in jax.tree.leaves(initial.params.filter(cfg.trainable_filter)))==52153376
        latest=run/'training/latest.npz';latest.parent.mkdir(exist_ok=True)
        def generate(params,observation,noise):
            model=nnx.merge(initial.model_def,params);model.eval()
            return model.sample_actions(jax.random.key(0),observation,num_steps=10,noise=noise)
        if job['mode']=='gate':
            reference=json.loads((R/p['preflight_run']/'gpu4/initial_parameters.json').read_text());all_hashes=leaf_digest(initial.params);assert all_hashes==reference['all_parameters']
            write(run/'initial_parameters.json',reference)
            data=FullDataset(p,'train');samples=json.loads((R/p['sample_manifest']).read_text())['samples'][:4]
            indices=[data.index.locate(s['episode_id'],s['frame_index']) for s in samples];(obs,actions),rows=data.get_batch(indices)
            cpu_noise=validation_noise(rows)
            write(out/'gate_input_receipts.json',[dict(row,noise_sha256=hashlib.sha256(z.tobytes()).hexdigest()) for row,z in zip(rows,cpu_noise)])
            obs=jax.device_put(obs,ds);noise=jax.device_put(cpu_noise,ds);progress('full_model_generation_exit_gate')
            fn=jax.jit(generate,in_shardings=(layout.params,ds,ds),out_shardings=ds)
            with sharding.set_mesh(mesh):
                lowered=fn.lower(initial.params,obs,noise);compiled=lowered.compile();result=compiled(initial.params,obs,noise)
            jax.block_until_ready(result);output=np.asarray(jax.device_get(result));expected=np.load(R/p['generation_reference_run']/'training_input_generation.npy',mmap_mode='r')[:4]
            np.save(out/'gate_generated.npy',output)
            with sharding.set_mesh(mesh):repeat=compiled(initial.params,obs,noise)
            jax.block_until_ready(repeat);repeat=np.asarray(jax.device_get(repeat));np.save(out/'gate_repeated.npy',repeat)
            write(out/'generation_comparison.json',dict(bitwise_to_reference=bool(np.array_equal(output,expected)),bitwise_repeat=bool(np.array_equal(output,repeat)),max_absolute_error=float(np.max(np.abs(output-expected))),relative_L2=float(np.linalg.norm((output-expected).astype(np.float64))/np.linalg.norm(expected.astype(np.float64))),hlo_sha256=hashlib.sha256(lowered.as_text().encode()).hexdigest(),compile_path='explicit lower().compile() as admitted preflight',persistent_cache=False))
            np.testing.assert_array_equal(output,repeat)
            np.testing.assert_array_equal(output,expected)
            # Nonzero real optimizer state is read from the already audited probe;
            # convert/restore in RAM, with no new optimizer updates or disk copy.
            progress('full_model_nonzero_checkpoint_codec_gate')
            legacy=R/p['preflight_run']/'gpu4/probe_step2.npz';legacy_meta=json.loads(legacy.with_suffix('.json').read_text())
            loaded,key,cursor=restore_legacy(legacy,initial,cfg,legacy_meta['binding'])
            arrays,metadata=state_payload(loaded,cfg,key,dict(probe_cursor=cursor),binding,dict(test_only=True));buffer=io.BytesIO();write_archive(buffer,arrays,metadata)
            rebuilt,key2,meta=load_state(buffer,initial,cfg,binding);jax.block_until_ready(rebuilt)
            assert leaf_digest((loaded.params.filter(cfg.trainable_filter),loaded.opt_state))==leaf_digest((rebuilt.params.filter(cfg.trainable_filter),rebuilt.opt_state))
            np.testing.assert_array_equal(jax.random.key_data(key),jax.random.key_data(key2));assert int(rebuilt.step)==2 and meta['cursor']['probe_cursor']==8
            buffer.close();del arrays,loaded,rebuilt
            assert leaf_digest(initial.params)==all_hashes
            np.random.seed(42);random.seed(42)
            cursor=dict(step=0,epoch=1,frames_in_epoch=0,log_segments=[])
            receipt=save_state(latest,initial,cfg,train_rng,cursor,binding,dict(execution_plan_sha256=sha(Path(a.plan)),mode='formal_initial'))
            jax.block_until_ready(initial)
            finish(dict(status='PASS_GATE',step=0,optimizer_updates=0,new_simulations=0,full_model_output_bitwise=True,nonzero_optimizer_codec_bitwise=True,initial_checkpoint=receipt,exit_strategy='boundedchild_after_synchronizedwork_and_durablecommit',elapsed_seconds=time.monotonic()-started))
        assert latest.exists(),'Formalinitial must passgate first'
        state,train_rng,metadata=load_state(latest,initial,cfg,binding);jax.block_until_ready(state);start_step=int(state.step);assert start_step==job['expected_step']
        assert verify_log_segments(metadata['cursor']['log_segments'])==start_step
        assert metadata['cursor']['step']==start_step
        if start_step:
            e=(start_step-1)//60750+1;_,h=epoch_order(e)
            assert metadata['cursor']['epoch']==e and metadata['cursor']['frames_in_epoch']==(start_step-(e-1)*60750)*4 and metadata['cursor']['epoch_order_sha256']==h
        if job['mode']=='finalize':
            assert start_step in [60750,121500,182250];epoch=start_step//60750;adapter=run/'training'/('epoch%d.npz'%epoch)
            if not adapter.exists():save_parameters(adapter,state,cfg,binding,'epoch%d'%epoch)
            check=inspect_checkpoint(adapter,binding);assert check['step']==start_step
            hashes=leaf_digest(state.params);old=json.loads((run/'initial_parameters.json').read_text());train_keys=set(old['trainable_parameters'])
            assert all(hashes[k]==v for k,v in old['all_parameters'].items() if k not in train_keys)
            write(run/'training'/('epoch%d_frozen_check.json'%epoch),dict(status='PASS',step=start_step,all_frozen_unchanged=True))
            finish(dict(status='PASS_FINALIZE',step=start_step,optimizer_updates=0,elapsed_seconds=time.monotonic()-started))
        if job['mode']=='train':
            end=job['end_step'];assert start_step<end<=182250 and end-start_step<=1000
            epoch=start_step//60750+1;assert end<=epoch*60750;order,orderhash=epoch_order(epoch);data=FullDataset(p,'train')
            fn=jax.jit(functools.partial(upstream.train_step,cfg),in_shardings=(rep,layout,ds),out_shardings=(layout,rep))
            compiled=None
            previous_segments=metadata['cursor']['log_segments'];logfile=out/'updates.jsonl';assert not logfile.exists();log_hasher=hashlib.sha256();rows=0
            with logfile.open('xb') as stream:
                for step in range(start_step,end):
                    progress('training',completed_update=step,end_update=end,epoch=epoch)
                    cursor=(step-(epoch-1)*60750)*4;indices=order[cursor:cursor+4];tick=time.monotonic();raw,receipts=data.get_batch(indices);batch=jax.device_put(raw,ds);io_seconds=time.monotonic()-tick
                    tick=time.monotonic()
                    with sharding.set_mesh(mesh):
                        if compiled is None:compiled=fn.lower(train_rng,state,batch).compile()
                        state,info=compiled(train_rng,state,batch)
                    jax.block_until_ready((state,info));compute_seconds=time.monotonic()-tick;values={k:float(v) for k,v in jax.device_get(info).items()}
                    assert int(state.step)==step+1 and all(np.isfinite(v) for v in values.values())
                    input_hash=hashlib.sha256(''.join(r['model_input_sha256'] for r in receipts).encode()).hexdigest()
                    record=dict(step=step+1,epoch=epoch,input_indices=indices.tolist(),input_sha256=input_hash,loss=values['loss'],grad_norm=values['grad_norm'],compute_seconds=compute_seconds,input_seconds=io_seconds)
                    line=(json.dumps(record,separators=(',',':'))+'\n').encode();stream.write(line);log_hasher.update(line);rows+=1
                    if rows%10==0:stream.flush()
                    if interrupted[0]:break
                stream.flush();os.fsync(stream.fileno());size=stream.tell()
            step=int(state.step);segments=previous_segments+[dict(path=str(logfile.relative_to(R)),bytes=size,sha256=log_hasher.hexdigest(),rows=rows)]
            cursor=dict(step=step,epoch=epoch,frames_in_epoch=(step-(epoch-1)*60750)*4,epoch_order_sha256=orderhash,log_segments=segments)
            progress('committing_checkpoint',step=step);receipt=save_state(latest,state,cfg,train_rng,cursor,binding,dict(execution_plan_sha256=sha(Path(a.plan)),job_sha256=sha(jobfile)))
            if step%60750==0:
                adapter=run/'training'/('epoch%d.npz'%(step//60750))
                if not adapter.exists():save_parameters(adapter,state,cfg,binding,'epoch%d'%(step//60750))
                frozen=leaf_digest(state.params);initial_hashes=json.loads((run/'initial_parameters.json').read_text());trained=set(initial_hashes['trainable_parameters']);assert all(frozen[k]==h for k,h in initial_hashes['all_parameters'].items() if k not in trained)
                write(run/'training'/('epoch%d_frozen_check.json'%(step//60750)),dict(status='PASS',step=step,all_frozen_unchanged=True))
            jax.block_until_ready(state)
            finish(dict(status='PASS_TRAIN_SEGMENT',start_step=start_step,step=step,updates=rows,checkpoint=receipt,interrupted=interrupted[0],elapsed_seconds=time.monotonic()-started),130 if interrupted[0] else 0)
        # Validation uses only observation + private fixed noise; actions/end masks
        # remain on the CPU and never enter the generation function.
        data=FullDataset(p,'validation');point=job['point'];folder=run/'validation'/point;folder.mkdir(parents=True,exist_ok=True)
        progressfile=folder/'progress.json';final=folder/'predictions.npy';partial=folder/'predictions.partial.npy';existing=json.loads(progressfile.read_text()) if progressfile.exists() else dict(committed_rows=0,chunks=[],point=point,step=start_step,binding=binding)
        assert existing['step']==start_step and existing['point']==point and existing['binding']==binding
        if final.exists():assert existing['committed_rows']==30465;finish(dict(status='PASS_VALIDATION_SEGMENT',point=point,step=start_step,committed_rows=30465,new_rows=0,elapsed_seconds=time.monotonic()-started))
        array=np.lib.format.open_memmap(partial,mode='r+' if partial.exists() else 'w+',dtype=np.float32,shape=(30465,10,32))
        for chunk in existing['chunks']:assert hashlib.sha256(array[chunk['start']:chunk['end']].tobytes()).hexdigest()==chunk['sha256']
        begin=existing['committed_rows'];assert begin==job['start_row'];end=min(30465,begin+2048);commit_start=begin
        fn=jax.jit(generate,in_shardings=(layout.params,ds,ds),out_shardings=ds)
        compiled=None
        for start in range(begin,end,4):
            valid=min(4,30465-start);indices=np.minimum(np.arange(start,start+4),30464);(obs,unused_actions),rows=data.get_batch(indices)
            observation=jax.device_put(obs,ds);noise=jax.device_put(validation_noise(rows),ds)
            with sharding.set_mesh(mesh):
                if compiled is None:compiled=fn.lower(state.params,observation,noise).compile()
                pred=compiled(state.params,observation,noise)
            jax.block_until_ready(pred);values=np.asarray(jax.device_get(pred));assert values.shape==(4,10,32) and np.isfinite(values).all();array[start:start+valid]=values[:valid]
            committed=start+valid
            if committed-commit_start>=128 or committed==end or interrupted[0]:
                array.flush()
                with partial.open('rb') as f:os.fsync(f.fileno())
                existing['chunks'].append(dict(start=commit_start,end=committed,sha256=hashlib.sha256(array[commit_start:committed].tobytes()).hexdigest()));existing['committed_rows']=committed;write(progressfile,existing);commit_start=committed
            progress('validation',point=point,completed=committed,total=30465)
            if interrupted[0]:break
        array.flush();del array
        if existing['committed_rows']==30465:os.replace(partial,final);fsync_directory(folder)
        jax.block_until_ready(state)
        finish(dict(status='PASS_VALIDATION_SEGMENT',point=point,step=start_step,committed_rows=existing['committed_rows'],new_rows=existing['committed_rows']-begin,interrupted=interrupted[0],elapsed_seconds=time.monotonic()-started),130 if interrupted[0] else 0)
    except BaseException as exc:
        import traceback
        traceback.print_exc();write(out/'failure.json',dict(status='FAILED',error=type(exc).__name__+': '+str(exc),elapsed_seconds=time.monotonic()-started));sys.stdout.flush();sys.stderr.flush();os._exit(1)

if __name__=='__main__':main()
