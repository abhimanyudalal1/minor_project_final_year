import importlib.util
from pathlib import Path
import h5py
import numpy as np
import pytest
from burgers_research.common import file_hash,load_config,matrix
from burgers_research.data import Store
from burgers_research.evaluation import evaluate,evaluate_controls
from burgers_research.training import train,lock_selection

spec=importlib.util.spec_from_file_location('prepare_pdebench',Path('scripts/prepare_pdebench.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_import_and_training(tmp_path):
    p=tmp_path/'source.h5';x=(np.arange(1024)+.5)/1024;t=np.arange(202)*.01
    with h5py.File(p,'w') as f:
        f['x-coordinate']=x;f['t-coordinate']=t
        f['tensor']=np.stack([np.sin(2*np.pi*(x[None,:]-.2*t[:201,None]))*np.exp(-.03*t[:201,None])+i*.1 for i in range(5)]).astype('f4')
    cfg=load_config('configs/pdebench.yaml');cfg['fixture']=True;cfg['data']['counts']={'train':2,'validation':1,'test':2};cfg['data']['shard_size']=2
    cfg['model']['fno'].update(width=4,modes=4,projection=8);cfg['model']['unet']['channels']=[2,4,8,16]
    cfg['training'].update(epochs=1,seeds=[0],batch_size=100,validation_every=1,checkpoint_every_updates=100)
    cfg['runtime'].update(device='cpu',microbatch=20,evaluation_batch=2,cache_training=False)
    cfg['evaluation']['horizons']=[2,4]
    root=tmp_path/'converted'
    module.convert(p,root,cfg,file_hash(p),range(2),[2],[3,4])
    tr=Store(cfg,root,'train'); v=Store(cfg,root,'validation');te=Store(cfg,root,'test')
    assert len(tr.rows)==2 and len(v.rows)==1 and len(te.rows)==2
    assert tr.normalization==v.normalization==te.normalization
    np.testing.assert_allclose(tr.trajectory(0,1024),np.sin(2*np.pi*(x[None,:]-.2*t[:201,None]))*np.exp(-.03*t[:201,None]),atol=1e-7)
    tr.close();v.close();te.close()
    for run in matrix(cfg):train(cfg,root,run)
    lock_selection(cfg,root)
    for run in matrix(cfg):evaluate(cfg,root,run)
    evaluate_controls(cfg,root)
    assert (root/'evaluation/unet-s0/n256-native.json').exists()
    with pytest.raises(ValueError,match='overlapping'):
        module.convert(p,tmp_path/'bad',cfg,file_hash(p),[0,1],[1],[3,4])
