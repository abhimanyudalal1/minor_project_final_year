"""Convert checksum-verified PDEBench data to a separate, auditable pipeline dataset."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from scipy.signal import resample
from burgers_research.common import data_signature, digest, file_hash, load_config, write_json


def convert(source, root, cfg, expected_sha, train, validation, test):
    source, root = Path(source), Path(root)
    if file_hash(source) != expected_sha:
        raise ValueError('Source SHA256 does not match verified archive')
    d=cfg['data']; counts=d['counts']
    split={'train':list(train),'validation':list(validation),'test':list(test)}
    all_indices=sum(split.values(),[])
    if len(all_indices)!=len(set(all_indices)) or any(len(split[k])!=counts[k] for k in split):
        raise ValueError('Invalid or overlapping trajectory splits')
    folder=root/'data'; folder.mkdir(parents=True,exist_ok=True)
    if (folder/'manifest.json').exists():
        raise ValueError('Dataset already exists; use a fresh work directory')
    signature=data_signature(cfg); shards=[]; total=squares=0.; elements=0
    with h5py.File(source,'r') as f:
        data=f['tensor']; x=f['x-coordinate'][:];t=f['t-coordinate'][:]
        if data.ndim!=3 or data.shape[1:]!=(201,1024) or data.shape[0]<=max(all_indices):
            raise ValueError('Unexpected source tensor shape')
        if len(t)!=202 or not np.allclose(t[:201],np.arange(201)*.01,atol=3e-7,rtol=0):
            raise ValueError('Unexpected source time coordinates')
        if not np.allclose(x,(np.arange(1024)+.5)/1024,atol=1e-7,rtol=0):
            raise ValueError('Unexpected source grid')
        rows=[]
        for part,indices in split.items():
            for start in range(0,len(indices),d['shard_size']):
                ids=indices[start:start+d['shard_size']]; name=f'{part}-{start//d["shard_size"]:04d}.h5'; path=folder/name
                if path.exists(): raise ValueError(f'Refusing to overwrite {path}')
                temp=path.with_suffix('.partial')
                labels=[f'{part}-{idx:05d}' for idx in ids]
                with h5py.File(temp,'w') as out:
                    out.attrs['data_signature']=signature;out.attrs['viscosity']=d['viscosity']
                    out.create_dataset('times',data=np.arange(201)*.01)
                    out.create_dataset('ids',data=np.array(labels,dtype='S40'))
                    out.create_dataset('seeds',data=np.array(ids,dtype=np.int64))
                    out.create_dataset('x_reference',data=x)
                    ref=out.create_dataset('reference',(len(ids),201,1024),dtype='f8',chunks=(1,1,1024),compression='lzf')
                    grids={}
                    for n in d['resolutions']:
                        out.create_dataset(f'x_{n}',data=x[0]+np.arange(n)/n)
                        grids[n]=out.create_dataset(f'u_{n}',(len(ids),201,n),dtype='f4',chunks=(1,16,n),compression='lzf')
                    for j,idx in enumerate(ids):
                        a=data[idx,:,:]
                        if not np.isfinite(a).all(): raise ValueError(f'Nonfinite source trajectory {idx}')
                        ref[j]=a
                        for n,target in grids.items():
                            value=a if n==1024 else resample(a,n,axis=-1).astype(np.float32)
                            target[j]=value
                            if part=='train' and n==d['base_n']:
                                z=value[:101].astype(np.float64);total+=float(z.sum());squares+=float((z*z).sum());elements+=z.size
                        rows.append({'id':labels[j],'split':part,'source_index':int(idx)})
                temp.replace(path)
                shards.append({'file':name,'sha256':file_hash(path),'data_signature':signature,'split':part,'ids':labels,'source_indices':ids})
                print('Prepared',name,flush=True)
    mean=total/elements
    norm={'mean':mean,'std':float(np.sqrt(squares/elements-mean**2)),'count':elements,'split':'train','resolution':d['base_n'],'end_time':d['train_time']}
    manifest={'data_signature':signature,'fixture':cfg['fixture'],'config':cfg,'shards':shards,'normalization':norm,
      'source':{'dataset':'PDEBench Burgers DaRUS 8.0','sha256':expected_sha,'path':str(source.resolve()),
                'reference':'source numerical solution at N=1024; not an 8192-point reference',
                'time':'first 201 coordinates; final extra coordinate has no solution snapshot',
                'grid':'source cell centres; Fourier-resampled grids share source first centre',
                'effective_viscosity':'0.001/pi, conditional on upstream generator version'}}
    manifest['dataset_hash']=digest({'signature':signature,'normalization':norm,'shards':{s['file']:s['sha256'] for s in shards}})
    write_json(folder/'splits.json',{'data_signature':signature,'trajectories':rows})
    write_json(folder/'manifest.json',manifest)
    # Verify derived shards, IDs, all values and checksums before enabling Store.
    seen=[]
    for shard in shards:
        path=folder/shard['file']
        if file_hash(path)!=shard['sha256']:raise ValueError('Shard hash mismatch')
        with h5py.File(path) as f:
            seen += [v.decode() for v in f['ids'][:]]
            for key in ['reference']+[f'u_{n}' for n in d['resolutions']]:
                for i in range(len(f[key])):
                    if not np.isfinite(f[key][i]).all():raise ValueError('Nonfinite converted data')
    if seen != [r['id'] for r in rows]:raise ValueError('Split/manifest mismatch')
    write_json(root/'verification/data.json',{'passed':True,'dataset_hash':manifest['dataset_hash'],
        'manifest_sha256':file_hash(folder/'manifest.json'),'trajectories':len(seen),'method':'PDEBench conversion validation; not solver verification'})
    print('Verified converted dataset',manifest['dataset_hash'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='data/pdebench/1D_Burgers_Sols_Nu0.001.hdf5')
    p.add_argument('--root',default='artifacts/pdebench');p.add_argument('--config',default='configs/pdebench.yaml');args=p.parse_args()
    # Keep the pilot's held-out IDs fixed when expanding the training prefix.
    cfg = load_config(args.config)
    count = cfg['data']['counts']['train']
    if not 0 < count <= 8750:
        raise ValueError('Training prefix would overlap the held-out splits')
    convert(args.source,args.root,cfg,'afeeed1c40ce01d2ba5e1702f4f66b6bf95c65943eb0e4847691e9a10d1cb50d',range(count),range(8750,9000),range(9000,10000,2))
