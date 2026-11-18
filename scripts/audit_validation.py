"""Read-only checkpoint diagnostics on every validation trajectory; no test access."""
from pathlib import Path
import numpy as np
import torch
from burgers_research.common import load_config, file_hash, source_hash, write_json
from burgers_research.data import Store
from burgers_research.models import build_model, parameter_count
from burgers_research.training import predict

ROOT = Path(__file__).resolve().parents[1]/"artifacts"/"study"

@torch.no_grad()
def main():
    cfg=load_config(ROOT/"execution/source/configs/production.yaml")
    store=Store(cfg,ROOT,"validation")
    norm=store.manifest["normalization"]
    device=torch.device("mps")
    summaries=[]
    try:
        for name in ["unet-s0","fno-s0","physics-spectral-l1-s0","persistence"]:
            model=None
            if name != "persistence":
                path=ROOT/"runs"/name/"best.pt"
                checkpoint=torch.load(path,map_location="cpu",weights_only=False)
                assert checkpoint['dataset_hash']==store.manifest['dataset_hash']
                assert checkpoint['source_hash']==source_hash()
                model=build_model(checkpoint['spec']['model'],cfg).to(device)
                model.load_state_dict(checkpoint['model']); model.eval()
            records=[]
            for start in range(0,len(store.rows),16):
                indices=range(start,min(start+16,len(store.rows)))
                truth=np.stack([store.trajectory(i,256,101) for i in indices]).astype(np.float64)
                tensor=torch.tensor(truth,dtype=torch.float32,device=device)
                current=tensor[:,0:1]
                predictions=[current[:,0].cpu().numpy()]
                teacher_errors=[]
                for t in range(1,101):
                    teacher=predict(model,tensor[:,t-1:t],norm) if model else tensor[:,t-1:t]
                    teacher_errors.append(teacher[:,0].cpu().numpy()-truth[:,t])
                    current=predict(model,current,norm) if model else current
                    predictions.append(current[:,0].cpu().numpy())
                pred=np.stack(predictions,axis=1).astype(np.float64)
                err=pred[:,1:]-truth[:,1:]
                rms0=np.maximum(np.sqrt(np.mean(truth[:,0]**2,axis=-1)),1e-12)
                denominator=np.maximum(np.sqrt(np.mean(truth[:,1:]**2,axis=-1)),np.maximum(rms0[:,None]*1e-8,1e-12))
                relative=np.sqrt(np.mean(err**2,axis=-1))/denominator
                teacher_relative=np.sqrt(np.mean(np.stack(teacher_errors,axis=1)**2,axis=-1))/denominator
                drift=pred[:,1:].mean(-1)-truth[:,0].mean(-1)[:,None]
                for j,i in enumerate(indices):
                    records.append({"id":store.rows[i][2],"one_step_relative_l2":float(teacher_relative[j].mean()),
                        "rollout_relative_l2":float(relative[j].mean()),
                        "error_by_step":{str(t):float(relative[j,t-1]) for t in (1,10,50,100)},
                        "max_mass_drift":float(np.abs(drift[j]).max()),
                        "normalized_max_mass_drift":float(np.abs(drift[j]).max()/rms0[j]),
                        "error_squared_sum":float(np.sum(np.mean(err[j]**2,axis=-1))),
                        "mean_mode_error_squared_sum":float(np.sum(err[j].mean(-1)**2))})
            row={"model":name,"parameters":parameter_count(model) if model else 0,
                 **{k:float(np.mean([r[k] for r in records])) for k in
                    ('one_step_relative_l2','rollout_relative_l2','max_mass_drift','normalized_max_mass_drift')},
                 "mean_mode_fraction_of_squared_error":sum(r['mean_mode_error_squared_sum'] for r in records)/sum(r['error_squared_sum'] for r in records),
                 "error_by_step":{str(t):float(np.mean([r['error_by_step'][str(t)] for r in records])) for t in (1,10,50,100)}}
            if model:
                row['checkpoint_sha256']=file_hash(path)
            summaries.append(row)
            write_json(ROOT/"quick-results"/f"audit-{name}.json",{"summary":row,"records":records})
            print(row,flush=True)
        write_json(ROOT/"quick-results/audit-summary.json",{"split":"validation","trajectories":len(store.rows),"resolution":256,"summaries":summaries})
    finally:
        store.close()

if __name__=="__main__":
    main()
