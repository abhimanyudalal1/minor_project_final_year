"""Rebuild the pilot review and figures from saved validation artifacts."""
import os
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]/'artifacts/study/eight-hour'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'plot-cache'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'plot-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


def main():
    result=json.loads((ROOT/'results.json').read_text())
    names={'unet':'U-Net','fno':'Vanilla FNO','physics-fno':'Physics FNO'}
    history={family:{e:[] for e in (10,20,26)} for family in names}
    ids=None
    for run in result['runs']:
        c=torch.load(ROOT/'milestones'/run['name']/'epoch-26.pt',map_location='cpu',weights_only=False)
        assert c['state']['epoch']==26 and c['state']['cursor']==0
        assert c['dataset_hash']==result['dataset_hash']
        data=json.loads((ROOT/f"validation-{run['name']}.json").read_text())
        current_ids=[r['id'] for r in data['records']]
        assert len(current_ids)==250 and all(i.startswith('validation-') for i in current_ids)
        assert ids is None or ids==current_ids
        ids=current_ids
        assert np.isfinite([r['rollout_error'] for r in data['records']]).all()
        for row in c['state']['history']:
            if row['epoch'] in (10,20):
                history[run['family']][row['epoch']].append(row['validation_rollout_error'])
        history[run['family']][26].append(run['rollout_error'])
    means={family:{e:float(np.mean(v)) for e,v in points.items()} for family,points in history.items()}
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for i,(family,label) in enumerate(names.items()):
        vals=[r['rollout_error'] for r in result['runs'] if r['family']==family]
        axes[0].scatter(i+np.linspace(-0.13,0.13,3),vals,label=label,s=55)
        axes[0].plot([i-.2,i+.2],[np.mean(vals)]*2,color=f'C{i}',linewidth=2)
        axes[1].plot([10,20,26],[means[family][e] for e in (10,20,26)],'o-',label=label)
    axes[0].set_xticks(range(3),list(names.values()))
    axes[0].set(title='Epoch 26: dots = seeds, bars = means',ylabel='Mean rollout relative L2 (log scale)',yscale='log')
    axes[1].set(title='Architecture ranking changes across checkpoints',xlabel='Epoch',ylabel='Mean over three seeds',yscale='log')
    axes[1].set_xticks([10,20,26]);axes[1].legend()
    fig.suptitle('Validation only · N=256 · 100-step rollouts · full dataset')
    fig.tight_layout();fig.savefig(ROOT/'validation-review.png',dpi=180);plt.close(fig)
    lines=['# Reviewed eight-hour pilot','',
           'Training ran for 7 hours 45 minutes; final validation/reporting finished at 06:33 AM IST on 15 November 2026. '
           'All nine runs reached epoch 26. Four runs also completed epoch 27, but those extra epochs are excluded from the matched comparison.', '',
           'All 250 validation trajectories were used, at N=256 and a 100-step horizon. '
           'Checkpoint epoch/cursor, dataset hashes, matching trajectory IDs, and finite rollout errors were verified. '
           'The test set remains untouched. These are endpoint checkpoints, not each run’s best validation checkpoint.', '',
           '| Architecture | Epoch-26 mean | Seed SD | Mean normalized maximum mass drift |',
           '|---|---:|---:|---:|']
    for group in result['groups']:
        mass=np.mean([r['normalized_max_mass_drift'] for r in result['runs'] if r['family']==group['family']])
        lines.append(f"| {names[group['family']]} | {group['mean']:.5f} | {group['seed_sd']:.5f} | {mass:.5f} |")
    lines += ['', '## Checkpoint sensitivity', '',
              '| Architecture | Epoch 10 | Epoch 20 | Epoch 26 |','|---|---:|---:|---:|']
    for family,label in names.items():
        lines.append(f"| {label} | {means[family][10]:.5f} | {means[family][20]:.5f} | {means[family][26]:.5f} |")
    lines += ['',
        'At epoch 26 the physics FNO has the lowest mean error, and beats vanilla FNO in all three paired seeds. '
        'However, vanilla FNO has the lowest mean at epoch 20. The ranking is checkpoint-sensitive, so this pilot '
        'does not establish a stable architectural winner. The optimizer learning rate remained 0.001; its first '
        'scheduled reduction is at epoch 100. Oscillating scores motivate an optimization study, not selective '
        'architecture changes or claims of convergence. Lower mean mass drift also does not imply exact conservation.', '',
        '![Seed spread and checkpoint sensitivity](validation-review.png)', '',
        '## Sensible next experiment', '',
        'Predeclare an equally budgeted learning-rate comparison for all three models, using paired seeds and the full dataset. '
        'Include the current learning rate as the control; assess smaller rates using validation only. '
        'Track rollout validation more frequently under the same rule for every method. Preserve this pilot and all unfavorable results. '
        'After selecting and freezing the training protocol, perform resolution-transfer, energy, and residual diagnostics, '
        'then the locked test evaluation. No new training was launched as part of this review.', '',
        'Raw per-seed results: [results.md](results.md). Machine-readable aggregate: [results.json](results.json).']
    (ROOT/'review.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(means,indent=2));print(ROOT/'review.md')

if __name__=='__main__':
    main()
