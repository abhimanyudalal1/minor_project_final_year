import copy
import json

import numpy as np
import pytest
import torch
from torch import nn

from burgers_research.common import load_config, matrix
from burgers_research.data import Store, generate, reuse_dataset, validate
from burgers_research.evaluation import evaluate, evaluate_controls
from burgers_research.models import TimeStepper, build_model, parameter_count
from burgers_research.numerics import verify_solver
from burgers_research.physics import residual, training_loss
from burgers_research.reporting import build_report, summarize
from burgers_research.training import lock_selection, make_scheduler, train


def test_revision_registry_and_capacity():
    cfg = load_config('configs/revision.yaml')
    specs = list(matrix(cfg))
    assert len(specs) == len({s['name'] for s in specs}) == 12
    counts = [parameter_count(build_model(s['model'], cfg, s)) for s in specs[:4]]
    assert counts == [671457, 671460, 671460, 671460]
    assert load_config('configs/shock-evaluation.yaml')['data']['counts'] == {'train': 0, 'validation': 0, 'test': 300}


@pytest.mark.parametrize('n', [256, 512, 1024])
def test_projected_time_step_conserves_nonzero_physical_mean(n):
    torch.manual_seed(17)
    network = nn.Conv1d(1, 1, 1)
    with torch.no_grad():
        network.weight.fill_(-0.5)
        network.bias.fill_(2.0)
    model = TimeStepper(network, 0.01, conserve_mean=True)
    x = torch.randn(2, 1, n) + 0.7
    initial = x.clone()
    for _ in range(200):
        x = model(x)
    # Independent physical-space invariant, including normalization and 200 steps.
    torch.testing.assert_close((x*0.43+0.0075).mean(-1), (initial*0.43+0.0075).mean(-1), atol=2e-6, rtol=0)
    x.square().mean().backward()
    assert network.weight.grad is not None and torch.isfinite(network.weight.grad).all()
    with torch.no_grad():
        network.weight.zero_(); network.bias.zero_()
    torch.testing.assert_close(model(initial), initial, rtol=0, atol=0)


def test_two_step_loss_backpropagates_through_both_steps():
    torch.manual_seed(11)
    model = nn.Conv1d(1, 1, 1).double()
    x, target = torch.randn(2, 1, 32, dtype=torch.float64), torch.randn(2, 1, 32, dtype=torch.float64)
    norm = {'mean': 0.2, 'std': 0.7}
    spec = {'weight': 0.4, 'derivative': 'finite_difference', 'physics_steps': 2}
    actual, data, physics = training_loss(model, x, target, norm, 0.01, 0.001, spec)
    z1 = model((x-0.2)/0.7)
    u1 = z1*0.7+0.2
    u2 = model(z1)*0.7+0.2
    supervised = ((z1-(target-0.2)/0.7)**2).mean()
    penalty = sum(((0.01/0.7)*residual(a, b, 0.01, 0.001, 'finite_difference')).square().mean()
                  for a, b in [(x, u1), (u1, u2)]) / 2
    expected = supervised+0.4*penalty
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(data, supervised)
    torch.testing.assert_close(physics, penalty)
    ga = torch.autograd.grad(actual, tuple(model.parameters()))
    gb = torch.autograd.grad(expected, tuple(model.parameters()))
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b)


def test_cosine_reaches_floor_at_budget():
    p = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([p], lr=0.0003)
    scheduler = make_scheduler(optimizer, {'scheduler': 'cosine', 'epochs': 60, 'eta_min': 0.000003})
    rates = [optimizer.param_groups[0]['lr']]
    for _ in range(60):
        optimizer.step(); scheduler.step()
        rates.append(optimizer.param_groups[0]['lr'])
    assert all(a > b for a, b in zip(rates, rates[1:]))
    assert rates[30] == pytest.approx((0.0003+0.000003)/2)
    assert rates[-1] == pytest.approx(0.000003)


def test_revision_end_to_end_subset_restart_and_shift(tmp_path):
    cfg = load_config('configs/revision-fixture.yaml')
    source = tmp_path/'source'
    verify_solver(cfg, source); generate(cfg, source); validate(cfg, source)
    # New experiment identity reuses immutable numerical data, not checkpoints.
    cfg['dataset_protocol'] = cfg['protocol']
    cfg['protocol'] = 'revision-lowdata-fixture'
    cfg['training'].update(train_trajectories=2, subset_seed=7)
    root = tmp_path/'experiment'
    view = reuse_dataset(cfg, root, source)
    with pytest.raises(ValueError, match='linked dataset'):
        generate(cfg, root)
    store = Store(cfg, root, 'train')
    assert len(store.rows) == 2
    cached_cfg = copy.deepcopy(cfg)
    cached_cfg['runtime']['cache_training'] = True
    cached = Store(cached_cfg, root, 'train')
    for expected, actual in zip(store.pairs(np.array([3,0,2,1])), cached.pairs(np.array([3,0,2,1]))):
        np.testing.assert_array_equal(actual, expected)
    assert cached.normalization == store.normalization
    cached.close()
    raw = np.stack([store.trajectory(i, 32, 3) for i in range(2)]).astype(np.float64)
    assert store.normalization['count'] == raw.size
    np.testing.assert_allclose(store.normalization['mean'], raw.mean(), atol=1e-12)
    np.testing.assert_allclose(store.normalization['std'], raw.std(), atol=1e-12)
    store.close()
    val_store = Store(cfg, root, 'validation')
    assert val_store.normalization == view['normalization']
    assert not set(view['ids']) & {r[2] for r in val_store.rows}
    val_store.close()
    specs = list(matrix(cfg))
    with pytest.raises(ValueError, match='registered training matrix'):
        evaluate(cfg, root, specs[0])
    assert not (root/'evaluation-context.json').exists()
    constrained = specs[3]
    train(cfg, root, constrained, max_updates=1)
    train(cfg, root, constrained)
    other = dict(constrained, name='restart-reference')
    train(cfg, root, other)
    resumed = torch.load(root/'runs'/constrained['name']/'last.pt', weights_only=False)
    continuous = torch.load(root/'runs'/other['name']/'last.pt', weights_only=False)
    for k, value in resumed['model'].items():
        torch.testing.assert_close(value, continuous['model'][k], rtol=0, atol=0)
    assert resumed['scheduler'] == continuous['scheduler']
    assert resumed['training_view_hash'] == view['hash']
    for spec in specs:
        train(cfg, root, spec)
    selected = lock_selection(cfg, root)
    assert len(selected['checkpoints']) == 12
    assert selected['selected'] == {}
    for spec in specs:
        evaluate(cfg, root, spec)
    evaluate_controls(cfg, root)
    summary = summarize(cfg, root)
    assert {r['family'] for r in summary['table']} == {'unet', 'fno', 'physics-soft', 'physics-constrained'}
    assert any(r['metric'] == 'extrapolation_error' for r in summary['table'])
    assert len(summary['comparisons']) == 3*3*(3+4)
    report = build_report(cfg, root)
    assert 'not scientific evidence' in report.read_text()
    assert 'physics-constrained' in report.read_text()
    assert not (report.parent/'lambda.png').exists()

    # Separate low-viscosity dataset: no fitted test normalization or training.
    shock = copy.deepcopy(cfg)
    shock.pop('dataset_protocol'); shock['protocol'] = 'shock-fixture'
    shock['evaluation_only'] = True
    shock['training'].pop('train_trajectories')
    shock['data'].update(counts={'train': 0, 'validation': 0, 'test': 3}, viscosity=0.001, seed=987)
    shock_root = tmp_path/'shock'
    verify_solver(shock, shock_root); manifest = generate(shock, shock_root); validate(shock, shock_root)
    assert manifest['normalization']['mean'] is None
    output = root/'shock-evaluation'
    options = dict(data_cfg=shock, data_root=shock_root, output_root=output)
    for spec in specs:
        evaluate(cfg, root, spec, **options)
    evaluate_controls(cfg, root, **options)
    shifted = summarize(cfg, root, output)
    assert shifted['evaluation_context']['dataset_hash'] != selected['dataset_hash']
    assert shifted['evaluation_context']['shifted']
    assert 'Evaluation viscosity is 0.001' in build_report(cfg, root, output).read_text()
    with pytest.raises(ValueError, match='separate output directory'):
        evaluate(cfg, root, specs[0], data_cfg=shock, data_root=shock_root)
    broken = copy.deepcopy(shock); broken['data']['dt'] *= 2
    with pytest.raises(ValueError, match='timestep'):
        evaluate(cfg, root, specs[0], data_cfg=broken, data_root=shock_root, output_root=output)
    # Dataset link points at the original immutable source; no regeneration occurred.
    assert (root/'data').resolve() == (source/'data').resolve()


def test_rejects_tampered_subset(cfg, tmp_path):
    verify_solver(cfg, tmp_path); generate(cfg, tmp_path); validate(cfg, tmp_path)
    cfg['training'].update(train_trajectories=2, subset_seed=19)
    store = Store(cfg, tmp_path, 'train'); store.close()
    path = tmp_path/'training-view.json'
    saved = json.loads(path.read_text()); saved['normalization']['mean'] += 1
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='Training view changed'):
        Store(cfg, tmp_path, 'validation')
