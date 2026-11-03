"""Write the prespecified matrix; no model loading or outcome-dependent tuning."""
import json
from pathlib import Path

from stegatext_runtime.experiments.config import ROOT, resolve, cases
from stegatext_runtime.experiments.runner import planning

OUT = ROOT/'reports/controlled_comparison_20261019'
TOP_K = (32, 128, 256)
TAUS = (.25, .5, 1., 2.)
CONFIGS = [ROOT/f'configs/experiments/controlled_T1_k{k}.toml' for k in TOP_K]


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    base = (ROOT/'configs/experiments/token_budget_T1.toml').read_text().split('[[variants]]', 1)[0]
    base = base.replace('name = "token_budget_T1_k32"', 'name = "controlled_T1_k32"')
    base = base.replace('seeds = [211, 223]', 'seeds = [211, 223]')
    extra = '''  {id="library_zh", language="zh", text="请用中文写一句自然、完整的话，描述图书馆怎样帮助社区居民学习。"},
  {id="rain_zh", language="zh", text="请用中文写一句自然、完整的句子，描述雨停后公园里的景象。"},
  {id="library_en", language="en", text="Write one natural, complete sentence describing how a library helps people in a community learn."},
  {id="rain_en", language="en", text="Write one natural, complete sentence describing a park just after the rain has stopped."},
'''
    base = base.replace('\n]\n\n[message]', '\n'+extra+']\n\n[message]')
    base += '\n[execution]\norder = "hash_interleaved_v1"\norder_seed = 631\n'
    methods = [('direct','direct',None), ('basic','basic_upstream',None),
               ('mwis','mwis_upstream',None), ('stepwise','stepwise_reviewed',None),
               ('syncpool','syncpool',None), ('fullsync','fullsync',None), ('las','las',None)]
    methods += [(f'adaptive_{int(tau*100):03d}', 'adaptive', tau) for tau in TAUS]
    variants = '\n[[variants]]\nid = "ordinary"\nkind = "ordinary"\n'
    for ident, method, tau in methods:
        for short, coder in [('imec','imec_prefix_exact'), ('discop','discop_copy')]:
            variants += (f'\n[[variants]]\nid = "{ident}_{short}"\nkind = "planner"\n'
                         f'method = "{method}"\ncoder = "{coder}"\n')
            if tau is not None:
                variants += f'tau = {tau}\n'
            if short == 'imec':
                variants += 'coder_parameters = {block_size=6, window_size=3, coupling_strategy="greedy", schedule="max_entropy"}\n'
    plans = []
    for k, path in zip(TOP_K, CONFIGS):
        text = (base+variants).replace('controlled_T1_k32', f'controlled_T1_k{k}').replace('top_k = 32', f'top_k = {k}')
        path.write_text(text)
        config = resolve(path)
        assert len(list(cases(config))) == 368
        plans.append(planning(config))
    assert sum(p['cases'] for p in plans) == 1104
    (OUT/'plans.json').write_text(json.dumps(plans,ensure_ascii=False,indent=2)+'\n')
    return plans


if __name__ == '__main__':
    rows = prepare()
    print(json.dumps(dict(cases=sum(p['cases'] for p in rows), configurations=[str(p) for p in CONFIGS])))
