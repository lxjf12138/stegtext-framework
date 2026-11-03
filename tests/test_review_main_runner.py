"""Scientific run accounting: matched inputs, no retries, explicit stops only."""
from copy import deepcopy
from collections import Counter, defaultdict
import math

import pytest

from stegatext_runtime.experiments import runner
from stegatext_runtime.experiments.config import cases, fingerprint, resolve
from tools.review_main_report_20261023 import aggregate, cell_metrics
from tools.review_main_run_20261023 import PREP, checkpoint_batches, missing_cases, requested_stop_reasons, shard_slices, verify_matched_inputs, verify_slice


def test_ten_document_batches_cover_all_conditions_without_crossing_boundaries():
    counts=Counter()
    prompts=defaultdict(set)
    identities=set()
    for path in sorted((PREP/"configs").glob("*.toml")):
        parent=resolve(path)
        k=parent["source"]["sampling"]["top_k"]
        shard_names=set()
        for placement in shard_slices(32,k):
            start,stop=placement["prompt_start"],placement["prompt_stop"]
            batch=placement["batch_index"]
            assert start//10==batch==(stop-1)//10
            shard=deepcopy(parent)
            shard["name"]=f"batch_{batch}_start_{start}"
            assert shard["name"] not in shard_names
            shard_names.add(shard["name"])
            shard["inputs"]["prompts"]=parent["inputs"]["prompts"][start:stop]
            verify_slice(parent,shard)
            for case in cases(shard):
                identity=(k,case["id"])
                assert identity not in identities
                identities.add(identity)
                key=(batch,case["model"],k,case["variant"]["id"])
                counts[key]+=1
                prompts[key].add(case["prompt"]["id"])
    assert len(identities)==3776
    for batch,size in enumerate((10,10,10,2)):
        cells={key:n for key,n in counts.items() if key[0]==batch}
        assert len(cells)==118 and set(cells.values())=={size}
        for model in ("llama_instruct","qwen_instruct"):
            sets=[prompts[key] for key in cells if key[1]==model]
            assert all(s==sets[0] for s in sets)
    assert [sum(n for key,n in counts.items() if key[0]==b) for b in range(4)]==[1180,1180,1180,236]


def test_batch_checkpoint_waits_for_every_shard_and_preserves_completion(tmp_path):
    design=dict(batches=[dict(batch_index=0,cases=20,stego_cases=10,shard_names=["a","b"])])
    runner.write(tmp_path/"runs/a/COMPLETE.json",dict(processed=10,failed=1))
    checkpoint_batches(tmp_path,design)
    marker=tmp_path/"batches/batch_01/COMPLETE.json"
    assert not marker.exists()
    runner.write(tmp_path/"runs/b/COMPLETE.json",dict(processed=10,failed=0))
    checkpoint_batches(tmp_path,design)
    first=runner.read(marker)
    checkpoint_batches(tmp_path,design)
    assert runner.read(marker)==first
    assert first["cases"]==20 and first["stego_cases"]==10


def test_completed_first_batch_has_comparisons_while_full_plan_remains_partial():
    rows=[]
    for index in range(32):
        for variant,method,nll in (("ordinary","ordinary",10.),("las_imec","las",11.)):
            done=index<10
            rows.append(dict(model="qwen_instruct",top_k=128,variant=variant,method=method,
                batch_index=index//10,prompt_id=f"doc{index}",
                sender_status="ok" if done else "unattempted",
                receiver_status=None if method=="ordinary" else "ok" if done else "unattempted",
                nll_total=nll if done else None,evaluator_tokens=10 if done else 0,
                exact=done and method!="ordinary",bytes=20 if done else 0,
                strict_verified_bits=10 if done and method!="ordinary" else None,
                body_tokens=10 if done else None))
    all_cells=cell_metrics(rows)
    first_cells=cell_metrics([r for r in rows if r["batch_index"]==0])
    assert all(c["planned"]==32 and c["ppl_ratio_to_ordinary"] is None for c in all_cells)
    assert all(c["planned"]==10 and c["scored"]==10 for c in first_cells)
    las=next(c for c in first_cells if c["variant"]=="las_imec")
    assert las["ppl_ratio_to_ordinary"]==pytest.approx(math.exp(.1))
    assert las["verified_bpb"]==pytest.approx(.5)


def test_all_k_use_identical_inputs_and_reject_a_smaller_full_subset():
    configs=[resolve(p) for p in sorted((PREP/"configs").glob("*.toml"))]
    assert len(configs)==10
    verify_matched_inputs(configs)
    for config in configs:
        assert len(config["inputs"]["prompts"])==32
    changed=deepcopy(configs)
    full=next(c for c in changed if c["source"]["sampling"]["top_k"]==0)
    full["inputs"]["prompts"]=full["inputs"]["prompts"][:4]
    with pytest.raises(AssertionError,match="identical prompts"):
        verify_matched_inputs(changed)


def test_slicing_real_main_configuration_preserves_cases_and_random_streams():
    parent=resolve(PREP/"configs/review_main_qwen_instruct_k128_s470001.toml")
    shard=deepcopy(parent)
    shard["name"]="same_experiment_smaller_batch"
    shard["inputs"]["prompts"]=shard["inputs"]["prompts"][4:8]
    verify_slice(parent,shard)
    changed=deepcopy(shard)
    changed["inputs"]["prompts"][0]["assistant_prefix"]+="changed"
    with pytest.raises(AssertionError,match="Sliced case"):
        verify_slice(parent,changed)


def test_resume_keeps_completed_failure_and_rejects_changed_execution(tmp_path, monkeypatch):
    execution={"locked":"environment"}
    cases=[dict(id="failed",variant={"kind":"planner"}),dict(id="missing",variant={"kind":"planner"})]
    monkeypatch.setattr(runner,"ordered_cases",lambda _:cases)
    runner.write(tmp_path/"sender/failed.json",dict(case_id="failed",stage="sender",status="error",
        execution_sha256=fingerprint(execution),error="TimeoutError: retained observation"))
    assert [c["id"] for c in missing_cases({},tmp_path,"sender",execution)]==["missing"]
    assert runner.read(tmp_path/"sender/failed.json")["status"]=="error"
    with pytest.raises(AssertionError):
        missing_cases({},tmp_path,"sender",{"locked":"changed"})


def test_resource_observation_is_not_an_automatic_stop_request(tmp_path):
    runner.write(tmp_path/"RESOURCES.json",dict(elapsed_seconds=100*3600,peak_rss_gib=13))
    runner.write(tmp_path/"LIVE.json",dict(disk_free_gib=19,swap_growth_gib=1))
    assert requested_stop_reasons(tmp_path)==[]
    runner.write(tmp_path/"STOP_REQUESTED.json",dict(reason="User requested a stop"))
    assert requested_stop_reasons(tmp_path)==["explicit_stop"]


def test_ppl_is_token_weighted_and_failed_receiver_stays_in_rate_denominator():
    a=dict(method="las",sender_status="ok",receiver_status="ok",exact=True,
        nll_total=1.,evaluator_tokens=1,bytes=10,strict_verified_bits=20,body_tokens=1)
    b=dict(a,nll_total=27.,evaluator_tokens=9,bytes=90,body_tokens=9,
        receiver_status="error",exact=False,strict_verified_bits=0)
    result=aggregate([a,b])
    assert result["nll_total"]/result["evaluator_tokens"]==pytest.approx(2.8)
    assert result["sentence_mean_log_ppl"]==pytest.approx(2.)
    assert result["verified_bpb"]==pytest.approx(.2)
    assert result["exact_receivers"]==1
    assert result["receiver_status"]=={"ok":1,"error":1}
    c=dict(a,sender_status="unattempted",receiver_status="unattempted",exact=None,
        nll_total=None,evaluator_tokens=0,bytes=0,body_tokens=None,strict_verified_bits=0)
    assert aggregate([a,b,c])["planned"]==3
