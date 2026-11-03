"""The new scorer must preserve native scores and use the complete path law."""
import math

import pytest
import torch

from stegatext_runtime.experiments.native_ppl import score_sequence
from tests.test_native_token_ppl import scoring_fixture
from tools.expanded_native_ppl_20261021 import EntropyReference, TILTS, enriched_score
from tools.expanded_records_20261021 import archive, raw_digest, read_record
from stegatext_runtime.experiments.runner import write


@pytest.mark.parametrize('ids',[[1,3],[0,2,3],[3]])
def test_entropy_observer_preserves_every_existing_native_score(ids):
    source,ref,sequence=scoring_fixture()
    original=score_sequence(source,sequence(ids),reference=ref)
    enhanced=enriched_score(source,sequence(ids),reference=ref)
    for key,value in original.items():
        if key=='per_token':
            for old,new in zip(value,enhanced[key]):
                assert all(new[k]==v for k,v in old.items())
        else: assert enhanced[key]==value
    total=enhanced['entropy_inclusive']
    assert total['normalized_sampler_nll']==pytest.approx(total['conditional_entropy_nats']+total['surprisal_residual_nats'])


def test_surprisal_calibration_factors_have_mean_one_under_normalized_source():
    source,ref,_=scoring_fixture()
    observed=EntropyReference(ref)
    logits=ref.next_logits([4,4])
    q=observed.probabilities([4,4],logits=logits)
    log_mgfs=observed.diagnostics[0]['sampler_surprisal_log_mgfs']
    for theta,mgf in zip(TILTS,log_mgfs):
        assert sum(float(p)*math.exp(theta*(-math.log(float(p)))-mgf) for p in q)==pytest.approx(1.)
    assert observed.diagnostics[0]['sampler_entropy_nats']==pytest.approx(-sum(float(p)*math.log(float(p)) for p in q))


def test_lossless_archive_keeps_exact_bytes_and_validated_record(tmp_path):
    path=tmp_path/'sender/test.json'
    write(path,dict(text='中文',trace=[dict(p=[.1,.9])]*50))
    before=path.read_bytes();digest=raw_digest(path)
    archive(tmp_path)
    assert not path.exists() and raw_digest(path)==digest
    assert read_record(path)['text']=='中文'
    import gzip
    assert gzip.decompress(path.with_suffix('.json.gz').read_bytes())==before
    archive(tmp_path)  # Resuming archive does not rewrite or lose evidence.
