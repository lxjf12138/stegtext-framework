"""Lossless, bounded archives for the new batch; historical records untouched."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import shutil

from stegatext_runtime.experiments.recording import canonical_digest
from stegatext_runtime.experiments.runner import read, write

MAX_RECORD = 128*2**20


def digest_stream(handle):
    h=hashlib.sha256()
    while block:=handle.read(2**20): h.update(block)
    return h.hexdigest()


def raw_digest(path):
    actual=path if path.exists() else path.with_suffix(path.suffix+'.gz')
    opener=gzip.open if actual.suffix=='.gz' else open
    with opener(actual,'rb') as handle: return digest_stream(handle)


def read_record(path):
    if path.exists():
        assert path.stat().st_size<=MAX_RECORD,path
        return read(path)
    with gzip.open(path.with_suffix(path.suffix+'.gz'),'rb') as handle:
        data=handle.read(MAX_RECORD+1)
    assert len(data)<=MAX_RECORD,path
    result=json.loads(data)
    digest=result.pop('record_sha256')
    assert canonical_digest(result)==digest,path
    return result


def archive(directory):
    ledger_path=directory/'TRACE_ARCHIVE.json'
    ledger=read(ledger_path) if ledger_path.exists() else dict(policy='Lossless gzip; verified raw bytes and original canonical record hashes',files={})
    for stage in ['sender','receiver']:
        for path in sorted((directory/stage).glob('*.json')):
            assert path.stat().st_size<=MAX_RECORD,path
            target=path.with_suffix('.json.gz')
            temporary=target.with_suffix('.gz.tmp')
            original=raw_digest(path)
            if not target.exists():
                with path.open('rb') as source,gzip.open(temporary,'wb',compresslevel=1) as sink:
                    shutil.copyfileobj(source,sink,length=2**20)
                with gzip.open(temporary,'rb') as handle: assert digest_stream(handle)==original
                temporary.replace(target)
            with gzip.open(target,'rb') as handle: assert digest_stream(handle)==original
            name=str(path.relative_to(directory))
            with target.open('rb') as handle: compressed_hash=digest_stream(handle)
            ledger['files'][name]=dict(raw_sha256=original,raw_bytes=path.stat().st_size,
                gzip_sha256=compressed_hash,gzip_bytes=target.stat().st_size)
            write(ledger_path,ledger)
            path.unlink()
    assert all(not (directory/name).exists() and (directory/(name+'.gz')).exists() for name in ledger['files'])
    write(directory/'ARCHIVE_COMPLETE.json',dict(files=len(ledger['files']),
        raw_bytes=sum(r['raw_bytes'] for r in ledger['files'].values()),
        gzip_bytes=sum(r['gzip_bytes'] for r in ledger['files'].values())))
    return ledger
