"""External resource guard for the serial, resumable review parameter scan."""
from __future__ import annotations

import fcntl
import atexit
import json
import os
import signal
import subprocess
import sys
import time

import psutil

from stegatext_runtime.experiments.runner import ROOT, read, write
from tools.review_long_sweep_20261022 import OUT


def main():
    limits=read(OUT/'DESIGN.json')['resource_limits']
    lock=(OUT/'guard.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic();initial_swap=psutil.swap_memory().used
    state=dict(status='running',guard_pid=os.getpid(),limits=limits,peak_process_tree_rss_gib=0,
        minimum_observed_available_gib=1000,maximum_artifact_mib=0,maximum_swap_growth_gib=0,
        started_unix=time.time())
    with (OUT/'run.log').open('a') as log:
        proc=subprocess.Popen([sys.executable,'-B',str(ROOT/'tools/review_long_sweep_20261022.py'),'execute'],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        def cleanup():
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid,signal.SIGTERM)
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:
                    pass
        atexit.register(cleanup)
        state['driver_pid']=proc.pid
        write(OUT/'LAUNCH.json',dict(guard_pid=os.getpid(),driver_pid=proc.pid,started_unix=time.time()))
        with (OUT/'resources.jsonl').open('a') as history:
            while proc.poll() is None:
                elapsed=time.monotonic()-started
                try:
                    family=[psutil.Process(os.getpid())]+psutil.Process(os.getpid()).children(recursive=True)
                except psutil.Error:
                    family=[]
                rss=0
                for p in family:
                    try: rss+=p.memory_info().rss
                    except psutil.Error: pass
                files=list(OUT.rglob('*'))
                sizes=[];large=0;sender=receiver=0
                for p in files:
                    if not p.is_file(): continue
                    try: size=p.stat().st_size
                    except FileNotFoundError: continue
                    sizes.append(size)
                    if p.parent.name in ('sender','receiver'):
                        if p.suffix=='.json': large=max(large,size)
                        if p.name.endswith(('.json','.json.gz')):
                            sender+=p.parent.name=='sender';receiver+=p.parent.name=='receiver'
                available=psutil.virtual_memory().available
                swap=max(0,psutil.swap_memory().used-initial_swap)
                disk=__import__('shutil').disk_usage(OUT).free
                active=read(OUT/'ACTIVE.json') if (OUT/'ACTIVE.json').exists() else {}
                live=dict(unix=time.time(),elapsed_seconds=elapsed,rss_gib=rss/2**30,
                    available_gib=available/2**30,swap_growth_gib=swap/2**30,
                    artifact_mib=sum(sizes)/2**20,largest_record_mib=large/2**20,
                    sender_records=sender,receiver_records=receiver,active=active)
                state.update(elapsed_seconds=elapsed,
                    peak_process_tree_rss_gib=max(state['peak_process_tree_rss_gib'],live['rss_gib']),
                    minimum_observed_available_gib=min(state['minimum_observed_available_gib'],live['available_gib']),
                    maximum_artifact_mib=max(state['maximum_artifact_mib'],live['artifact_mib']),
                    maximum_swap_growth_gib=max(state['maximum_swap_growth_gib'],live['swap_growth_gib']))
                history.write(json.dumps(live)+'\n');history.flush()
                write(OUT/'LIVE.json',live);write(OUT/'RESOURCES.json',state)
                reasons=[]
                if rss/2**30>limits['process_tree_rss_gib']: reasons.append('process_tree_rss')
                if available/2**30<limits['minimum_system_available_gib']: reasons.append('system_available_memory')
                if swap/2**30>limits['maximum_swap_growth_gib']: reasons.append('swap_growth')
                if sum(sizes)/2**30>limits['artifact_gib']: reasons.append('artifact_size')
                if large/2**20>limits['maximum_record_mib']: reasons.append('single_record_size')
                if disk/2**30<limits['minimum_disk_free_gib']: reasons.append('disk_free')
                if elapsed>limits['total_wall_seconds']: reasons.append('total_wall_time')
                if active and time.time()-active['started_unix']>limits['shard_wall_seconds']:
                    reasons.append('shard_phase_wall_time')
                if (OUT/'STOP_REQUESTED.json').exists(): reasons.append('explicit_stop_request')
                if reasons:
                    state.update(status='resource_stop' if reasons!=['explicit_stop_request'] else 'stopped',reasons=reasons)
                    write(OUT/'STOP_STATE.json',dict(reasons=reasons,live=live))
                    os.killpg(proc.pid,signal.SIGTERM)
                    try: proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    break
                time.sleep(5)
        code=proc.wait()
    state.update(returncode=code,elapsed_seconds=time.monotonic()-started,finished_unix=time.time())
    if state['status']=='running': state['status']='finished' if code==0 else 'failed'
    write(OUT/'RESOURCES.json',state)
    print(json.dumps(state),flush=True)
    raise SystemExit(code if code>=0 else 1)


if __name__=='__main__':
    main()
