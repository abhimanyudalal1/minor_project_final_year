"""Resumable download of the requested official PDEBench Burgers file.

Uses previously fetched DaRUS metadata, verifies published byte count and MD5,
adds SHA256, and records HDF5 schema without training or evaluating a model.
"""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import h5py

ROOT = Path(__file__).resolve().parents[1]/'data/pdebench'
NAME = '1D_Burgers_Sols_Nu0.001.hdf5'


def now():
    return datetime.now(timezone.utc).isoformat()


def save(state):
    state['updated_utc'] = now()
    temporary = ROOT/'download-status.tmp'
    temporary.write_text(json.dumps(state, indent=2)+'\n')
    temporary.replace(ROOT/'download-status.json')


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with open(ROOT/'download.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        version = json.loads((ROOT/'darus-metadata.json').read_text())['data']
        record = next(f for f in version['files'] if f['label'] == NAME)
        file = record['dataFile']
        url = f"https://darus.uni-stuttgart.de/api/access/datafile/{file['id']}"
        target = ROOT/NAME
        partial = ROOT/(NAME+'.partial')
        state = {'status':'starting','pid':os.getpid(),'started_utc':now(),
                 'filename':NAME,'source_url':url,'dataset_doi':'10.18419/darus-2986',
                 'dataset_version':f"{version['versionNumber']}.{version['versionMinorNumber']}",
                 'file_version':record['version'],'expected_bytes':file['filesize'],
                 'expected_md5':file['checksum']['value'],'verified':False}
        child = None

        def stop(signum, frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, stop)
        try:
            save(state)
            if not target.exists():
                for attempt in range(1, 9):
                    start = time.monotonic()
                    initial = partial.stat().st_size if partial.exists() else 0
                    if initial == file['filesize']:
                        break
                    if initial > file['filesize']:
                        raise ValueError('Partial file exceeds official size; retain it for inspection')
                    state.update(status='downloading',attempt=attempt,downloaded_bytes=initial)
                    save(state)
                    command = ['curl','--fail','--location','--continue-at','-',
                               '--connect-timeout','30','--speed-time','90','--speed-limit','1024',
                               '--silent','--show-error','--output',str(partial),url]
                    with open(ROOT/'transfer.log','ab',buffering=0) as log:
                        child = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                        while child.poll() is None:
                            size = partial.stat().st_size if partial.exists() else 0
                            speed = max(0,size-initial)/max(time.monotonic()-start,0.001)
                            state.update(downloaded_bytes=size,percent=100*size/file['filesize'],
                                         bytes_per_second=speed,
                                         estimated_seconds_remaining=(file['filesize']-size)/speed if speed else None)
                            save(state)
                            try:
                                child.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                pass
                    if child.returncode == 0:
                        break
                    state.update(last_curl_exit=child.returncode)
                    save(state)
                    if attempt == 8:
                        raise RuntimeError('Download attempts exhausted; partial data retained for resumption')
                    time.sleep(10)
            path = target if target.exists() else partial
            if not path.exists() or path.stat().st_size != file['filesize']:
                raise ValueError('Downloaded size does not match official metadata')
            state.update(status='verifying',downloaded_bytes=path.stat().st_size,percent=100.0)
            save(state)
            md5, sha = hashlib.md5(), hashlib.sha256()
            with open(path,'rb') as stream:
                for block in iter(lambda:stream.read(8*1024*1024),b''):
                    md5.update(block);sha.update(block)
            if md5.hexdigest() != file['checksum']['value']:
                raise ValueError('Published MD5 mismatch; file retained but NOT marked verified')
            schema = {}
            with h5py.File(path,'r') as f:
                def inspect(name,item):
                    if isinstance(item,h5py.Dataset):
                        entry = {'shape':list(item.shape),'dtype':str(item.dtype)}
                        if item.ndim == 1 and item.size and ('coordinate' in name or name in ('t','x','time')):
                            entry.update(first=float(item[0]),last=float(item[-1]))
                        schema[name] = entry
                f.visititems(inspect)
            if path == partial:
                partial.replace(target)
            state.update(status='complete',verified=True,md5=md5.hexdigest(),sha256=sha.hexdigest(),
                         completed_utc=now(),local_path=str(target),schema=schema,
                         equation_conventions_audited=False)
            (ROOT/'provenance.json').write_text(json.dumps(state,indent=2)+'\n')
        except BaseException as exc:
            if child is not None and child.poll() is None:
                child.terminate();child.wait(timeout=15)
            state.update(status='stopped' if isinstance(exc,KeyboardInterrupt) else 'failed',message=str(exc))
            raise
        finally:
            save(state)


if __name__ == '__main__':
    main()
