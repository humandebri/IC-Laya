#!/usr/bin/env python3
"""Compare Laya W8A8 Wasm candidates in one isolated PocketIC instance.

Each variant is installed in its own canister and executes the same synthetic
matrix inputs. Record dynamic IC instructions, checksums, update wall time and
PocketIC server CPU time when the server process can be identified.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import tempfile
import time

from ic import Principal
from ic.candid import Types, decode, encode, labelHash
from pocket_ic import PocketIC

SHAPES = ((28, 3072, 1024), (128, 3072, 1024), (128, 5248, 1024), (128, 1024, 2624))
KEY_OK = '_' + str(labelHash('Ok'))
KEY_INSTRUCTIONS = '_' + str(labelHash('instructions'))
KEY_CHECKSUM = '_' + str(labelHash('checksum'))


def server_cpu():
    """Current CPU seconds of this Python process's PocketIC server."""
    try:
        import psutil
        port_file = f'{tempfile.gettempdir()}/pocket_ic_{os.getpid()}.port'
        configured_server = os.environ.get('POCKET_IC_BIN')
        server_path = os.path.realpath(configured_server) if configured_server else None
        fallback = []
        for proc in psutil.process_iter(['cmdline', 'exe']):
            try:
                cmdline = proc.info['cmdline'] or []
                executable = proc.info['exe'] or ''
                if any(port_file in part for part in cmdline):
                    sample = proc.cpu_times()
                    return sample.user + sample.system
                if server_path and os.path.realpath(executable) == server_path:
                    fallback.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        # Dedicated CI runners have only one PocketIC server. Its command line
        # does not always expose the Python client's port-file path.
        if len(fallback) == 1:
            sample = fallback[0].cpu_times()
            return sample.user + sample.system
    except (ImportError, PermissionError):
        pass
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', action='append', required=True, metavar='NAME=FILE')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--warmups', type=int, default=1)
    a = p.parse_args()
    if a.repeats < 1 or a.warmups < 0:
        p.error('repeats must be positive and warmups nonnegative')
    variants = {}
    for item in a.variant:
        name, sep, filename = item.partition('=')
        if not sep or not name or name in variants:
            p.error(f'invalid or duplicate variant: {item}')
        variants[name] = Path(filename).read_bytes()
    if len(variants) < 2:
        p.error('provide at least two variants')
    owner = Principal.from_hex('0102030405')
    report = {
        'host': platform.platform(), 'machine': platform.machine(),
        'pocket_ic_bin': os.environ.get('POCKET_IC_BIN'),
        'method': 'isolated PocketIC, owner-only benchmark_int8_kernel, deterministic synthetic tensors',
        'variants': {}, 'samples': [], 'summary': [],
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    pic = PocketIC()
    pic.set_sender(owner)
    ids = {}
    for name, wasm in variants.items():
        report['variants'][name] = {'sha256': hashlib.sha256(wasm).hexdigest(), 'bytes': len(wasm)}
        cid = pic.create_canister()
        pic.add_cycles(cid, 100_000_000_000_000)
        pic.install_code(cid, gzip.compress(wasm, compresslevel=9, mtime=0),
                         [{'type': Types.Principal, 'value': owner.bytes}])
        ids[name] = cid

    def call(name, shape, rep):
        request = bytes(encode([{'type': Types.Nat32, 'value': v} for v in shape]))
        before_cpu = server_cpu()
        started = time.perf_counter()
        reply = bytes(pic.update_call(ids[name], 'benchmark_int8_kernel', request))
        elapsed = time.perf_counter() - started
        after_cpu = server_cpu()
        value = decode(reply)[0]['value']
        if KEY_OK not in value:
            raise RuntimeError(f'{name}, {shape}: {value}')
        ok = value[KEY_OK]
        return {'variant': name, 'shape': shape, 'rep': rep,
                'instructions': ok[KEY_INSTRUCTIONS],
                'checksum': bytes(ok[KEY_CHECKSUM]).hex(),
                'update_wall_s': elapsed,
                'server_cpu_s': (after_cpu - before_cpu if after_cpu is not None and before_cpu is not None else None)}

    for shape in SHAPES:
        for rep in range(-a.warmups, a.repeats):
            names = list(variants)
            # Alternate order to reduce time drift favoring one variant.
            if rep % 2:
                names.reverse()
            for name in names:
                row = call(name, shape, rep)
                if rep >= 0:
                    report['samples'].append(row)
                    a.output.write_text(json.dumps(report, indent=2) + '\n')
                    print(shape, name, rep, row['instructions'], round(row['update_wall_s'], 4), flush=True)
        sample = [r for r in report['samples'] if tuple(r['shape']) == shape]
        if len({r['checksum'] for r in sample}) != 1:
            raise RuntimeError(f'checksum mismatch at {shape}')
        baseline = statistics.median(r['instructions'] for r in sample if r['variant'] == list(variants)[0])
        for name in variants:
            part = [r for r in sample if r['variant'] == name]
            report['summary'].append({
                'shape': shape, 'variant': name,
                'median_instructions': int(statistics.median(r['instructions'] for r in part)),
                'instruction_reduction_percent': 100 * (1 - statistics.median(r['instructions'] for r in part) / baseline),
                'median_wall_s': statistics.median(r['update_wall_s'] for r in part),
                'median_server_cpu_s': (statistics.median(r['server_cpu_s'] for r in part)
                                        if all(r['server_cpu_s'] is not None for r in part) else None),
                'total_server_cpu_s': (sum(r['server_cpu_s'] for r in part)
                                       if all(r['server_cpu_s'] is not None for r in part) else None),
            })
        a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
