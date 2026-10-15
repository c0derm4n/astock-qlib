"""正式策略 API 的一致只读快照与独立刷新任务。"""
from __future__ import annotations
import copy
import json
import os
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path
from src.formal_service import ROOT, OUT, runtime_status, now, preview
from src.formal_rules import registry
from src.runs import sha256
from backend.data_service import ArtifactUnavailableError

_lock = threading.Lock()
_cache = {}
_job = {'running': False, 'phase': 'idle', 'started_at': None, 'finished_at': None, 'error': None, 'log_tail': []}


def snapshot():
    try:
        pointer = json.loads((OUT / 'current.json').read_text(encoding='utf-8'))
        directory = (ROOT / pointer['directory']).resolve()
        if directory.parent != (OUT / 'runs').resolve() or directory.name != pointer['run_id']:
            raise ValueError('正式产物指针不合法')
        with _lock:
            if _cache.get('run_id') != pointer['run_id']:
                completed = json.loads((directory / 'completion.json').read_text(encoding='utf-8'))
                if sha256(directory / 'registration.json') != completed['artifact_hashes']['registration.json']:
                    raise ValueError('正式登记文件校验失败')
                definitions = json.loads((directory / 'registration.json').read_text(encoding='utf-8'))['registry']['strategies']
                payloads, contexts = [], {}
                for definition in definitions:
                    id_ = definition['id']
                    for name in [id_ + '.json', id_ + '_context.json']:
                        if sha256(directory / name) != completed['artifact_hashes'][name]:
                            raise ValueError('正式产物校验失败：' + name)
                    payloads.append(json.loads((directory / (id_ + '.json')).read_text(encoding='utf-8')))
                    contexts[id_] = json.loads((directory / (id_ + '_context.json')).read_text(encoding='utf-8'))
                _cache.clear()
                _cache.update(run_id=pointer['run_id'], payloads=payloads, contexts=contexts, definitions=definitions)
            result = {'run_id': pointer['run_id'], 'completed_at': pointer['completed_at'],
                      'data_end': pointer['data_end'], 'refresh_error': pointer.get('refresh_error'),
                      'strategies': copy.deepcopy(_cache['payloads'])}
        for item in result['strategies']:
            item['runtime'] = runtime_status(item)
        return result
    except (OSError, KeyError, ValueError) as exc:
        raise ArtifactUnavailableError('正式双策略快照尚不可用，请运行刷新：' + str(exc)) from exc


def account_preview(id_, run_id, cash, holdings):
    result = snapshot()
    if result['run_id'] != run_id:
        raise ValueError('策略快照已更新，请刷新页面后重新试算')
    payload = next((s for s in result['strategies'] if s['id'] == id_), None)
    if payload is None:
        raise KeyError(id_)
    # 策略参数取所见运行的登记文件，不受尚未刷新配置的影响。
    registration = json.loads((OUT / 'runs' / run_id / 'registration.json').read_text(encoding='utf-8'))
    definition = next(d for d in registration['registry']['strategies'] if d['id'] == id_)
    # 与所见快照绑定。并发刷新后缓存可能换版，重新按已验证目录读上下文。
    context = json.loads((OUT / 'runs' / run_id / (id_ + '_context.json')).read_text(encoding='utf-8'))
    answer = preview(context, definition, cash, holdings)
    answer['runtime'] = payload['runtime']
    answer['run_id'] = run_id
    return answer


def status():
    with _lock:
        return copy.deepcopy(_job)


def _worker(fetch):
    logs = deque(maxlen=120)
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, '-m', 'src.formal_service'] + (['--fetch'] if fetch else [])
        with (OUT / 'refresh.log').open('w', encoding='utf-8') as file:
            with subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                encoding='utf-8', errors='replace', env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1'),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0) as proc:
                for line in proc.stdout:
                    file.write(line)
                    file.flush()
                    logs.append(line.rstrip())
                    with _lock:
                        _job['log_tail'] = list(logs)
                        if line.startswith('免费源') or line.startswith('免费增量'):
                            _job['phase'] = 'fetch'
                        elif line.startswith('正式 LightGBM 年度模型'):
                            _job['phase'] = 'model'
                        elif '完成：资金' in line:
                            _job['phase'] = 'backtest'
                        elif line.startswith('正式双策略快照已发布'):
                            _job['phase'] = 'publish'
                if proc.wait() != 0:
                    raise RuntimeError('正式策略刷新失败，原有效快照保留；详见刷新日志')
        pointer = json.loads((OUT / 'current.json').read_text(encoding='utf-8'))
        with _lock:
            _job['error'] = pointer.get('refresh_error')
            _job['phase'] = 'done'
    except Exception as exc:
        with _lock:
            _job['error'] = str(exc)
            _job['phase'] = 'failed'
    finally:
        with _lock:
            _job.update(running=False, finished_at=now().isoformat())


def start(fetch=True):
    with _lock:
        if _job['running']:
            raise RuntimeError('已有正式策略刷新任务在运行')
        _job.update(running=True, phase='fetch' if fetch else 'model', started_at=now().isoformat(), finished_at=None, error=None, log_tail=[])
    threading.Thread(target=_worker, args=(fetch,), daemon=True).start()
    return status()
