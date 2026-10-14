#!/usr/bin/env python3
"""Импорт переписки ChatGPT с локальными изображениями в историю Codex."""

import argparse
import base64
from datetime import datetime, timezone
import math
import mimetypes
import re
import shutil
import sqlite3
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import tempfile
import time
import uuid


def image_reference(path):
    path = Path(path).resolve(strict=True)
    mime = mimetypes.guess_type(path.name)[0]
    if mime not in ('image/png', 'image/jpeg', 'image/webp', 'image/gif'):
        raise ValueError('Неподдерживаемый формат картинки: ' + str(path))
    return path, mime


def agent_text(item):
    text = item.get('text', '')
    for n, image in enumerate(item.get('images', []), 1):
        path, _ = image_reference(image)
        text += ('\n\n' if text else '') + f'![Сгенерированное изображение {n}](<{path}>)'
    return text


def ordered_turns(source):
    if source.get('page', {}).get('hasMore'):
        raise ValueError('История неполная: загрузите все страницы.')
    if not source.get('thread', {}).get('id') or not source['thread'].get('title'):
        raise ValueError('Отсутствуют идентификатор или название исходного чата.')
    turns = source.get('turns', [])
    if not turns:
        raise ValueError('История пуста.')
    seen = set()
    for turn in turns:
        identifier = turn.get('id')
        if identifier and identifier in seen:
            raise ValueError('Обмен присутствует в источнике несколько раз.')
        if identifier:
            seen.add(identifier)
        if turn.get('status') != 'completed' or turn.get('error'):
            raise ValueError('Обмен не завершён успешно.')
        start = turn.get('startedAt')
        end = turn.get('completedAt')
        if not isinstance(start, (int, float)) or not math.isfinite(start):
            raise ValueError('Отсутствует время начала обмена.')
        if end is not None and (not isinstance(end, (int, float)) or not math.isfinite(end) or end < start):
            raise ValueError('Неверный порядок дат обмена.')
    return sorted(turns, key=lambda t: t['startedAt'])


def turn_items(turn):
    items = []
    roles = []
    for item in turn.get('items', []):
        if item.get('truncated'):
            raise ValueError('Сообщение обрезано.')
        if item['type'] == 'userMessage':
            content = []
            for part in item.get('content', []):
                if part.get('truncated'):
                    raise ValueError('Сообщение обрезано.')
                if part['type'] == 'text':
                    if '[User attached' in part['text']:
                        raise ValueError('Экспорт содержит заглушку вместо изображения.')
                    content.append({'type': 'input_text', 'text': part['text']})
                elif part['type'] in ('localImage', 'local_image'):
                    path, mime = image_reference(part['path'])
                    encoded = base64.b64encode(path.read_bytes()).decode('ascii')
                    content.append({'type': 'input_image', 'image_url': f'data:{mime};base64,{encoded}'})
                else:
                    raise ValueError('Неподдерживаемое содержимое: ' + part['type'])
            if not content:
                raise ValueError('Пустое сообщение пользователя.')
            roles.append('user')
            items.append({'type': 'message', 'role': 'user', 'content': content})
        elif item['type'] == 'agentMessage':
            text = agent_text(item)
            if not text:
                raise ValueError('Пустой ответ ассистента.')
            roles.append('assistant')
            items.append({'type': 'message', 'role': 'assistant', 'phase': 'final_answer', 'content': [{'type': 'output_text', 'text': text}]})
        else:
            raise ValueError('Неподдерживаемое сообщение: ' + item['type'])
    if not roles or roles[0] != 'user' or 'assistant' not in roles or roles.count('user') != 1:
        raise ValueError('В каждом обмене нужны вопрос и ответ в этом порядке.')
    return items


def prepare_source(source):
    turns = ordered_turns(source)
    for field in ('createdAt', 'updatedAt'):
        value = source['thread'].get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Отсутствует исходная дата чата: ' + field)
    if source['thread']['updatedAt'] < source['thread']['createdAt']:
        raise ValueError('Дата обновления раньше создания чата.')
    referenced = {str(image_reference(c['path'])[0]) for t in turns for i in t['items'] for c in i.get('content', []) if c.get('type') in ('localImage', 'local_image')}
    for attachment in source.get('attachments', []):
        if not attachment.get('path'):
            raise ValueError('Для вложения не указан локальный файл.')
        if str(image_reference(attachment['path'])[0]) not in referenced:
            raise ValueError('Вложение не сопоставлено с сообщением.')
    return [item for turn in turns for item in turn_items(turn)]


def visible_items_to_raw(items):
    actual = []
    for item in items:
        if item['type'] in ('UserMessage', 'userMessage'):
            content = []
            for part in item['content']:
                if part['type'] == 'text':
                    content.append({'type': 'input_text', 'text': part['text']})
                elif part['type'] in ('localImage', 'local_image'):
                    file, mime = image_reference(part['path'])
                    content.append({'type': 'input_image', 'image_url': 'data:' + mime + ';base64,' + base64.b64encode(file.read_bytes()).decode('ascii')})
                else:
                    raise ValueError('Неизвестное видимое содержимое: ' + part['type'])
            actual.append({'type': 'message', 'role': 'user', 'content': content})
        else:
            text = item.get('text')
            if text is None:
                text = '\n'.join(c['text'] for c in item['content'] if c['type'] == 'Text')
            actual.append({'type': 'message', 'role': 'assistant', 'phase': 'final_answer', 'content': [{'type': 'output_text', 'text': text}]})
    return actual


def utc_stamp(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')


def build_rollout(rows, source, thread_id, items):
    result = deepcopy(rows)
    if not result or result[0].get('type') != 'session_meta' or result[0]['payload'].get('id') != thread_id:
        raise ValueError('Журнал принадлежит другому чату.')
    if any(r.get('payload', {}).get('type') == 'task_started' for r in result):
        raise ValueError('Журнал уже содержит обмен; правка запрещена.')
    tail = result[-len(items):]
    keys = ('type', 'role', 'content', 'phase')
    if len(tail) != len(items) or any(r.get('type') != 'response_item' or any(r.get('payload', {}).get(k) != i.get(k) for k in keys) for r, i in zip(tail, items)):
        raise ValueError('Неожиданный конец журнала; правка запрещена.')
    events = []
    for turn in ordered_turns(source):
        turn_id = str(uuid.uuid4())
        started = turn['startedAt']
        completed = turn.get('completedAt')
        events.append({'timestamp': utc_stamp(started), 'type': 'event_msg', 'payload': {'type': 'task_started', 'turn_id': turn_id, 'started_at': int(started), 'model_context_window': 475000, 'collaboration_mode_kind': 'default'}})
        visible_events = []
        for item in turn['items']:
            if item['type'] == 'userMessage':
                visible_content = deepcopy(item['content'])
                for part in visible_content:
                    if part['type'] == 'localImage':
                        part['type'] = 'local_image'
                visible = {'type': 'UserMessage', 'id': 'imported-' + uuid.uuid4().hex, 'content': visible_content}
                timestamp = item.get('createdAt', started)
            else:
                visible = {'type': 'AgentMessage', 'id': 'imported-' + uuid.uuid4().hex, 'content': [{'type': 'Text', 'text': agent_text(item)}], 'phase': 'final_answer'}
                timestamp = item.get('createdAt', completed)
            # Протокол требует числовую отметку: при неизвестном времени ответа
            # используем известное начало обмена, а не момент переноса.
            event_time = timestamp if timestamp is not None else started
            visible_events.append({'timestamp': utc_stamp(event_time), 'type': 'event_msg', 'payload': {'type': 'item_completed', 'thread_id': thread_id, 'turn_id': turn_id, 'item': visible, 'started_at_ms': int(event_time * 1000), 'completed_at_ms': int(event_time * 1000)}})
        # Сначала все исходные response_item, затем события отображения.
        events.extend(visible_events)
        final = next((agent_text(i) for i in reversed(turn['items']) if i['type'] == 'agentMessage'), None)
        events.append({'timestamp': utc_stamp(completed if completed is not None else started), 'type': 'event_msg', 'payload': {'type': 'task_complete', 'turn_id': turn_id, 'started_at': int(started), 'completed_at': None if completed is None else int(completed), 'last_agent_message': final}})
    next_ordinal = max((r.get('ordinal', n) for n, r in enumerate(rows)), default=-1) + 1
    for n, event in enumerate(events):
        event['ordinal'] = next_ordinal + n
    result.extend(events)
    return result


def copy_images(source, project_root):
    result = deepcopy(source)
    copied = {}
    def keep(path):
        original, _ = image_reference(path)
        if original not in copied:
            checksum = hashlib.sha256(original.read_bytes()).hexdigest()
            if original.is_relative_to(project_root):
                target = original
            else:
                target = project_root / 'images' / (checksum[:24] + original.suffix.lower())
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != checksum:
                    raise ValueError('Конфликт файла картинки.')
                if not target.exists():
                    shutil.copyfile(original, target)
            copied[original] = {'path': str(target), 'sha256': checksum, 'originalPath': str(original)}
        return copied[original]['path']
    for turn in result['turns']:
        for item in turn['items']:
            for content in item.get('content', []):
                if content.get('type') in ('localImage', 'local_image'):
                    content['path'] = keep(content['path'])
            if item.get('images'):
                item['images'] = [keep(path) for path in item['images']]
    for attachment in result.get('attachments', []):
        attachment['path'] = keep(attachment['path'])
    return result, list(copied.values())


def write_atomic(path, content, expected=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    replacement = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent, prefix=path.name + '.', suffix='.importing', delete=False) as output:
            replacement = Path(output.name)
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        if expected is not None and path.read_bytes() != expected:
            raise RuntimeError('Файл изменился во время операции; запись остановлена: ' + str(path))
        replacement.replace(path)
    finally:
        if replacement and replacement.exists():
            replacement.unlink()


def codex_directory(codex_home=None):
    return Path(codex_home or os.environ.get('CODEX_HOME') or Path.home() / '.codex').resolve()


def state_database(codex_home=None):
    for candidate in sorted(codex_directory(codex_home).glob('state*.sqlite'), reverse=True):
        connection = sqlite3.connect('file:' + str(candidate) + '?mode=ro', uri=True)
        try:
            columns = {r[1] for r in connection.execute('pragma table_info(threads)')}
            if {'id', 'rollout_path', 'created_at', 'updated_at'} <= columns:
                return candidate
        finally:
            connection.close()
    raise RuntimeError('Индекс чатов Codex не найден.')


def restore_dates(thread_id, source, data_dir, codex_home=None, expected_items=None):
    for field in ('createdAt', 'updatedAt'):
        value = source['thread'].get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Отсутствует исходная дата чата: ' + field)
    connection = sqlite3.connect(state_database(codex_home), timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        record = connection.execute('SELECT * FROM threads WHERE id=?', (thread_id,)).fetchone()
        if not record:
            raise ValueError('Чат не найден в индексе.')
        path = Path(record['rollout_path'])
        original = path.read_bytes()
        lines = original.splitlines(keepends=True)
        rows = [json.loads(line) for line in lines]
        if rows[0]['payload'].get('id') != thread_id:
            raise ValueError('Журнал принадлежит другому чату.')
        existing = [r['payload']['item'] for r in rows if r.get('payload', {}).get('type') == 'item_completed' and r['payload']['item']['type'] in ('UserMessage', 'AgentMessage')]
        expected = expected_items or prepare_source(source)
        actual = visible_items_to_raw(existing)
        if actual != expected:
            raise ValueError('В чате появились новые сообщения или содержимое отличается; даты не изменены.')
        backup_dir = Path(data_dir) / ('dates-' + thread_id + '-' + uuid.uuid4().hex[:8])
        backup_dir.mkdir(parents=True)
        (backup_dir / 'rollout.before.jsonl').write_bytes(original)
        (backup_dir / 'thread.before.json').write_text(json.dumps(dict(record), ensure_ascii=False, indent=2))
        date_bytes = utc_stamp(source['thread']['createdAt']).encode()
        def fixed_date(match):
            old = match.group(2)
            if len(old) != len(date_bytes):
                raise ValueError('Нельзя менять длину заголовка: нарушатся смещения индекса.')
            return match.group(1) + date_bytes + match.group(3)
        lines[0] = re.sub(rb'("timestamp"\s*:\s*")([0-9TZ:.-]+)(")', fixed_date, lines[0])
        # Уточняем времена обменов добавочными событиями, не меняя старые смещения.
        imported_turns = ordered_turns(source)
        started_events = [r for r in rows if r.get('payload', {}).get('type') == 'task_started']
        if len(started_events) != len(imported_turns):
            raise ValueError('Число обменов отличается; даты не изменены.')
        next_ordinal = max(r.get('ordinal', n) for n, r in enumerate(rows)) + 1
        additions = []
        for offset, (local, old) in enumerate(zip(started_events, imported_turns)):
            end = old.get('completedAt')
            additions.append({'timestamp': utc_stamp(end if end is not None else old['startedAt']), 'ordinal': next_ordinal + offset, 'type': 'event_msg', 'payload': {'type': 'task_complete', 'turn_id': local['payload']['turn_id'], 'started_at': int(old['startedAt']), 'completed_at': None if end is None else int(end), 'last_agent_message': next((agent_text(i) for i in reversed(old['items']) if i['type'] == 'agentMessage'), None)}})
        revised = b''.join(lines) + b''.join((json.dumps(row, ensure_ascii=False) + '\n').encode() for row in additions)
        created = source['thread']['createdAt']
        updated = source['thread']['updatedAt']
        values = {'created_at': int(created), 'updated_at': int(updated), 'created_at_ms': int(created * 1000), 'updated_at_ms': int(updated * 1000), 'recency_at': int(updated), 'recency_at_ms': int(updated * 1000)}
        values = {k: v for k, v in values.items() if k in record.keys()}
        connection.execute('BEGIN IMMEDIATE')
        current = connection.execute('SELECT updated_at_ms FROM threads WHERE id=?', (thread_id,)).fetchone()
        if current[0] != record['updated_at_ms']:
            raise RuntimeError('Чат изменился во время операции; даты не записаны.')
        write_atomic(path, revised, expected=original)
        try:
            connection.execute('UPDATE threads SET ' + ','.join(k + '=?' for k in values) + ' WHERE id=?', (*values.values(), thread_id))
            connection.commit()
        except BaseException:
            connection.rollback()
            write_atomic(path, original, expected=revised)
            raise
        return {'createdAt': created, 'updatedAt': updated, 'backup': str(backup_dir)}
    finally:
        connection.close()


def desktop_running(directory):
    if directory != codex_directory():
        return False
    processes = subprocess.run(['ps', '-axo', 'comm='], capture_output=True, text=True, check=True).stdout
    return any(re.search(r'/(ChatGPT|Codex)\.app/Contents/MacOS/(ChatGPT|Codex)$', line.strip()) for line in processes.splitlines())


def register_project(project, thread_id, data_dir, codex_home=None):
    directory = codex_directory(codex_home)
    if desktop_running(directory):
        request = Path(data_dir) / ('register-' + thread_id + '.json')
        request.parent.mkdir(parents=True, exist_ok=True)
        request.write_text(json.dumps({'project': project, 'threadId': thread_id, 'codexHome': str(directory)}, ensure_ascii=False, indent=2))
        return {'pending': True, 'request': str(request.resolve()), 'reason': 'Приложение открыто и перезаписывает реестр из памяти. Регистрация выполняется после его закрытия.'}
    connection = sqlite3.connect(state_database(codex_home), timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        if not connection.execute('SELECT id FROM projects WHERE id=?', (project['id'],)).fetchone():
            raise ValueError('Проект отсутствует в локальном API.')
        record = connection.execute('SELECT * FROM threads WHERE id=?', (thread_id,)).fetchone()
        if not record:
            raise ValueError('Чат отсутствует в локальном API.')
        if record['project_id'] not in (None, project['id']):
            raise ValueError('Привязка чата изменилась после подготовки заявки.')
        if 'cwd' in record.keys() and str(Path(record['cwd']).resolve()) != str(Path(project['roots'][0]['path']).resolve()):
            raise ValueError('Рабочая папка чата отличается от заявки.')
    finally:
        connection.close()
    path = directory / '.codex-global-state.json'
    original = path.read_bytes() if path.exists() else b'{}'
    state = json.loads(original)
    root = project['roots'][0]['path']
    projects = state.setdefault('local-projects', {})
    legacy = next((key for key, item in projects.items() if item.get('rootPaths') == [root]), None)
    if legacy is None:
        legacy = 'local-' + hashlib.md5(root.encode()).hexdigest()
        projects[legacy] = {'id': legacy, 'name': project['name'], 'rootPaths': [root], 'createdAt': project['createdAt'] * 1000, 'updatedAt': project['updatedAt'] * 1000}
    order = state.setdefault('project-order', [])
    if legacy not in order:
        order.insert(0, legacy)
    roots = state.setdefault('electron-saved-workspace-roots', [])
    if root not in roots:
        roots.append(root)
    state.setdefault('thread-project-assignments', {})[thread_id] = {'projectKind': 'local', 'projectId': legacy}
    state.setdefault('thread-project-membership-host-ids', {})[thread_id] = 'local'
    state.setdefault('thread-workspace-root-hints', {})[thread_id] = root
    state.setdefault('app-server-project-id-by-legacy-project-id-by-host', {}).setdefault('local:' + str(directory), {})[legacy] = project['id']
    if thread_id in state.get('projectless-thread-ids', []):
        state['projectless-thread-ids'].remove(thread_id)
    state.get('thread-projectless-output-directories', {}).pop(thread_id, None)
    backup = Path(data_dir) / ('projects-' + uuid.uuid4().hex[:8] + '.before.json')
    backup.parent.mkdir(parents=True, exist_ok=True)
    backup.write_bytes(original)
    write_atomic(path, json.dumps(state, ensure_ascii=False, separators=(',', ':')).encode(), expected=original if path.exists() else None)
    connection = sqlite3.connect(state_database(codex_home), timeout=10)
    try:
        if not connection.execute('SELECT id FROM projects WHERE id=?', (project['id'],)).fetchone():
            raise ValueError('Проект отсутствует в локальном API.')
        connection.execute('UPDATE threads SET project_id=? WHERE id=?', (project['id'], thread_id))
        connection.commit()
    except BaseException:
        current = path.read_bytes()
        if json.loads(current) == state:
            write_atomic(path, original, expected=current)
        raise
    finally:
        connection.close()
    return {'projectId': legacy, 'nativeProjectId': project['id'], 'registryBackup': str(backup)}


class Client:
    def __init__(self, executable='codex', codex_home=None):
        env = os.environ.copy()
        if codex_home:
            env['CODEX_HOME'] = str(codex_home)
        self.process = subprocess.Popen([executable, 'app-server', '--stdio'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, bufsize=0)
        self.pending = b''
        self.sequence = 0
        try:
            self.rpc('initialize', {'clientInfo': {'name': 'cloud_to_local_chat', 'version': '0.1'}, 'capabilities': {'experimentalApi': True}})
            self.send({'method': 'initialized', 'params': {}})
        except BaseException:
            self.close()
            raise

    def send(self, message):
        self.process.stdin.write((json.dumps(message) + '\n').encode())

    def rpc(self, method, params):
        self.sequence += 1
        self.send({'id': self.sequence, 'method': method, 'params': params})
        deadline = time.monotonic() + 40
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while time.monotonic() < deadline:
                if b'\n' not in self.pending:
                    if not selector.select(timeout=1):
                        continue
                    chunk = os.read(self.process.stdout.fileno(), 65536)
                    if not chunk:
                        raise RuntimeError('App-server завершился.')
                    self.pending += chunk
                    if b'\n' not in self.pending:
                        continue
                line, self.pending = self.pending.split(b'\n', 1)
                response = json.loads(line)
                if response.get('id') == self.sequence:
                    if 'error' in response:
                        raise RuntimeError(json.dumps(response['error'], ensure_ascii=False))
                    return response['result']
        raise TimeoutError(method)

    def close(self):
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        for stream in (self.process.stdin, self.process.stdout):
            stream.close()


def migrate(source, project_root, project_name, data_dir, executable='codex', codex_home=None):
    prepare_source(source)
    project_root = Path(project_root).resolve()
    data_dir = Path(data_dir).resolve()
    identity = hashlib.sha256((source['thread']['id'] + '\0' + str(project_root) + '\0' + str(codex_home or '')).encode()).hexdigest()[:24]
    receipt_path = data_dir / (identity + '.json')
    if receipt_path.exists():
        raise ValueError('Импорт уже запускался: проверьте квитанцию ' + str(receipt_path))
    project_root.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    if codex_home:
        codex_home = Path(codex_home).resolve()
        codex_home.mkdir(parents=True, exist_ok=True)
    source, images = copy_images(source, project_root)
    items = prepare_source(source)
    client = Client(executable, codex_home)
    receipt = {'sourceId': source['thread']['id'], 'status': 'creating'}
    try:
        project = client.rpc('project/create', {'idempotencyKey': 'cloud-to-local-' + hashlib.sha256(str(project_root).encode()).hexdigest(), 'name': project_name, 'roots': [{'path': str(project_root)}]})['project']
        thread = client.rpc('thread/start', {'cwd': str(project_root), 'projectId': project['id'], 'historyMode': 'paginated', 'ephemeral': False, 'threadSource': 'user'})['thread']
        receipt.update(project=project, threadId=thread['id'], path=thread['path'], images=images, provenance=source.get('provenance', {}))
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
        client.rpc('thread/name/set', {'threadId': thread['id'], 'name': source['thread']['title']})
        client.rpc('thread/inject_items', {'threadId': thread['id'], 'items': items})
    finally:
        client.close()
    path = Path(thread['path'])
    original = path.read_text()
    backup = data_dir / (identity + '.before.jsonl')
    backup.write_text(original)
    rows = [json.loads(line) for line in original.splitlines()]
    revised = build_rollout(rows, source, thread['id'], items)
    replacement = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, prefix=path.name + '.', suffix='.importing', delete=False) as output:
            replacement = Path(output.name)
            output.write(original)
            for row in revised[len(rows):]:
                output.write(json.dumps(row, ensure_ascii=False) + '\n')
            output.flush()
            os.fsync(output.fileno())
        replacement.replace(path)
    finally:
        if replacement and replacement.exists():
            replacement.unlink()
    receipt.update(status='written', backup=str(backup))
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    client = Client(executable, codex_home)
    try:
        hydrated = client.rpc('thread/resume', {'threadId': thread['id']})['thread']['turns']
        actual = visible_items_to_raw([i for turn in hydrated for i in turn['items']])
        expected = items
        if actual != expected:
            raise RuntimeError('Прочитанные сообщения не совпали с оригиналом; сохранена резервная копия.')
        receipt.update(status='messages-verified', messageCount=len(actual), exactTextMatch=True)
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    finally:
        client.close()
    receipt['dates'] = restore_dates(thread['id'], source, data_dir, codex_home, expected_items=items)
    receipt['registration'] = register_project(project, thread['id'], data_dir, codex_home)
    receipt['status'] = 'prepared'
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', nargs='?', type=Path, help='Полный JSON ответа read_thread')
    parser.add_argument('--project-root', type=Path)
    parser.add_argument('--project-name')
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--codex', default='codex')
    parser.add_argument('--codex-home', type=Path, help='Отдельное хранилище для теста')
    parser.add_argument('--apply', action='store_true', help='Создать локальный чат; без флага только проверка источника')
    parser.add_argument('--register-request', type=Path, nargs='+', help='Завершить привязку проектов после закрытия приложения')
    parser.add_argument('--wait-for-app-exit', action='store_true', help='Дождаться закрытия приложения для записи реестра (до 15 минут)')
    args = parser.parse_args()
    if args.register_request:
        if not args.apply:
            parser.error('Для записи реестра добавьте --apply.')
        requests = [json.loads(path.read_text()) for path in args.register_request]
        if args.wait_for_app_exit:
            print('Ожидание закрытия ChatGPT. Реестр не изменён.', flush=True)
            deadline = time.monotonic() + 900
            while any(desktop_running(Path(request['codexHome']).resolve()) for request in requests):
                if time.monotonic() > deadline:
                    raise TimeoutError('Приложение не закрыто за 15 минут; реестр не изменён.')
                time.sleep(0.5)
            time.sleep(2)
        results = []
        for request_path, request in zip(args.register_request, requests):
            if request.get('sourcePath'):
                source = json.loads(Path(request['sourcePath']).read_text())
                request['dates'] = restore_dates(request['threadId'], source, args.data_dir, request['codexHome'])
            result = register_project(request['project'], request['threadId'], args.data_dir, request['codexHome'])
            results.append(result)
            if not result.get('pending'):
                request['registration'] = result
                request['status'] = 'registered'
                request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2))
        print(json.dumps({'registrations': results, 'status': 'pending' if any(result.get('pending') for result in results) else 'registered'}, ensure_ascii=False, indent=2), flush=True)
        return
    if not args.source or not args.project_root or not args.project_name:
        parser.error('Укажите источник, --project-root и --project-name.')
    source = json.loads(args.source.read_text())
    items = prepare_source(source)
    if not args.apply:
        print(json.dumps({'sourceId': source['thread']['id'], 'messages': len(items), 'mode': 'preview'}, ensure_ascii=False))
        return
    print(json.dumps(migrate(source, args.project_root, args.project_name, args.data_dir, args.codex, args.codex_home), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
