from copy import deepcopy
import unittest
import base64
import json
from pathlib import Path
import sqlite3
import tempfile

from cloud_to_local_chat import build_rollout, prepare_source, restore_dates, register_project, copy_images, copy_files, existing_project, reset_catalog_sync


def source():
    return {'thread': {'id': 'synthetic-source', 'title': 'Вымышленный чат', 'createdAt': 999.456, 'updatedAt': 1005.789}, 'page': {'hasMore': False}, 'attachments': [], 'turns': [{'id': 'synthetic-turn', 'status': 'completed', 'error': None, 'startedAt': 1000.123, 'completedAt': 1001.987, 'items': [{'type': 'userMessage', 'content': [{'type': 'text', 'text': '  Вопрос\nс новой строкой'}]}, {'type': 'agentMessage', 'text': '**Ответ**\n\nТекст без пересказа.'}]}]}


class ImportTests(unittest.TestCase):
    def test_file_cards_follow_project_files_and_missing_files_stay_missing(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            pdf = root / 'Доверенность.pdf'
            pdf.write_bytes(b'%PDF-1.7\noriginal')
            missing = root / 'absent.pdf.sig'
            data = source()
            question = data['turns'][0]['items'][0]
            question['id'] = 'u'
            header = '# Files mentioned by the user:\n\n## Доверенность.pdf: ' + str(pdf) + '\n\n## Доверенность.pdf.sig: ' + str(missing) + '\n\n## My request:\n'
            question['content'].insert(0, {'type': 'text', 'text': header})
            data['provenance'] = {'fileAttachments': [
                {'messageId': 'u', 'id': 'file-pdf', 'name': 'Доверенность.pdf', 'status': 'available', 'path': str(pdf)},
                {'messageId': 'u', 'id': 'file-sig', 'name': 'Доверенность.pdf.sig', 'status': 'unavailable', 'path': str(missing)}]}
            project = root / 'project'
            normalized = copy_files(data, project)
            files = normalized['provenance']['fileAttachments']
            self.assertEqual(Path(files[0]['path']).read_bytes(), pdf.read_bytes())
            self.assertEqual(files[0]['sha256'], hashlib.sha256(pdf.read_bytes()).hexdigest())
            for file in files:
                self.assertTrue(Path(file['path']).is_relative_to(project))
                self.assertIn(file['path'], normalized['turns'][0]['items'][0]['content'][0]['text'])
            self.assertFalse(Path(files[1]['path']).exists())
            self.assertEqual(normalized['turns'][0]['items'][0]['content'][1:], question['content'][1:])
            self.assertEqual(copy_files(normalized, project), normalized)

    def test_catalog_refresh_preserves_entries_and_other_hosts(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory).resolve()
            (home / 'sqlite').mkdir()
            database = home / 'sqlite/codex-dev.db'
            connection = sqlite3.connect(database)
            connection.executescript('CREATE TABLE local_thread_catalog_sync_state(host_id TEXT PRIMARY KEY,watermark_updated_at REAL,initial_build_complete INTEGER,observation_sequence INTEGER,last_full_reconciled_at INTEGER); CREATE TABLE local_thread_catalog(thread_id TEXT PRIMARY KEY,display_title TEXT);')
            connection.executemany('INSERT INTO local_thread_catalog_sync_state VALUES(?,?,?,?,?)', [('local', 1000, 1, 42, 900), ('remote', 2000, 1, 99, 1900)])
            connection.execute('INSERT INTO local_thread_catalog VALUES(?,?)', ('keep', 'Keep title'))
            connection.commit()
            result = reset_catalog_sync(home / 'backups', home)
            self.assertTrue(Path(result['backup']).exists())
            self.assertEqual(connection.execute("SELECT * FROM local_thread_catalog_sync_state WHERE host_id='local'").fetchone(), ('local', None, 0, 42, None))
            self.assertEqual(connection.execute("SELECT * FROM local_thread_catalog_sync_state WHERE host_id='remote'").fetchone(), ('remote', 2000, 1, 99, 1900))
            self.assertEqual(connection.execute('SELECT * FROM local_thread_catalog').fetchall(), [('keep', 'Keep title')])
            self.assertTrue(reset_catalog_sync(home / 'backups', home)['alreadyRequested'])
            connection.close()

    def test_reuses_project_by_folder_not_name(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            database = root / 'state.sqlite'
            with sqlite3.connect(database) as connection:
                connection.executescript('CREATE TABLE projects(id TEXT,name TEXT,created_at_ms INTEGER,updated_at_ms INTEGER); CREATE TABLE project_roots(project_id TEXT,position INTEGER,path TEXT);')
                connection.execute('INSERT INTO projects VALUES(?,?,?,?)', ('existing', 'Менталка', 100000, 110000))
                connection.execute('INSERT INTO project_roots VALUES(?,?,?)', ('existing', 0, str(root)))
            with patch('cloud_to_local_chat.state_database', return_value=database):
                self.assertEqual(existing_project(root)['id'], 'existing')
                self.assertIsNone(existing_project(root / 'different'))

    def test_rejects_incomplete_sources(self):
        for modify in [lambda s: s['page'].update(hasMore=True), lambda s: s['turns'][0]['items'][1].update(truncated=True), lambda s: s['turns'][0]['items'][0]['content'][0].update(truncated=True), lambda s: s['attachments'].append({'name': 'image.png'}), lambda s: s['turns'].append(deepcopy(s['turns'][0])), lambda s: s['turns'][0].update(status='failed'), lambda s: s['turns'][0].update(completedAt=1)]:
            with self.subTest(modify=modify):
                data = source()
                modify(data)
                with self.assertRaises(ValueError):
                    prepare_source(data)

    def test_rejects_image_placeholder(self):
        data = source()
        data['turns'][0]['items'][0]['content'][0]['text'] = '[User attached 1 image; image contents were not included]'
        with self.assertRaises(ValueError):
            prepare_source(data)

    def test_rejects_unsupported_content_and_roles(self):
        for modify in [lambda s: s['turns'][0]['items'][0]['content'].append({'type': 'image', 'image_url': 'example'}), lambda s: s['turns'][0]['items'].reverse(), lambda s: s['turns'][0]['items'][1].update(type='toolCall')]:
            data = source()
            modify(data)
            with self.assertRaises(ValueError):
                prepare_source(data)

    def test_preserves_text_and_visible_order(self):
        data = source()
        items = prepare_source(data)
        rows = [{'type': 'session_meta', 'payload': {'id': 'new-thread'}, 'timestamp': '2026-01-01T00:00:00Z'}] + [{'type': 'response_item', 'payload': deepcopy(i), 'timestamp': '2026-01-01T00:00:00Z'} for i in items]
        original = deepcopy(rows)
        result = build_rollout(rows, data, 'new-thread', items)
        visible = [r for r in result if r.get('payload', {}).get('type') == 'item_completed']
        self.assertEqual([r['payload']['item']['type'] for r in visible], ['UserMessage', 'AgentMessage'])
        self.assertEqual(visible[0]['payload']['item']['content'][0]['text'], data['turns'][0]['items'][0]['content'][0]['text'])
        self.assertEqual(visible[1]['payload']['item']['content'][0]['text'], data['turns'][0]['items'][1]['text'])
        self.assertEqual(visible[0]['payload']['started_at_ms'], 1000123)
        self.assertEqual(visible[1]['payload']['completed_at_ms'], 1001987)
        response_indexes = [n for n, r in enumerate(result) if r['type'] == 'response_item']
        visible_indexes = [n for n, r in enumerate(result) if r.get('payload', {}).get('type') == 'item_completed']
        self.assertLess(max(response_indexes), min(visible_indexes))
        self.assertEqual([r['ordinal'] for r in result[len(rows):]], list(range(len(rows), len(result))))
        self.assertEqual(rows, original)

    def test_refuses_existing_or_changed_journal(self):
        items = prepare_source(source())
        base = [{'type': 'session_meta', 'payload': {'id': 'new-thread'}, 'timestamp': '2026-01-01T00:00:00Z'}] + [{'type': 'response_item', 'payload': deepcopy(i), 'timestamp': '2026-01-01T00:00:00Z'} for i in items]
        for modify in [lambda r: r[0]['payload'].update(id='another-thread'), lambda r: r[-1]['payload']['content'][0].update(text='Другой текст'), lambda r: r.insert(1, {'type': 'event_msg', 'payload': {'type': 'task_started'}})]:
            rows = deepcopy(base)
            modify(rows)
            with self.assertRaises(ValueError):
                build_rollout(rows, source(), 'new-thread', items)

    def test_accepts_server_added_message_metadata(self):
        items = prepare_source(source())
        rows = [{'type': 'session_meta', 'payload': {'id': 'new-thread'}, 'timestamp': '2026-01-01T00:00:00Z'}]
        for item in items:
            payload = deepcopy(item)
            payload.update(id='server-generated', internal_chat_message_metadata_passthrough={'turn_id': 'auto-compact-0'})
            rows.append({'type': 'response_item', 'payload': payload, 'timestamp': '2026-01-01T00:00:00Z'})
        result = build_rollout(rows, source(), 'new-thread', items)
        self.assertEqual(len([r for r in result if r.get('payload', {}).get('type') == 'item_completed']), 2)

    def test_multiple_turns_keep_chronology(self):
        data = source()
        later = deepcopy(data['turns'][0])
        later.update(id='second', startedAt=1002.0, completedAt=1003.0)
        later['items'][0]['content'][0]['text'] = 'Второй вопрос'
        data['turns'] = [later, data['turns'][0]]
        items = prepare_source(data)
        self.assertEqual(items[0]['content'][0]['text'], source()['turns'][0]['items'][0]['content'][0]['text'])
        self.assertEqual(items[2]['content'][0]['text'], 'Второй вопрос')

    def test_unknown_answer_time_uses_source_turn_for_display(self):
        data = source()
        data['turns'][0]['completedAt'] = None
        data['turns'][0]['items'][1]['createdAt'] = None
        items = prepare_source(data)
        rows = [{'type': 'session_meta', 'payload': {'id': 'new-thread'}, 'timestamp': '2026-01-01T00:00:00Z'}] + [{'type': 'response_item', 'payload': deepcopy(i), 'timestamp': '2026-01-01T00:00:00Z'} for i in items]
        result = build_rollout(rows, data, 'new-thread', items)
        visible = [r for r in result if r.get('payload', {}).get('type') == 'item_completed']
        self.assertEqual(visible[-1]['payload']['started_at_ms'], 1000123)
        self.assertIsNone(result[-1]['payload']['completed_at'])
        self.assertIsNone(data['turns'][0]['items'][1]['createdAt'])

    def test_images_are_copied_without_changing_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = root / 'temporary.png'
            original.write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aZ1UAAAAASUVORK5CYII='))
            data = source()
            data['turns'][0]['items'][0]['content'].append({'type': 'localImage', 'path': str(original)})
            data['turns'][0]['items'][1]['images'] = [str(original)]
            data['attachments'] = [{'name': original.name, 'path': str(original)}]
            project = root / 'project'
            normalized, images = copy_images(data, project)
            self.assertEqual(len(images), 1)
            target = Path(images[0]['path'])
            self.assertEqual(target.read_bytes(), original.read_bytes())
            self.assertTrue(target.is_relative_to(project))
            raw = prepare_source(normalized)
            self.assertEqual(raw[0]['content'][1]['type'], 'input_image')
            rows = [{'type': 'session_meta', 'payload': {'id': 'new-thread'}, 'timestamp': '2026-01-01T00:00:00Z'}] + [{'type': 'response_item', 'payload': deepcopy(i), 'timestamp': '2026-01-01T00:00:00Z'} for i in raw]
            visible = [r['payload']['item'] for r in build_rollout(rows, normalized, 'new-thread', raw) if r.get('payload', {}).get('type') == 'item_completed']
            self.assertEqual(visible[0]['content'][1]['type'], 'local_image')
            self.assertIn(str(target), visible[1]['content'][0]['text'])


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.path = self.home / 'thread.jsonl'
        data = source()
        items = prepare_source(data)
        rows = [{'ordinal': 0, 'timestamp': '2026-01-01T00:00:00.000Z', 'type': 'session_meta', 'payload': {'id': 'new-thread', 'timestamp': '2026-01-01T00:00:00.000Z'}}] + [{'ordinal': n + 1, 'timestamp': '2026-01-01T00:00:00.000Z', 'type': 'response_item', 'payload': deepcopy(i)} for n, i in enumerate(items)]
        rows = build_rollout(rows, data, 'new-thread', items)
        self.path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        connection = sqlite3.connect(self.home / 'state_5.sqlite')
        connection.execute('CREATE TABLE threads(id TEXT PRIMARY KEY, rollout_path TEXT, created_at INTEGER, updated_at INTEGER, created_at_ms INTEGER, updated_at_ms INTEGER, recency_at INTEGER, recency_at_ms INTEGER, project_id TEXT)')
        connection.execute('INSERT INTO threads VALUES(?,?,?,?,?,?,?,?,?)', ('new-thread', str(self.path), 2000, 2001, 2000000, 2001000, 2001, 2001000, None))
        connection.execute('CREATE TABLE projects(id TEXT PRIMARY KEY)')
        connection.execute("INSERT INTO projects VALUES('native-project')")
        connection.commit()
        connection.close()

    def test_dates_preserve_existing_body_offsets(self):
        original = self.path.read_bytes()
        result = restore_dates('new-thread', source(), self.home / 'data', self.home)
        changed = self.path.read_bytes()
        before = original.splitlines(keepends=True)
        after = changed.splitlines(keepends=True)
        self.assertEqual(len(before[0]), len(after[0]))
        self.assertEqual(before[1:], after[1:len(before)])
        self.assertTrue(Path(result['backup'], 'rollout.before.jsonl').exists())
        connection = sqlite3.connect(self.home / 'state_5.sqlite')
        dates = connection.execute("SELECT created_at_ms,updated_at_ms,recency_at,recency_at_ms FROM threads WHERE id='new-thread'").fetchone()
        connection.close()
        self.assertEqual(dates, (999456, 1005789, 2001, 2001000))

    def test_dates_refuse_chat_with_new_messages(self):
        with self.path.open('a') as output:
            output.write(json.dumps({'type': 'event_msg', 'payload': {'type': 'item_completed', 'item': {'type': 'UserMessage', 'id': 'extra', 'content': [{'type': 'text', 'text': 'Новый вопрос'}]}}}) + '\n')
        original = self.path.read_bytes()
        with self.assertRaises(ValueError):
            restore_dates('new-thread', source(), self.home / 'data', self.home)
        self.assertEqual(self.path.read_bytes(), original)

    def test_registration_preserves_unrelated_projects(self):
        state_path = self.home / '.codex-global-state.json'
        state_path.write_text(json.dumps({'unrelated': {'keep': True}, 'local-projects': {'existing': {'name': 'Другой проект', 'rootPaths': ['/other']}}, 'project-order': ['existing']}))
        project_root = self.home / 'project'
        project_root.mkdir()
        result = register_project({'id': 'native-project', 'name': 'Вымышленный проект', 'roots': [{'path': str(project_root)}], 'createdAt': 1000, 'updatedAt': 1001}, 'new-thread', self.home / 'data', self.home)
        state = json.loads(state_path.read_text())
        self.assertEqual(state['unrelated'], {'keep': True})
        self.assertEqual(state['local-projects']['existing']['name'], 'Другой проект')
        self.assertEqual(state['thread-project-assignments']['new-thread']['projectId'], result['projectId'])
        connection = sqlite3.connect(self.home / 'state_5.sqlite')
        self.assertEqual(connection.execute("SELECT project_id FROM threads WHERE id='new-thread'").fetchone()[0], 'native-project')
        connection.close()


if __name__ == '__main__':
    unittest.main()
