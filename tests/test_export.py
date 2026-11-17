import unittest
from contextlib import closing
import tempfile
from pathlib import Path
from cloud_export import active_messages, chat_id, convert, image_suffix, allowed_asset_url, manual_login, has_login, snapshot_profile, http_get, list_project_chats, document_suffix, export_payload, fetch_asset, shared_payload


def payload():
    return {'conversation_id': 'test', 'title': 'Test', 'create_time': 100, 'update_time': 110,
            'current_node': 'a', 'mapping': {
                'u': {'parent': None, 'message': {'id': 'u', 'author': {'role': 'user'}, 'create_time': 101,
                     'content': {'content_type': 'text', 'parts': ['Question']}}},
                'a': {'parent': 'u', 'message': {'id': 'a', 'author': {'role': 'assistant'}, 'status': 'finished_successfully',
                     'content': {'content_type': 'text', 'parts': ['Answer']}}},
                'old': {'parent': 'u', 'message': {'id': 'old', 'author': {'role': 'assistant'},
                       'content': {'content_type': 'text', 'parts': ['Old answer']}}}}}


class ExportTests(unittest.TestCase):
    def test_public_text_export_uses_request_context_without_browser_login(self):
        from unittest.mock import Mock, patch
        from cloud_export import export_chat
        context = Mock()
        context.get.return_value = Mock(status=200, json=Mock(return_value=payload()))
        runtime = Mock()
        runtime.request.new_context.return_value = context
        with tempfile.TemporaryDirectory() as directory:
            with patch('playwright.sync_api.sync_playwright') as factory, patch('cloud_export.manual_login') as login:
                factory.return_value.__enter__.return_value = runtime
                result = export_chat('https://chatgpt.com/share/test', directory)
                self.assertEqual(result['turns'][0]['items'][1]['text'], 'Answer')
                self.assertTrue((Path(directory) / 'cloud-test' / 'raw.json').is_file())
                login.assert_not_called()
                context.dispose.assert_called_once()
                runtime.chromium.launch_persistent_context.assert_not_called()

    def test_shared_history_requires_matching_id_and_valid_json(self):
        from unittest.mock import Mock
        context = Mock()
        context.request.get.return_value = Mock(status=200, json=Mock(return_value=payload()))
        self.assertEqual(shared_payload(context, 'test')['title'], 'Test')
        self.assertNotIn('headers', context.request.get.call_args.kwargs)
        with self.assertRaises(ValueError):
            shared_payload(context, 'different')
        context.request.get.return_value = Mock(status=403)
        with self.assertRaises(RuntimeError):
            shared_payload(context, 'test')
        context.request.get.return_value = Mock(status=200, json=Mock(side_effect=ValueError('HTML')))
        with self.assertRaisesRegex(ValueError, 'JSON'):
            shared_payload(context, 'test')

    def test_shared_image_preserves_original_and_scopes_resolver(self):
        from unittest.mock import Mock
        data = payload()
        data['mapping']['u']['message']['content'] = {'content_type': 'multimodal_text', 'parts': [
            'Question', {'content_type': 'image_asset_pointer',
                         'asset_pointer': 'sediment://file-image?shared_conversation_id=test'}]}
        context = Mock()
        image = b'\x89PNG\r\n\x1a\n'
        context.request.get.side_effect = [
            Mock(status=200, json=Mock(return_value={'download_url': 'https://files.oaiusercontent.com/image'})),
            Mock(status=200, body=Mock(return_value=image))]
        with tempfile.TemporaryDirectory() as directory:
            result = export_payload(context, 'secret', data, 'test', directory, lambda *a, **kw: None, shared=True)
            path = result['turns'][0]['items'][0]['content'][1]['path']
            self.assertEqual(Path(path).read_bytes(), image)
            self.assertEqual(result['provenance']['method'], 'public-share-response')
            self.assertEqual(context.request.get.call_args_list[0].kwargs['params'], {'shared_conversation_id': 'test'})
            self.assertNotIn('headers', context.request.get.call_args_list[1].kwargs)
        data['mapping']['u']['message']['content']['parts'][1]['asset_pointer'] += '-other'
        context.reset_mock()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                export_payload(context, 'secret', data, 'test', directory, lambda *a, **kw: None, shared=True)
        context.request.get.assert_not_called()

    def test_library_download_keeps_auth_on_chatgpt_only(self):
        from unittest.mock import Mock
        context = Mock()
        context.request.get.side_effect = [Mock(status=302, headers={'location': 'https://files.oaiusercontent.com/file'}), Mock(status=200)]
        response = fetch_asset(context, 'secret', '/api/library/files/libfile_one/project-content')
        self.assertEqual(response.status, 200)
        self.assertEqual(context.request.get.call_args_list[0].kwargs['headers'], {'Authorization': 'Bearer secret'})
        self.assertNotIn('headers', context.request.get.call_args_list[1].kwargs)
        context.request.get.side_effect = [Mock(status=302, headers={'location': 'https://evil.test/file'})]
        with self.assertRaises(ValueError):
            fetch_asset(context, 'secret', '/api/library/files/libfile_one/project-content')
        with self.assertRaises(ValueError):
            fetch_asset(context, 'secret', '//evil.test/file')

    def test_project_image_uses_context_resolver_when_direct_link_is_denied(self):
        from unittest.mock import Mock
        data = payload()
        data['gizmo_id'] = 'g-p-test'
        data['mapping']['u']['message']['content'] = {'content_type': 'multimodal_text', 'parts': [
            'Question', {'content_type': 'image_asset_pointer', 'asset_pointer': 'sediment://file-image'}]}
        context = Mock()
        context.request.get.side_effect = [Mock(status=403),
            Mock(status=200, json=Mock(return_value={'download_url': 'https://files.oaiusercontent.com/image'})),
            Mock(status=200, body=Mock(return_value=b'\x89PNG\r\n\x1a\n'))]
        with tempfile.TemporaryDirectory() as directory:
            result = export_payload(context, 'secret', data, 'test', Path(directory), lambda *a, **kw: None)
            self.assertEqual(result['turns'][0]['items'][0]['content'][0]['text'], 'Question')
            query = context.request.get.call_args_list[1].kwargs['params']
            self.assertEqual(query['conversation_id'], 'test')
            self.assertEqual(query['gizmo_id'], 'g-p-test')
            self.assertNotIn('headers', context.request.get.call_args_list[2].kwargs)

    def test_sandbox_output_is_downloaded_and_linked_without_added_prose(self):
        from unittest.mock import Mock
        data = payload()
        data['mapping']['a']['message']['content']['parts'] = ['[Download](sandbox:/mnt/data/icon.svg)']
        context = Mock()
        svg = b'<svg xmlns="http://www.w3.org/2000/svg"></svg>'
        context.request.get.side_effect = [Mock(status=200, json=Mock(return_value={'download_url': 'https://files.oaiusercontent.com/icon'})),
                                          Mock(status=200, body=Mock(return_value=svg))]
        with tempfile.TemporaryDirectory() as directory:
            result = export_payload(context, 'secret', data, 'test', Path(directory), lambda *a, **kw: None)
            file = result['provenance']['fileAttachments'][0]
            self.assertEqual(Path(file['path']).read_bytes(), svg)
            self.assertEqual(file['placement'], 'assistant-link')
            self.assertEqual(result['turns'][0]['items'][1]['text'], '[Download](<' + file['path'] + '>)')
            self.assertEqual(context.request.get.call_args_list[0].kwargs['params'], {'message_id': 'a', 'sandbox_path': '/mnt/data/icon.svg'})

    def test_internal_execution_and_empty_tool_outputs_stay_in_raw(self):
        for content in ({'content_type': 'execution_output', 'text': 'Internal'}, {'content_type': 'multimodal_text', 'parts': []}):
            data = payload()
            data['mapping']['tool'] = {'parent': 'u', 'message': {'id': 'tool', 'author': {'role': 'tool'}, 'content': content}}
            data['mapping']['a']['parent'] = 'tool'
            result = convert(data, 'test', lambda _: self.fail('Unexpected image'))
            self.assertEqual(len(result['turns'][0]['items']), 2)
            self.assertEqual(result['provenance']['processCardsStoredInRaw'][0]['id'], 'tool')

    def test_pdf_and_signature_are_saved_and_linked_to_original_question(self):
        import hashlib
        from unittest.mock import Mock
        data = payload()
        pdf, signature = b'%PDF-1.7\noriginal', b'\x30\x82signature'
        attachments = [{'id': 'file-pdf', 'name': 'Document.pdf', 'mime_type': 'application/pdf', 'size': len(pdf)},
                       {'id': 'file-sig', 'name': 'Document.pdf.sig', 'size': len(signature)}]
        data['mapping']['u']['message']['metadata'] = {'attachments': attachments}
        context = Mock()
        context.request.get.side_effect = [
            Mock(status=404),
            Mock(status=200, json=Mock(return_value={'download_url': 'https://files.oaiusercontent.com/document'})),
            Mock(status=200, body=Mock(return_value=pdf)),
            Mock(status=200, json=Mock(return_value={'download_url': 'https://files.oaiusercontent.com/signature'})),
            Mock(status=200, body=Mock(return_value=signature)),
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = export_payload(context, 'secret', data, 'test', Path(directory), lambda *a, **kw: None)
            files = source['provenance']['fileAttachments']
            for entry, original in zip(files, (pdf, signature)):
                self.assertEqual(entry['messageId'], 'u')
                self.assertEqual(Path(entry['path']).read_bytes(), original)
                self.assertEqual(Path(entry['path']).parent.name, hashlib.sha256(original).hexdigest())
                self.assertEqual(Path(entry['path']).name, entry['name'])
                self.assertIn(entry['path'], '\n'.join(p['text'] for p in source['turns'][0]['items'][0]['content']))
            self.assertEqual(source['turns'][0]['items'][1]['text'], 'Answer')
            content = source['turns'][0]['items'][0]['content']
            self.assertTrue(content[0]['text'].startswith('# Files mentioned by the user:\n'))
            self.assertEqual(content[1:], [{'type': 'text', 'text': 'Question'}])
        for call in context.request.get.call_args_list:
            if call.args[0].startswith('https://files.'):
                self.assertNotIn('headers', call.kwargs)

    def test_missing_files_have_cards_without_changing_user_text_or_creating_files(self):
        from unittest.mock import Mock
        for status in (404, 410):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                data = payload()
                data['mapping']['u']['message']['metadata'] = {'attachments': [
                    {'id': 'file-pdf', 'name': 'Доверенность.pdf'}, {'id': 'file-sig', 'name': 'Доверенность.pdf.sig'}]}
                context = Mock()
                context.request.get.return_value = Mock(status=status)
                result = export_payload(context, 'secret', data, 'test', Path(directory), lambda *a, **kw: None)
                files = result['provenance']['fileAttachments']
                self.assertEqual(len(files), 2)
                for file in files:
                    self.assertEqual(file['status'], 'unavailable')
                    self.assertEqual(file['httpStatus'], status)
                    self.assertFalse(Path(file['path']).exists())
                    self.assertIn('## ' + file['name'] + ': ' + file['path'], result['turns'][0]['items'][0]['content'][0]['text'])
                self.assertEqual(result['turns'][0]['items'][0]['content'][1:], [{'type': 'text', 'text': 'Question'}])
                self.assertEqual(result['turns'][0]['startedAt'], 101)
                self.assertEqual(result['turns'][0]['items'][1]['text'], 'Answer')

    def test_auth_and_temporary_errors_do_not_become_file_placeholders(self):
        from unittest.mock import Mock
        for status in (401, 403, 429, 500):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                data = payload()
                data['mapping']['u']['message']['metadata'] = {'attachments': [{'id': 'file-pdf', 'name': 'Document.pdf'}]}
                context = Mock()
                context.request.get.return_value = Mock(status=status)
                with self.assertRaisesRegex(RuntimeError, 'HTTP ' + str(status)):
                    export_payload(context, 'secret', data, 'test', Path(directory), lambda *a, **kw: None)
                self.assertTrue((Path(directory) / 'raw.json').exists())
                self.assertFalse((Path(directory) / 'source.json').exists())
                self.assertFalse((Path(directory) / 'unavailable').exists())

    def test_internal_pdf_page_previews_stay_in_raw_not_assistant_answer(self):
        data = payload()
        data['mapping']['preview'] = {'parent': 'u', 'message': {'id': 'preview', 'author': {'role': 'tool', 'name': 'file_search'},
            'content': {'content_type': 'multimodal_text', 'parts': ['Internal PDF page previews', {'content_type': 'image_asset_pointer',
                'asset_pointer': 'sediment://cache#file_document#p_0.page.jpg'}]}}}
        data['mapping']['a']['parent'] = 'preview'
        result = convert(data, 'test', lambda _: self.fail('Internal PDF page must not become a generated image'))
        self.assertEqual(len(result['turns'][0]['items']), 2)
        self.assertEqual(result['turns'][0]['items'][1]['text'], 'Answer')
        self.assertEqual(result['provenance']['processCardsStoredInRaw'][0]['id'], 'preview')

    def test_document_bytes_and_supported_format_are_checked(self):
        self.assertEqual(document_suffix({'name': 'x.pdf', 'size': 5}, b'%PDF-'), '.pdf')
        self.assertEqual(document_suffix({'name': 'x.pdf.sig', 'size': 2}, b'\x30\x00'), '.pdf.sig')
        for attachment, data in [({'name': 'x.pdf'}, b'<html>error'),
                                 ({'name': 'x.pdf', 'size': 500}, b'%PDF-'),
                                 ({'name': 'x.pdf.sig'}, b''),
                                 ({'name': '../x.pdf'}, b'%PDF-'),
                                 ({'name': 'x.pdf\n## injected'}, b'%PDF-')]:
            with self.subTest(attachment=attachment), self.assertRaises(ValueError):
                document_suffix(attachment, data)

    def test_voice_transcripts_preserve_both_speakers_dates_and_order(self):
        data = payload()
        for node, kind in [('u', 'real_time_user_audio_video_asset_pointer'), ('a', 'audio_asset_pointer')]:
            message = data['mapping'][node]['message']
            message['create_time'] = 101 if node == 'u' else 102
            transcript = {'content_type': 'audio_transcription', 'text': 'Question' if node == 'u' else 'Answer'}
            pointer = {'content_type': kind, 'asset_pointer': 'sediment://file-test'}
            message['content'] = {'content_type': 'multimodal_text', 'parts': [pointer, transcript] if node == 'u' else [transcript, pointer]}
        result = convert(data, 'test', lambda _: self.fail('Audio must not be downloaded as an image'))
        exchange = result['turns'][0]
        self.assertEqual(exchange['startedAt'], 101)
        self.assertEqual(exchange['items'][0]['content'], [{'type': 'text', 'text': 'Question'}])
        self.assertEqual(exchange['items'][1]['text'], 'Answer')
        self.assertEqual(exchange['items'][1]['createdAt'], 102)
        self.assertEqual([v['role'] for v in result['provenance']['voiceTranscripts']], ['user', 'assistant'])

    def test_voice_without_transcription_and_video_are_not_silently_lost(self):
        for parts in (
            [{'content_type': 'audio_asset_pointer'}],
            [{'content_type': 'audio_asset_pointer'}, {'content_type': 'audio_transcription', 'text': None}],
            [{'content_type': 'real_time_user_audio_video_asset_pointer', 'frames_asset_pointers': ['frame']}, {'content_type': 'audio_transcription', 'text': 'Text'}],
            [{'content_type': 'real_time_user_audio_video_asset_pointer', 'video_container_asset_pointer': 'video'}, {'content_type': 'audio_transcription', 'text': 'Text'}],
        ):
            with self.subTest(parts=parts):
                data = payload()
                data['mapping']['u']['message']['content'] = {'content_type': 'multimodal_text', 'parts': parts}
                with self.assertRaises(ValueError):
                    convert(data, 'test', lambda _: self.fail('Unexpected download'))

    def test_explicit_empty_voice_transcript_preserves_following_answer(self):
        data = payload()
        data['mapping']['u']['message']['content'] = {'content_type': 'multimodal_text', 'parts': [
            {'content_type': 'audio_transcription', 'text': ''}, {'content_type': 'audio_asset_pointer'}]}
        result = convert(data, 'test', lambda _: self.fail('Unexpected download'))
        self.assertEqual(result['turns'][0]['items'][1]['text'], 'Answer')
        self.assertEqual(result['turns'][0]['items'][0]['content'][0]['text'], '')
        self.assertTrue(result['provenance']['voiceTranscripts'][0]['emptyTranscription'])

    def test_project_listing_checks_all_pages_and_repeated_cursor(self):
        from unittest.mock import Mock
        context = Mock()
        def response(data):
            return Mock(status=200, json=Mock(return_value=data))
        context.request.get.side_effect = [response({'items': [{'id': 'one', 'title': 'One'}], 'cursor': 'next'}), response({'items': [{'id': 'two', 'title': 'Two'}], 'cursor': None})]
        self.assertEqual([i['id'] for i in list_project_chats(context, 'token', 'g-p-test')], ['one', 'two'])
        context.request.get.side_effect = [response({'items': [], 'cursor': 'same'}), response({'items': [], 'cursor': 'same'})]
        with self.assertRaises(ValueError):
            list_project_chats(context, 'token', 'g-p-test')

    def test_unanswered_question_is_preserved_without_invented_answer(self):
        data = payload()
        data['current_node'] = 'u'
        result = convert(data, 'test', lambda _: '')
        self.assertTrue(result['turns'][0]['unanswered'])
        self.assertEqual([i['type'] for i in result['turns'][0]['items']], ['userMessage'])

    def test_network_errors_do_not_expose_credentials(self):
        from unittest.mock import Mock
        context = Mock()
        context.request.get.side_effect = RuntimeError('Authorization: Bearer secret-token')
        with self.assertRaises(RuntimeError) as error:
            http_get(context, 'https://chatgpt.com')
        self.assertNotIn('secret-token', str(error.exception))
        self.assertTrue(error.exception.__suppress_context__)

    def test_generation_internal_call_is_not_imported_as_answer(self):
        data = payload()
        data['mapping']['call'] = {'parent': 'u', 'message': {'id': 'call', 'author': {'role': 'assistant'}, 'channel': 'commentary', 'recipient': 'image-tool', 'content': {'content_type': 'code', 'text': 'internal generation request'}}}
        data['mapping']['a']['parent'] = 'call'
        result = convert(data, 'test', lambda _: self.fail('Unexpected download'))
        self.assertEqual(len(result['turns'][0]['items']), 2)
        self.assertEqual(result['turns'][0]['items'][1]['text'], 'Answer')

    def test_login_launches_normal_chrome_without_automation(self):
        from unittest.mock import patch, Mock
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / 'profile'
            process = Mock(returncode=0)
            process.poll.return_value = 0
            with patch('cloud_export.chrome_executable', return_value='/normal/chrome'), patch('cloud_export.has_login', side_effect=[False, True]), patch('cloud_export.subprocess.Popen', return_value=process) as launch:
                manual_login(profile, 'https://chatgpt.com/c/test', lambda *a, **kw: None)
                args = launch.call_args.args[0]
                self.assertEqual(args[0], '/normal/chrome')
                self.assertIn('--user-data-dir=' + str(profile), args)
                self.assertFalse(any('debug' in arg or 'automation' in arg or 'headless' in arg for arg in args))
                self.assertEqual(profile.stat().st_mode & 0o777, 0o700)

    def test_snapshot_reads_committed_wal_without_changing_source(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'Default').mkdir()
            (root / 'Local State').write_text('{}')
            with closing(sqlite3.connect(root / 'Default/Cookies')) as connection:
                connection.execute('PRAGMA journal_mode=WAL')
                connection.execute('CREATE TABLE cookies(host_key TEXT,name TEXT,encrypted_value BLOB)')
                connection.execute('INSERT INTO cookies VALUES(?,?,?)', ('.chatgpt.com', '__Secure-next-auth.session-token', b'encrypted'))
                connection.commit()
                self.assertTrue(has_login(root))
                snapshot = snapshot_profile(root)
                with closing(sqlite3.connect(snapshot / 'Default/Cookies')) as copy:
                    self.assertEqual(copy.execute('SELECT encrypted_value FROM cookies').fetchone()[0], b'encrypted')
                self.assertEqual(connection.execute('SELECT count(*) FROM cookies').fetchone()[0], 1)

    def test_user_and_assistant_images_stay_with_their_messages(self):
        data = payload()
        pointer = {'content_type': 'image_asset_pointer', 'asset_pointer': 'file-service://file-test'}
        data['mapping']['u']['message']['content'] = {'content_type': 'multimodal_text', 'parts': ['Question', pointer]}
        data['mapping']['a']['message']['content'] = {'content_type': 'multimodal_text', 'parts': [pointer]}
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / 'test.png'
            image.write_bytes(b'\x89PNG\r\n\x1a\n')
            result = convert(data, 'test', lambda _: str(image))
            items = result['turns'][0]['items']
            self.assertEqual(items[0]['content'][1]['path'], str(image))
            self.assertEqual(items[1]['images'], [str(image)])

    def test_current_branch_only_and_source_dates(self):
        result = convert(payload(), 'test', lambda _: self.fail('Unexpected download'))
        self.assertEqual(result['thread']['createdAt'], 100)
        self.assertEqual(result['turns'][0]['startedAt'], 101)
        self.assertEqual(result['turns'][0]['items'][1]['text'], 'Answer')
        self.assertIsNone(result['turns'][0]['completedAt'])

    def test_refuses_incomplete_or_cyclic_history(self):
        for parent in ('missing', 'a'):
            data = payload()
            data['mapping']['u']['parent'] = parent
            with self.assertRaises(ValueError):
                active_messages(data, 'test')

    def test_refuses_unknown_attachment_and_incomplete_answer(self):
        data = payload()
        data['mapping']['u']['message']['metadata'] = {'attachments': [{'mime_type': 'image/png'}]}
        with self.assertRaises(ValueError):
            convert(data, 'test', lambda _: '')
        data = payload()
        data['mapping']['a']['message']['status'] = 'in_progress'
        with self.assertRaises(ValueError):
            convert(data, 'test', lambda _: '')

    def test_url_and_image_validation(self):
        self.assertEqual(chat_id('https://chatgpt.com/c/test'), 'test')
        self.assertEqual(chat_id('https://chatgpt.com/share/test/'), 'test')
        for url in ('https://evil.test/c/test', 'http://chatgpt.com/c/test', 'https://me@chatgpt.com/c/test'):
            with self.assertRaises(ValueError):
                chat_id(url)
        self.assertFalse(allowed_asset_url('https://chatgpt.com.evil.test/image'))
        self.assertFalse(allowed_asset_url('http://chatgpt.com/image'))
        self.assertEqual(image_suffix(b'\x89PNG\r\n\x1a\n'), '.png')
        with self.assertRaises(ValueError):
            image_suffix(b'<html>Login</html>')


class RegistrationTests(unittest.TestCase):
    def test_explicit_catalog_selection_preserves_other_database(self):
        import sqlite3
        from cloud_to_local_chat import catalog_database, reset_catalog_sync
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'sqlite').mkdir()
            for name in ('codex.db', 'codex-dev.db'):
                with sqlite3.connect(root / 'sqlite' / name) as db:
                    db.execute('CREATE TABLE local_thread_catalog_sync_state(host_id TEXT,watermark_updated_at REAL,initial_build_complete INTEGER,last_full_reconciled_at REAL)')
                    db.executemany('INSERT INTO local_thread_catalog_sync_state VALUES(?,?,?,?)', [('local', 100, 1, 200), ('other', 300, 1, 400)])
            selected = root / 'sqlite/codex.db'
            with self.assertRaises(ValueError):
                catalog_database(root)
            result = reset_catalog_sync(root / 'data', root, selected)
            self.assertTrue(Path(result['backup']).is_file())
            with sqlite3.connect(selected) as db:
                self.assertEqual(db.execute("SELECT * FROM local_thread_catalog_sync_state WHERE host_id='local'").fetchone(), ('local', None, 0, None))
                self.assertEqual(db.execute("SELECT * FROM local_thread_catalog_sync_state WHERE host_id='other'").fetchone(), ('other', 300, 1, 400))
            with sqlite3.connect(root / 'sqlite/codex-dev.db') as db:
                self.assertEqual(db.execute("SELECT watermark_updated_at FROM local_thread_catalog_sync_state WHERE host_id='local'").fetchone(), (100,))
            with self.assertRaises(ValueError):
                catalog_database(root, root / 'sqlite')

    def test_registration_retries_pending_results_and_confirms_live_without_reset(self):
        import json
        from unittest.mock import patch
        from cloud_to_local_chat import finish_registrations
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = root / 'register.json'
            request.write_text(json.dumps({'project': {'id': 'project'}, 'threadId': 'thread', 'codexHome': str(root)}))
            with patch('cloud_to_local_chat.register_project', side_effect=[{'pending': True}, {'live': True}]) as register, patch('cloud_to_local_chat.time.sleep') as sleep, patch('cloud_to_local_chat.reset_catalog_sync') as reset:
                result = finish_registrations([request], root / 'data', wait=True)
                self.assertEqual(result['status'], 'registered')
                self.assertEqual(register.call_count, 2)
                sleep.assert_called_once_with(0.5)
                reset.assert_not_called()
                self.assertEqual(json.loads(request.read_text())['status'], 'registered')

    def test_pending_catalog_does_not_mark_request_registered(self):
        import json
        from unittest.mock import patch
        from cloud_to_local_chat import finish_registrations
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = root / 'register.json'
            original = json.dumps({'project': {'id': 'project'}, 'threadId': 'thread', 'codexHome': str(root)})
            request.write_text(original)
            with patch('cloud_to_local_chat.register_project', return_value={'projectId': 'project'}), patch('cloud_to_local_chat.reset_catalog_sync', return_value={'pending': True}):
                result = finish_registrations([request], root / 'data')
                self.assertEqual(result['status'], 'pending')
                self.assertEqual(request.read_text(), original)
                with self.assertRaises(TimeoutError):
                    finish_registrations([request], root / 'data', wait=True, timeout=0)

    def test_import_index_is_populated_and_live_registration_requires_visible_user_history(self):
        import json
        import sqlite3
        from cloud_to_local_chat import build_rollout, prepare_source, restore_dates, live_registration
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = convert(payload(), 'test', lambda _: self.fail('No images expected'))
            journal = root / 'thread.jsonl'
            items = prepare_source(source)
            rows = [{'ordinal': 0, 'timestamp': '2026-01-01T00:00:00.000Z', 'type': 'session_meta', 'payload': {'id': 'thread', 'timestamp': '2026-01-01T00:00:00.000Z'}}] + [
                {'ordinal': n+1, 'timestamp': '2026-01-01T00:00:00.000Z', 'type': 'response_item', 'payload': i} for n, i in enumerate(items)]
            journal.write_text(''.join(json.dumps(r) + '\n' for r in build_rollout(rows, source, 'thread', items)))
            project = {'id': 'native', 'roots': [{'path': str(root)}]}
            with sqlite3.connect(root / 'state_5.sqlite') as db:
                db.execute('CREATE TABLE threads(id TEXT,rollout_path TEXT,created_at INTEGER,updated_at INTEGER,created_at_ms INTEGER,updated_at_ms INTEGER,has_user_event INTEGER,title TEXT,first_user_message TEXT,preview TEXT,archived INTEGER,project_id TEXT)')
                db.execute('INSERT INTO threads VALUES(?,?,?,?,?,?,?,?,?,?,?,?)', ('thread', str(journal), 200, 201, 200000, 201000, 0, '', '', '', 0, 'native'))
            (root / 'sqlite').mkdir()
            with sqlite3.connect(root / 'sqlite/codex.db') as db:
                db.execute('CREATE TABLE local_thread_catalog(host_id TEXT,thread_id TEXT,missing_candidate INTEGER)')
                db.execute("INSERT INTO local_thread_catalog VALUES('local','thread',0)")
            (root / '.codex-global-state.json').write_text(json.dumps({'thread-project-assignments': {'thread': {'projectKind': 'local', 'projectId': 'legacy'}}, 'local-projects': {'legacy': {'rootPaths': [str(root)]}}}))
            self.assertIsNone(live_registration(project, 'thread', root))
            restore_dates('thread', source, root / 'data', root)
            with sqlite3.connect(root / 'state_5.sqlite') as db:
                self.assertEqual(db.execute('SELECT has_user_event,title,first_user_message,preview FROM threads').fetchone(), (1, 'Test', 'Question', 'Question'))
            self.assertTrue(live_registration(project, 'thread', root)['live'])
            with sqlite3.connect(root / 'sqlite/codex.db') as db:
                db.execute('UPDATE local_thread_catalog SET missing_candidate=1')
            self.assertIsNone(live_registration(project, 'thread', root))
