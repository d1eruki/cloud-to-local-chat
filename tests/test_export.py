import unittest
from contextlib import closing
import tempfile
from pathlib import Path
from cloud_export import active_messages, chat_id, convert, image_suffix, allowed_asset_url, manual_login, has_login, snapshot_profile, http_get, list_project_chats, document_suffix, export_payload


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
        for url in ('https://evil.test/c/test', 'http://chatgpt.com/c/test', 'https://me@chatgpt.com/c/test'):
            with self.assertRaises(ValueError):
                chat_id(url)
        self.assertFalse(allowed_asset_url('https://chatgpt.com.evil.test/image'))
        self.assertFalse(allowed_asset_url('http://chatgpt.com/image'))
        self.assertEqual(image_suffix(b'\x89PNG\r\n\x1a\n'), '.png')
        with self.assertRaises(ValueError):
            image_suffix(b'<html>Login</html>')
