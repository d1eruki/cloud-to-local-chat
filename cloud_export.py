"""Получение одного чата через отдельный браузер; формат ChatGPT экспериментальный."""

import hashlib
from contextlib import closing
import json
import math
import re
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse


def chat_id(url):
    parsed = urlparse(url)
    match = re.fullmatch(r'/c/([a-zA-Z0-9-]+)', parsed.path.rstrip('/'))
    if parsed.scheme != 'https' or parsed.hostname != 'chatgpt.com' or parsed.username or parsed.password or parsed.port not in (None, 443) or not match:
        raise ValueError('Нужна ссылка https://chatgpt.com/c/… на собственный чат.')
    return match.group(1)


def active_messages(payload, identifier):
    if payload.get('conversation_id', payload.get('id')) != identifier:
        raise ValueError('Получен другой чат.')
    mapping = payload.get('mapping', {})
    current = payload.get('current_node')
    seen, messages = set(), []
    while current:
        if current in seen or current not in mapping:
            raise ValueError('Повреждена или неполна цепочка сообщений.')
        seen.add(current)
        node = mapping[current]
        if node.get('message'):
            messages.append(node['message'])
        current = node.get('parent')
    if not messages:
        raise ValueError('История пуста или формат ChatGPT изменился.')
    return list(reversed(messages))


def convert(payload, identifier, get_image, get_file=None):
    """Выбирает текущую ветку; не переносит скрытые сообщения и старые варианты."""
    turns, turn, process_cards, voice_transcripts = [], None, [], []
    file_attachments = []
    for message in active_messages(payload, identifier):
        role = message.get('author', {}).get('role')
        metadata = message.get('metadata', {})
        if role == 'system' or metadata.get('is_visually_hidden_from_conversation'):
            continue
        content = message.get('content', {})
        kind = content.get('content_type')
        if role not in ('user', 'assistant', 'tool'):
            raise ValueError('Неизвестная роль в истории.')
        if role == 'assistant' and kind == 'code' and message.get('recipient') not in (None, 'all'):
            # Вызов генератора не является видимым ответом. Его результат ниже
            # должен быть изображением; другие результаты отклоняются.
            continue
        if (role == 'assistant' and kind in ('reasoning_recap', 'thoughts')) or (role == 'tool' and kind in ('text', 'tether_quote', 'tether_browsing_display')):
            # Отдельные карточки процесса остаются в полном raw.json.
            # В диалог импортируются вопросы и ответы; пропуск фиксируется.
            process_cards.append({'id': message['id'], 'contentType': kind})
            continue
        if (role == 'tool' and message.get('author', {}).get('name') == 'file_search'
                and kind == 'multimodal_text' and content.get('parts')
                and any(isinstance(p, dict) for p in content['parts'])
                and all(isinstance(p, str) or (isinstance(p, dict) and p.get('content_type') == 'image_asset_pointer'
                        and re.search(r'#file[-_][a-zA-Z0-9_-]+#p_\d+\.', p.get('asset_pointer', '')))
                        for p in content['parts'])):
            process_cards.append({'id': message['id'], 'contentType': kind,
                                  'reason': 'internal-document-page-previews'})
            continue
        if role == 'assistant' and message.get('channel') == 'analysis':
            raise ValueError('Этот чат содержит отдельные рассуждения или действия; перенос пока не поддерживается.')
        if kind not in ('text', 'multimodal_text'):
            raise ValueError('Неподдерживаемое содержимое: ' + str(kind))
        parts, images, audio_types = [], [], []
        has_transcription = False
        empty_transcription = False
        for part in content.get('parts', []):
            if isinstance(part, str):
                parts.append({'type': 'text', 'text': part})
            elif isinstance(part, dict) and part.get('content_type') == 'image_asset_pointer':
                images.append(get_image(part['asset_pointer']))
            elif isinstance(part, dict) and part.get('content_type') == 'audio_transcription':
                text = part.get('text')
                if not isinstance(text, str):
                    raise ValueError('У голосового сообщения отсутствует текстовая расшифровка.')
                if not text.strip():
                    empty_transcription = True
                parts.append({'type': 'text', 'text': text})
                has_transcription = True
            elif isinstance(part, dict) and part.get('content_type') in ('audio_asset_pointer', 'real_time_user_audio_video_asset_pointer'):
                if part.get('frames_asset_pointers') or part.get('video_container_asset_pointer'):
                    raise ValueError('Видеосодержимое пока не поддерживается.')
                audio_types.append(part['content_type'])
            else:
                raise ValueError('В сообщении есть неподдерживаемое вложение.')
        if audio_types and not has_transcription:
            raise ValueError('У голосового сообщения отсутствует текстовая расшифровка.')
        if has_transcription:
            voice_transcripts.append({'id': message['id'], 'role': role,
                                      'audioContentTypesStoredInRaw': audio_types,
                                      'emptyTranscription': empty_transcription})
        if metadata.get('attachments'):
            image_attachments = 0
            file_cards = []
            for attachment in metadata['attachments']:
                if str(attachment.get('mime_type', '')).startswith('image/'):
                    image_attachments += 1
                    continue
                document_suffix(attachment)
                if role != 'user' or get_file is None:
                    raise ValueError('Для вложенного документа требуется скачивание оригинала.')
                result = get_file(attachment)
                if not isinstance(result, dict):
                    result = {'path': str(result), 'status': 'available'}
                if result.get('status') not in ('available', 'unavailable'):
                    raise ValueError('Неизвестный статус вложения.')
                unavailable = result['status'] == 'unavailable'
                path = Path(result['path']).resolve(strict=not unavailable)
                if unavailable and path.exists():
                    raise ValueError('Путь недоступного файла уже существует.')
                file_cards.append('## ' + attachment['name'] + ': ' + str(path))
                file_attachments.append({**result, 'messageId': message['id'], 'id': attachment['id'],
                                         'name': attachment['name'], 'mimeType': attachment.get('mime_type'),
                                         'sourceSize': attachment.get('size'), 'path': str(path)})
            if image_attachments > len(images):
                raise ValueError('Не все вложения сопоставлены с оригиналами изображений.')
            if file_cards:
                parts.insert(0, {'type': 'text', 'text': '# Files mentioned by the user:\n\n'
                                + '\n\n'.join(file_cards) + '\n\n## My request:\n'})
        if role == 'tool' and (parts or not images):
            raise ValueError('Текстовые ответы инструментов пока не поддерживаются.')
        if role == 'user':
            stamp = message.get('create_time')
            if not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
                raise ValueError('У вопроса отсутствует исходная дата.')
            turn = {'id': message['id'], 'status': 'completed', 'error': None,
                    'startedAt': stamp, 'completedAt': None, 'items': []}
            turns.append(turn)
            turn['items'].append({'type': 'userMessage', 'id': message['id'],
                                  'content': parts + [{'type': 'localImage', 'path': p} for p in images]})
        else:
            if not turn:
                raise ValueError('Ответ без исходного вопроса.')
            if message.get('status') not in ('finished_successfully', None):
                raise ValueError('Ответ не завершён.')
            text = '\n'.join(p['text'] for p in parts)
            if text or images:
                turn['items'].append({'type': 'agentMessage', 'id': message['id'],
                                      'text': text, 'images': images,
                                      'createdAt': message.get('create_time')})
            # update_time может означать редактирование; не выдаём его за время ответа.
    for exchange in turns:
        if len(exchange['items']) == 1:
            # Исходные последовательные вопросы также сохраняются раздельно.
            exchange['unanswered'] = True
    source = {'thread': {'id': identifier, 'title': payload.get('title'),
                         'createdAt': payload.get('create_time'), 'updatedAt': payload.get('update_time')},
              'page': {'hasMore': False}, 'turns': turns,
              'provenance': {'method': 'authenticated-conversation-response', 'branch': payload.get('current_node'),
                             'processCardsStoredInRaw': process_cards,
                             'voiceTranscripts': voice_transcripts,
                             'fileAttachments': file_attachments}}
    from cloud_to_local_chat import prepare_source
    prepare_source(source)
    return source


def document_suffix(attachment, data=None):
    name = attachment.get('name')
    if (not isinstance(name, str) or not name or name in ('.', '..')
            or any(c in name for c in ('/', '\\', '\n', '\r', '\x00'))):
        raise ValueError('У документа отсутствует имя.')
    mime = attachment.get('mime_type')
    if name.lower().endswith('.pdf.sig') and mime in (None, 'application/octet-stream', 'application/pkcs7-signature'):
        suffix = '.pdf.sig'
    elif name.lower().endswith('.pdf') and mime in (None, 'application/pdf'):
        suffix = '.pdf'
    else:
        suffix = Path(name).suffix.lower()
    if data is not None:
        size = attachment.get('size')
        if not data or (size is not None and (type(size) is not int or len(data) != size)):
            raise ValueError('Размер скачанного документа отличается от исходного.')
        if suffix == '.pdf' and not data.startswith(b'%PDF-'):
            raise ValueError('Вместо PDF получен неизвестный файл или страница ошибки.')
    return suffix


def image_suffix(data):
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return '.png'
    if data.startswith(b'\xff\xd8\xff'):
        return '.jpeg'
    if data[:6] in (b'GIF87a', b'GIF89a'):
        return '.gif'
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return '.webp'
    raise ValueError('Вместо картинки получен неизвестный файл или страница ошибки.')


def allowed_asset_url(url):
    parsed = urlparse(url)
    host = parsed.hostname or ''
    return parsed.scheme == 'https' and not parsed.username and not parsed.password and (
        host == 'chatgpt.com' or host.endswith('.oaiusercontent.com') or
        host.endswith('.openai.com') or host.endswith('.blob.core.windows.net'))


def chrome_executable():
    candidates = [Path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')]
    for name in ('google-chrome', 'google-chrome-stable', 'chrome', 'chrome.exe'):
        executable = shutil.which(name)
        if executable:
            candidates.append(Path(executable))
    for executable in candidates:
        if executable.is_file():
            return str(executable)
    raise RuntimeError('Для входа нужен установленный обычный Google Chrome.')


def manual_login(profile, url, progress=print):
    """Обычный Chrome: без Playwright, порта отладки и управления во время входа."""
    executable = chrome_executable()
    profile.mkdir(parents=True, exist_ok=True, mode=0o700)
    profile.chmod(0o700)
    if has_login(profile):
        progress('[1/4] Сохранённая сессия входа обнаружена.', flush=True)
        return
    if (profile / 'SingletonLock').exists() or (profile / 'SingletonLock').is_symlink():
        raise RuntimeError('Отдельное окно входа уже открыто. Завершите вход в нём и повторите запуск.')
    progress('[1/4] Войдите в ChatGPT в обычном Chrome. Сохранение сессии определяется автоматически; закрывать рабочий браузер не нужно.', flush=True)
    process = subprocess.Popen([executable, '--user-data-dir=' + str(profile), '--no-first-run',
                                '--no-default-browser-check', url],
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 600
    while not has_login(profile):
        if process.poll() is not None:
            raise RuntimeError('Окно входа завершилось до сохранения сессии ChatGPT.')
        if time.monotonic() > deadline:
            raise TimeoutError('Сессия ChatGPT не появилась за 10 минут. Chrome оставлен открытым.')
        time.sleep(0.5)


def cookie_database(profile):
    for relative in ('Default/Cookies', 'Default/Network/Cookies'):
        path = profile / relative
        if path.exists():
            return path
    return None


def has_login(profile):
    path = cookie_database(profile)
    if path is None:
        return False
    try:
        with closing(sqlite3.connect('file:' + str(path.resolve()) + '?mode=ro', uri=True, timeout=1)) as connection:
            return bool(connection.execute("SELECT 1 FROM cookies WHERE (host_key IN ('chatgpt.com','.chatgpt.com','auth.openai.com','.auth.openai.com')) AND (name LIKE '__Secure-next-auth.session-token%' OR name LIKE '__Secure-authjs.session-token%') LIMIT 1").fetchone())
    except sqlite3.Error:
        return False


def snapshot_profile(profile):
    """Согласованный SQLite-снимок сессии только выделенного профиля утилиты."""
    cookie_path = cookie_database(profile)
    if cookie_path is None or not (profile / 'Local State').exists():
        raise RuntimeError('Сохранённая сессия Chrome не найдена.')
    target = profile / 'background'
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    target.chmod(0o700)
    state = (profile / 'Local State').read_bytes()
    json.loads(state)  # Не копируем недописанный файл.
    (target / 'Local State').write_bytes(state)
    destination = target / cookie_path.relative_to(profile)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect('file:' + str(cookie_path.resolve()) + '?mode=ro', uri=True, timeout=5)) as source:
        with closing(sqlite3.connect(destination)) as copy:
            source.backup(copy)
    return target


def http_get(context, url, **options):
    try:
        return context.request.get(url, **options)
    except Exception:
        # Сообщение Playwright может содержать заголовки авторизации и
        # подписанный URL. Не выводим их в консоль или цепочку исключений.
        raise RuntimeError('Сетевой запрос не выполнен. Проверьте соединение и повторите позже.') from None


def list_project_chats(context, token, project_id):
    if not re.fullmatch(r'g-p-[a-zA-Z0-9]+', project_id):
        raise ValueError('Нужен идентификатор облачного проекта g-p-…')
    cursor, cursors, result, identifiers = None, set(), [], set()
    while True:
        params = {'limit': 20, 'owned_only': 'true'}
        if cursor:
            params['cursor'] = cursor
        response = http_get(context, 'https://chatgpt.com/backend-api/gizmos/' + project_id + '/conversations',
                            headers={'Authorization': 'Bearer ' + token}, params=params, max_redirects=0)
        if response.status != 200:
            raise RuntimeError('Список чатов недоступен (HTTP ' + str(response.status) + ').')
        payload = response.json()
        if not isinstance(payload.get('items'), list):
            raise ValueError('Формат списка чатов изменился.')
        for item in payload['items']:
            identifier = item.get('id')
            if not identifier or identifier in identifiers:
                raise ValueError('Список содержит отсутствующий или повторный идентификатор.')
            if item.get('gizmo_id') not in (None, project_id):
                raise ValueError('В списке найден чат другого проекта.')
            identifiers.add(identifier)
            result.append({'id': identifier, 'title': item.get('title')})
        cursor = payload.get('cursor')
        if not cursor:
            return result
        if cursor in cursors:
            raise ValueError('Сайт повторил курсор страницы; список не считается полным.')
        cursors.add(cursor)


def export_chat(url, data_dir, progress=print):
    identifier = chat_id(url)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as error:
        raise RuntimeError('Установите зависимости: python3 -m pip install -r requirements.txt') from error
    output = Path(data_dir).resolve() / ('cloud-' + identifier)
    source_path = output / 'source.json'
    if source_path.exists():
        raise ValueError('Источник уже сохранён: ' + str(source_path))
    profile = Path(data_dir).resolve() / 'browser-profile'
    ready = profile / '.chatgpt-login-ready'
    executable = chrome_executable()
    if not ready.exists():
        manual_login(profile, url, progress)
    else:
        progress('[1/4] Используется сохранённый отдельный профиль Chrome.', flush=True)
    background = snapshot_profile(profile)
    with sync_playwright() as runtime:
        progress('[2/4] Получение истории и файлов в фоне.', flush=True)
        context = runtime.chromium.launch_persistent_context(str(background), executable_path=executable, headless=True,
                    ignore_default_args=['--use-mock-keychain', '--password-store=basic'])
        try:
            session = http_get(context, 'https://chatgpt.com/api/auth/session', timeout=20000)
            if session.status != 200:
                raise RuntimeError('Сессия входа недоступна (HTTP ' + str(session.status) + ').')
            token = session.json().get('accessToken')
            if not token:
                raise RuntimeError('Сессия истекла или не прочитана. Требуется вход в отдельном профиле.')
            response = http_get(context, 'https://chatgpt.com/backend-api/conversation/' + identifier,
                        headers={'Authorization': 'Bearer ' + token}, timeout=30000, max_redirects=0)
            if response.status != 200:
                raise RuntimeError('История недоступна (HTTP ' + str(response.status) + '). Перенос остановлен.')
            payload = response.json()
            source = export_payload(context, token, payload, identifier, output, progress)
            ready.write_text('Сессия проверена получением истории.\n', encoding='utf-8')
            progress('[3/4] История подготовлена: ' + str(len(source['turns'])) + ' обменов, ' + str(sum(len(i.get('images', [])) + sum(c.get('type') == 'localImage' for c in i.get('content', [])) for t in source['turns'] for i in t['items'])) + ' изображений.', flush=True)
            return source
        finally:
            context.close()


def export_payload(context, token, payload, identifier, output, progress=print):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "raw.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    cache = {}

    def download_file(attachment):
        suffix = document_suffix(attachment)
        identifier = attachment.get('id', '')
        if not isinstance(identifier, str) or not re.fullmatch(r'file[-_][a-zA-Z0-9_-]+', identifier):
            raise ValueError('Неподдерживаемый идентификатор документа.')
        key = (identifier, suffix)
        if key in cache:
            return cache[key]
        def unavailable(status):
            path = output / 'unavailable' / identifier / attachment['name']
            if path.exists():
                raise ValueError('Путь заглушки уже существует; требуется сверка вложения.')
            result = {'path': str(path), 'status': 'unavailable', 'httpStatus': status}
            cache[key] = result
            progress('  Файл недоступен (HTTP ' + str(status) + '); сохранена карточка вложения.', flush=True)
            return result
        resolver = http_get(context, 'https://chatgpt.com/backend-api/files/' + identifier + '/download',
                            headers={'Authorization': 'Bearer ' + token}, max_redirects=0)
        if resolver.status == 404:
            resolver = http_get(context, 'https://chatgpt.com/backend-api/files/download/' + identifier,
                                headers={'Authorization': 'Bearer ' + token}, max_redirects=0,
                                params={'conversation_id': payload.get('conversation_id', payload.get('id')),
                                        'download_intent': 'true', 'inline': 'false'})
        if resolver.status in (404, 410):
            return unavailable(resolver.status)
        if resolver.status != 200:
            raise RuntimeError('Оригинал документа недоступен (HTTP ' + str(resolver.status) + '). Перенос остановлен.')
        asset_url = resolver.json().get('download_url', '')
        if not allowed_asset_url(asset_url):
            raise ValueError('Неизвестный адрес сервера документов; файл не скачан.')
        response = http_get(context, asset_url, max_redirects=0, timeout=120000)
        if response.status in (404, 410):
            return unavailable(response.status)
        if response.status != 200:
            raise RuntimeError('Не удалось скачать документ (HTTP ' + str(response.status) + ').')
        data = response.body()
        document_suffix(attachment, data)
        checksum = hashlib.sha256(data).hexdigest()
        directory = output / 'files' / checksum
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / attachment['name']
        path.write_bytes(data)
        cache[key] = {'path': str(path), 'status': 'available', 'sha256': checksum}
        progress('  Документ: сохранён (' + str(len(data)) + ' байт).', flush=True)
        return cache[key]

    def download(pointer):
        if pointer in cache:
            return cache[pointer]
        match = re.fullmatch(r'(?:file-service://|sediment://)(file[-_][a-zA-Z0-9_-]+)', pointer)
        if not match:
            raise ValueError('Неподдерживаемая ссылка на оригинал изображения.')
        resolver = http_get(context, 'https://chatgpt.com/backend-api/files/' + match.group(1) + '/download',
                                       headers={'Authorization': 'Bearer ' + token}, max_redirects=0)
        if resolver.status != 200:
            raise RuntimeError('Оригинал изображения недоступен (HTTP ' + str(resolver.status) + '). Перенос остановлен.')
        asset_url = resolver.json().get('download_url', '')
        if not allowed_asset_url(asset_url):
            raise ValueError('Неизвестный адрес сервера изображений; файл не скачан.')
        # Bearer-токен передаётся только ChatGPT, не серверу хранения.
        response = http_get(context, asset_url, max_redirects=0, timeout=120000)
        if response.status != 200:
            raise RuntimeError('Не удалось скачать изображение (HTTP ' + str(response.status) + ').')
        data = response.body()
        suffix = image_suffix(data)
        directory = output / 'images'
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (hashlib.sha256(data).hexdigest() + suffix)
        path.write_bytes(data)
        cache[pointer] = str(path)
        progress('  Картинка ' + str(sum(isinstance(key, str) for key in cache)) + ': сохранена (' + str(len(data)) + ' байт).', flush=True)
        return str(path)

    source = convert(payload, identifier, download, download_file)
    (output / "source.json").write_text(json.dumps(source, ensure_ascii=False, indent=2), encoding="utf-8")
    return source
