# ===========================================================================
# Скрипт: построение MANIFEST.json — карты целостности репозитория.
#
# Для каждого отслеживаемого файла считается SHA-256 и записывается в
# MANIFEST.json (относительный путь -> хеш). Это позволяет проверяющему
# убедиться, что файлы не подменены, и что состав репозитория полный.
#
# Исключаются только заведомо временные/окруженческие пути (см. .gitignore):
# кэш Python, виртуальное окружение, кэш pytest, служебные каталоги и сам
# MANIFEST.json (иначе он хешировал бы сам себя).
#
# Запуск из корня проекта:
#     python scripts/16_build_manifest.py
#
# Проверка (0 несовпадений — целостность подтверждена):
#     python scripts/16_build_manifest.py --check
# ===========================================================================

import os
import sys
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "MANIFEST.json"

# Каталоги, которые не входят в карту целостности (временные/окруженческие).
EXCLUDED_DIRECTORY_NAMES = [
    "__pycache__",
    ".venv",
    ".git",
    ".pytest_cache",
    "models_cache",
    ".ipynb_checkpoints",
]

# Отдельные файлы, которые не хешируются.
EXCLUDED_FILE_NAMES = [
    "MANIFEST.json",
    ".DS_Store",
]

# Расширения, которые не хешируются (скомпилированный байткод).
EXCLUDED_SUFFIXES = [
    ".pyc",
]


def path_is_excluded(relative_path):
    """Проверить, нужно ли исключить файл из карты целостности."""
    parts = relative_path.split("/")
    for part in parts:
        if part in EXCLUDED_DIRECTORY_NAMES:
            return True
    file_name = parts[-1]
    if file_name in EXCLUDED_FILE_NAMES:
        return True
    for suffix in EXCLUDED_SUFFIXES:
        if file_name.endswith(suffix):
            return True
    return False


def compute_sha256(file_path):
    """Посчитать SHA-256 файла, читая его блоками (память не раздувается)."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as source:
        while True:
            chunk = source.read(65536)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def collect_manifest():
    """Собрать словарь {относительный путь: sha256} по всему дереву проекта."""
    collected = {}
    for current_directory, subdirectory_names, file_names in os.walk(ROOT):
        # Отсекаем исключённые каталоги прямо в обходе (быстрее и чище).
        kept_subdirectories = []
        for subdirectory_name in subdirectory_names:
            if subdirectory_name not in EXCLUDED_DIRECTORY_NAMES:
                kept_subdirectories.append(subdirectory_name)
        subdirectory_names[:] = kept_subdirectories

        for file_name in file_names:
            absolute_path = Path(current_directory) / file_name
            relative_path = str(absolute_path.relative_to(ROOT)).replace(os.sep, "/")
            if path_is_excluded(relative_path):
                continue
            collected[relative_path] = compute_sha256(absolute_path)
    return collected


def write_manifest(manifest_dictionary):
    """Записать MANIFEST.json с сортировкой ключей для детерминированности."""
    ordered = {}
    for key in sorted(manifest_dictionary.keys()):
        ordered[key] = manifest_dictionary[key]
    text = json.dumps(ordered, ensure_ascii=False, indent=2)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as target:
        target.write(text + "\n")


def check_manifest(manifest_dictionary):
    """Сверить текущие файлы с сохранённым MANIFEST.json.

    Возвращает число несовпадений (отсутствующие, лишние, изменённые файлы).
    """
    if not MANIFEST_PATH.exists():
        print("MANIFEST.json отсутствует — нечего сверять.")
        return 1

    with open(MANIFEST_PATH, "r", encoding="utf-8") as source:
        saved = json.load(source)

    mismatch_count = 0

    for relative_path in sorted(saved.keys()):
        if relative_path not in manifest_dictionary:
            print("ПРОПАЛ ФАЙЛ: " + relative_path)
            mismatch_count += 1
        elif manifest_dictionary[relative_path] != saved[relative_path]:
            print("ИЗМЕНЁН ФАЙЛ: " + relative_path)
            mismatch_count += 1

    for relative_path in sorted(manifest_dictionary.keys()):
        if relative_path not in saved:
            print("НОВЫЙ ФАЙЛ (нет в манифесте): " + relative_path)
            mismatch_count += 1

    return mismatch_count


def main():
    manifest_dictionary = collect_manifest()

    check_mode = "--check" in sys.argv[1:]
    if check_mode:
        mismatch_count = check_manifest(manifest_dictionary)
        print("Несовпадений: " + str(mismatch_count))
        if mismatch_count > 0:
            sys.exit(1)
        print("Целостность подтверждена: 0 несовпадений.")
        return

    write_manifest(manifest_dictionary)
    print("MANIFEST.json обновлён. Файлов в карте целостности: "
          + str(len(manifest_dictionary)))


if __name__ == "__main__":
    main()
