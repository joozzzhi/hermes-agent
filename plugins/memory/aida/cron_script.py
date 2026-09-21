"""Запускалка ночной работы для расписания.

`hermes cron` берёт скрипты только из папки профиля, поэтому копия этого файла кладётся
в `<HERMES_HOME>/scripts/aida_nightly.py`. Здесь он лежит, чтобы на новой машине его было
откуда взять, и чтобы правка жила в репозитории, а не только на одном ноуте.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _load_job():
    try:
        from plugins.memory.aida import nightly_job
    except ImportError:
        # Скрипт запускают из папки профиля, где о репозитории агента ничего не известно.
        import hermes_constants  # лежит в корне репозитория агента

        sys.path.insert(0, str(Path(hermes_constants.__file__).resolve().parent))
        from plugins.memory.aida import nightly_job
    return nightly_job


if __name__ == "__main__":
    raise SystemExit(_load_job().main([]))
