"""Мост к коду проекта sunday (П3: тракт не переписываем — импортируем).

Осторожно с коллизией имён: пакет `datawizard` есть и в sunday, и в
rlbinancetrader (старый). В одном процессе может жить ТОЛЬКО ОДИН из них.
ensure_sunday_first() ставит sunday в начало sys.path и проверяет, что
`datawizard` разрешился именно в sunday; если старый datawizard уже
импортирован — падает сразу, а не молча работает с чужим кодом.
"""
import importlib
import os
import re
import sys

SUNDAY_ROOT = "/home/cubecloud/Python/projects/sunday"
_PG_ENV_FILES = ("PSGSQL_KEY.env", "PSGSQLKEYS.env")
_PG_REQUIRED_VARS = ("PSGSQL_KEY", "PSGSQLPARSE_USERNAME", "PSGSQLPARSE_PASSWORD",
                     "PSGSQLPARSE_USERNAME_IV", "PSGSQLPARSE_PASSWORD_IV")

_ENV_LINE = re.compile(r"^(?:export\s+)?([A-Z_][A-Z0-9_]*)=(.*)$")


def load_pg_env() -> None:
    """Догружает PG-креды из *.env sunday, если их нет в окружении процесса.

    Значения в файлах зашифрованы (secureapikey) — сюда попадают как есть,
    расшифровку делает dbbinance. Ничего не печатаем.
    """
    missing = [v for v in _PG_REQUIRED_VARS if v not in os.environ]
    if not missing:
        return
    for fname in _PG_ENV_FILES:
        path = os.path.join(SUNDAY_ROOT, fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                m = _ENV_LINE.match(line.strip())
                if m and m.group(1) not in os.environ:
                    os.environ[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    still = [v for v in _PG_REQUIRED_VARS if v not in os.environ]
    if still:
        raise RuntimeError(f"PG creds not found (missing {still}); "
                           f"check {SUNDAY_ROOT}/*.env")


def ensure_sunday_first() -> None:
    """sunday в начало sys.path + защита от чужого datawizard."""
    if "datawizard" in sys.modules:
        loaded = getattr(sys.modules["datawizard"], "__file__", "") or ""
        if SUNDAY_ROOT not in loaded:
            raise ImportError(
                "Пакет datawizard уже импортирован из rlbinancetrader "
                f"({loaded}); sunday-модули в этом процессе использовать "
                "нельзя. Разделяйте процессы (см. докстринг sundaybridge).")
    if sys.path[0] != SUNDAY_ROOT:
        while SUNDAY_ROOT in sys.path:
            sys.path.remove(SUNDAY_ROOT)
        sys.path.insert(0, SUNDAY_ROOT)


def sunday_module(name: str):
    """Импорт модуля sunday с гарантией правильного datawizard.

    >>> pt = sunday_module("datawizard.powertrend")
    >>> leg = pt.TrendLegPowerPctDF()
    """
    ensure_sunday_first()
    load_pg_env()
    mod = importlib.import_module(name)
    if name.split(".")[0] == "datawizard":
        assert SUNDAY_ROOT in (mod.__file__ or ""), \
            f"{name} resolved outside sunday: {mod.__file__}"
    return mod


def sunday_git_head() -> str:
    """Текущий коммит sunday-репо — для манифестов кэша (П1)."""
    import subprocess
    out = subprocess.run(["git", "-C", SUNDAY_ROOT, "rev-parse", "HEAD"],
                        capture_output=True, text=True, check=True)
    return out.stdout.strip()
