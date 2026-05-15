from pathlib import Path

def get_pkg_abs_path():
    return Path(__file__).parent.parent.resolve()