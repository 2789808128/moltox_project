from pathlib import Path


def get_project_root() -> Path:
    """
    自动推断项目根目录:
    src/utils/paths.py -> 上两级是 src, 再上一级是项目根目录
    """
    return Path(__file__).resolve().parents[2]


def project_path(*parts) -> Path:
    return get_project_root().joinpath(*parts)