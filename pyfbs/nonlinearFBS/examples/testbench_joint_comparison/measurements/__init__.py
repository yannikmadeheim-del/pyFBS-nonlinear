"""Measurement tables for the joint-comparison example.

A table is addressed by its file name without the extension, e.g.
``"collocated_all_coupling_example"``. There is no registry, so any .xlsx
dropped into this folder is selectable straight away.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent

DEFAULT = "collocated_sensor_coupling_example"


def available():
    """Names of every workbook in this folder, sorted.

    Excel keeps a ``~$name.xlsx`` lock file beside an open workbook; those are
    not measurement tables and would otherwise appear as selectable names.
    """
    return sorted(p.stem for p in HERE.glob("*.xlsx")
                  if not p.name.startswith("~$"))


def resolve(name=DEFAULT):
    """Path of the workbook called ``name`` (with or without the extension)."""
    stem = Path(name).stem
    path = HERE / f"{stem}.xlsx"
    if not path.is_file():
        raise FileNotFoundError(
            f"no measurement table {stem!r} in {HERE}; "
            f"available: {', '.join(available())}")
    return path
