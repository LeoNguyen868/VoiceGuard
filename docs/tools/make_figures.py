"""Write img/<name>.boxes.tex (one \\annbox per numbered part) from the img/boxes.json of capture.py.

  python docs/tools/make_figures.py docs/user-guide/img
"""
import json
import sys
from pathlib import Path

folder = Path(sys.argv[1])
for name, rows in json.loads((folder / "boxes.json").read_text(encoding="utf-8")).items():
    lines = [f"\\annbox{{{i}}}{{{x}}}{{{y}}}{{{w}}}{{{h}}}" for i, x, y, w, h in rows]
    (folder / f"{name}.boxes.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(name, len(rows))
