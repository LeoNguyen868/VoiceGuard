"""Manage console accounts.

  python -m voiceguard.users add nv01 --name "Nguyễn Văn A"     # asks for the password
  python -m voiceguard.users list
  python -m voiceguard.users passwd nv01
  python -m voiceguard.users disable nv01

Every account opens the whole console. The database is VG_DB (default data/voiceguard.db).
"""
import argparse
import getpass
import os
from datetime import datetime
from pathlib import Path

from .store import Store

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("username")
    a.add_argument("--name", default=None)
    sub.add_parser("list")
    p = sub.add_parser("passwd")
    p.add_argument("username")
    d = sub.add_parser("disable")
    d.add_argument("username")
    e = sub.add_parser("enable")
    e.add_argument("username")
    args = ap.parse_args()
    st = Store(os.environ.get("VG_DB", str(ROOT / "data" / "voiceguard.db")))
    if args.cmd in ("add", "passwd"):
        pw = getpass.getpass("Mật khẩu: ")
        if len(pw) < 8 or pw != getpass.getpass("Nhập lại: "):
            raise SystemExit("mật khẩu phải từ 8 ký tự và hai lần nhập phải giống nhau")
        st.set_user(args.username, pw, getattr(args, "name", None))
        print("đã lưu")
    elif args.cmd == "list":
        for u in st.users():
            last = datetime.fromtimestamp(u["last_login"]).strftime("%Y-%m-%d %H:%M") if u["last_login"] else "-"
            print(f"{u['username']:16s} {'khóa' if u['disabled'] else 'hoạt động':10s} {last}  {u['full_name']}")
    else:
        st.set_user(args.username, disabled=int(args.cmd == "disable"))
        print("đã khóa" if args.cmd == "disable" else "đã mở khóa")


if __name__ == "__main__":
    main()
