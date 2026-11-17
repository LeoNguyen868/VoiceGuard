"""Rule-based pre-filter on the caller number, applied before the examination service.

The detector is not trusted alone. Numbers on the blacklist are stopped without running the
model; calls from numbers that are unknown or hidden get at least MEDIUM risk (extra verification);
registered customer numbers are still screened, because caller ID can be spoofed.
"""
import re

RULES = {
    "blacklist": "Số nằm trong blacklist: tạm dừng giao dịch, chuyển bộ phận chống gian lận",
    "registered": "Số đã đăng ký của khách hàng: áp dụng chính sách rủi ro chuẩn",
    "unknown": "Số chưa có trong danh sách khách hàng: mức rủi ro tối thiểu là MEDIUM",
    "hidden": "Số bị ẩn: mức rủi ro tối thiểu là MEDIUM",
    "not_provided": "Không có thông tin số gọi đến: áp dụng chính sách rủi ro chuẩn",
}


def normalize(number):
    """Digits only, Vietnamese +84 prefix folded to 0: '+84 912-345-678' -> '0912345678'."""
    digits = re.sub(r"\D", "", number or "")
    if digits.startswith("84") and len(digits) == 11:
        digits = "0" + digits[2:]
    return digits


def load_numbers(path):
    if not path:
        return set()
    with open(path, encoding="utf-8") as f:
        return {normalize(line) for line in f if normalize(line)}


def classify(number, registered, blacklist):
    """Category of a caller number given normalised registered and blacklist sets."""
    if number is None or number.strip() == "":
        return "not_provided"
    n = normalize(number)
    if not n:
        return "hidden"
    if n in blacklist:
        return "blacklist"
    if n in registered:
        return "registered"
    return "unknown"
