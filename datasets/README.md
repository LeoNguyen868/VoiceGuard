# Dữ liệu kiểm thử

Repo không chứa âm thanh. Tải hai bộ dữ liệu dưới đây theo điều khoản của bên phát hành, đặt ở một thư mục bất kỳ, rồi trỏ `ITW`, `ASV5`, `ASV5_PROTOCOL` tới đó (xem `reproduce.sh`).

## In-the-Wild

- **Nội dung:** 31,779 đoạn tiếng Anh của người nổi tiếng thu từ mạng (19,963 thật, 11,816 giả).
- **Nguồn:** Müller và cộng sự, *Does Audio Deepfake Detection Generalize?*, Interspeech 2022, trang phát hành của nhóm tác giả.
- **Dữ liệu dùng:**
  - các file `.wav`;
  - `meta.csv` (`file,speaker,label`), `voiceguard` đọc trực tiếp file này làm nhãn.
- **Giao thức:** chấm toàn bộ 31,779 đoạn:
  - window 4 s đầu (lặp lại nếu ngắn hơn), là quyết định sau 4 s của dịch vụ;
  - toàn bộ độ dài, là giao thức của Nes2Net.

```bash
python -m voiceguard.datasets check --data_dir $ITW --labels $ITW/meta.csv
```

## ASVspoof 5, phần dev

- **Nội dung:** 140,950 đoạn, tấn công A09–A16. Giọng thật lấy từ Multilingual LibriSpeech tiếng Anh.
- **Nguồn:** bản phát hành chính thức trên Zenodo, hoặc bản sao Hugging Face `jungjee/asvspoof5`. Chỉ cần:
  - các file `flac_D_*`;
  - protocol `ASVspoof5.dev.track_1.tsv`.
- **Kiểm tra:** đối chiếu md5 của file tải về với README trên Zenodo.
- **Giấy phép:** xem `LICENSE.txt` của bản phát hành.
- **Giao thức:**
  - tập con phân tầng 20,000 đoạn (4,446 thật, 15,554 giả), `--subset 20000 --seed 2026`, cố định cho mọi mô hình;
  - window 4 s và toàn bộ độ dài.
- **Kiểm tra dấu hiệu tắt:** trên tập con này, AUC của độ dài, mức năng lượng và khoảng lặng đầu, cuối đều ≤ 0.57, nên không có dấu hiệu tắt đáng kể.

```bash
python -m voiceguard.datasets asv5 --protocol $ASV5_PROTOCOL --out datasets/asv5_dev_labels.txt
python -m voiceguard.datasets check --data_dir $ASV5 --labels datasets/asv5_dev_labels.txt
```

Nhãn lấy từ cột KEY (cột 9) của protocol; tên đoạn lấy từ cột 2.
