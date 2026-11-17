# VoiceGuard

Dịch vụ giám định giọng nói nhân bản (deepfake audio) cho kênh thoại doanh nghiệp. Mã nguồn đi kèm đồ án tốt nghiệp *Hệ thống giám định deepfake audio (voice cloning) chống lừa đảo trực tuyến sử dụng CNN-Transformer* (Đại học Phenikaa, 2026).

Mô hình triển khai là **bản 14 lớp**:
- **Encoder:** 14 lớp Transformer đầu của wav2vec 2.0 XLS-R 300M, lấy từ checkpoint Nes2Net-X In-the-Wild của nhóm tác giả, đóng băng.
- **Classifier:** Nes2Net-X, huấn luyện lại trên ASVspoof 2019 LA.

So với mô hình gốc 24 lớp, bản này bền hơn khi âm thanh đi qua các codec thoại tốc độ bit thấp (Opus, AMR-NB, GSM), và suy luận rẻ hơn vì chỉ chạy 14/24 lớp. Số liệu tham chiếu ở `results/reference.md`.

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `voiceguard/` | Mã nguồn: mô hình, kênh truyền, cửa sổ trượt, chính sách rủi ro, lớp lọc số gọi đến, API, giao diện web, công cụ tái lập |
| `voiceguard/static/` | Giao diện web |
| `weights/` | Nơi đặt checkpoint; cách dựng bản 14 lớp |
| `configs/policy/` | Ngưỡng rủi ro của bản 14 lớp theo từng kênh (FRR 1% và 7% trên In-the-Wild, window 4 s) |
| `datasets/` | Cách lấy và chuẩn bị In-the-Wild, ASVspoof 5 |
| `codecs/` | Các kênh truyền mô phỏng bằng ffmpeg và yêu cầu bản build |
| `results/` | Số liệu tham chiếu để đối chiếu khi tái lập |
| `reproduce.sh` | Toàn bộ quy trình tái lập |

## Cài đặt

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # hoặc .../whl/cpu
pip install -r requirements.txt
python -m voiceguard.channel        # kiểm tra ffmpeg chạy được những codec nào
```

Cần ffmpeg trong `PATH`. Hai kênh AMR cần bản build có `libopencore_amrnb` và `libvo_amrwbenc` (xem `codecs/README.md`).

## Dựng checkpoint

Xem `weights/README.md`. Tóm tắt:

```bash
python -m voiceguard.convert authors  --src WavLM_Nes2Net_X_ITW.pt --dst weights/nes2netx_itw_vg.pt
python -m voiceguard.convert assemble --base weights/nes2netx_itw_vg.pt --head weights/L14_head.pt \
    --dst weights/voiceguard_L14.pt
```

## Chạy dịch vụ

```bash
uvicorn voiceguard.api:app --port 8000      # mở http://localhost:8000
```

- Mặc định dùng `weights/voiceguard_L14.pt` với ngưỡng `configs/policy/L14_clean_4s.json`.
- Biến môi trường:
  - `VG_CKPT`, `VG_POLICY`: checkpoint và ngưỡng khác (ví dụ ngưỡng theo kênh `configs/policy/L14_opus12_4s.json`);
  - `VG_DEVICE`, `VG_FP16=1`, `VG_CUDA_GRAPH=1`: thiết bị và tăng tốc;
  - `VG_REGISTERED`, `VG_BLACKLIST`: danh sách số cho lớp lọc;
  - `VG_RANDOM=tiny`: trọng số ngẫu nhiên, chỉ để thử giao diện.
- API:
  - `GET /health`;
  - `POST /v1/analyze` (multipart: `file`, `mode` = `window` | `full`, `channel`, `caller_number`);
  - cuộc gọi streaming: `POST /v1/calls`, `POST /v1/calls/{id}/audio` (PCM 16-bit 16 kHz), `POST /v1/calls/{id}/end`.
- Score là logit lớp thật: cao hơn là giống thật hơn. Audit log không lưu âm thanh và số điện thoại.

## Tái lập kết quả

```bash
export ITW=/data/in_the_wild            # thư mục có meta.csv và các file wav
export ASV5=/data/asvspoof5/flac_D      # các file FLAC phần dev
export ASV5_PROTOCOL=/data/asvspoof5/ASVspoof5.dev.track_1.tsv
bash reproduce.sh
```

`reproduce.sh` chấm mô hình gốc và bản 14 lớp trong cùng một lượt chạy encoder:
- In-the-Wild: toàn bộ 31,779 đoạn, window 4 s và toàn bộ độ dài, qua các kênh;
- ASVspoof 5 dev: tập con phân tầng 20,000 đoạn, seed 2026.

Sau đó, với mỗi cặp, `voiceguard.compare` tính chênh lệch EER với khoảng tin cậy 95% bằng paired bootstrap.

Từng bước riêng lẻ:

```bash
python -m voiceguard.evaluate --ckpt weights/nes2netx_itw_vg.pt --ckpt_shared weights/voiceguard_L14.pt \
    --data_dir $ITW --labels $ITW/meta.csv --channel opus12 --durations 4 full --fp16 \
    --prefix itw_base_ --prefix_shared itw_L14_
python -m voiceguard.compare --labels $ITW/meta.csv \
    --ref results/scores/itw_base_opus12_4s.txt --model results/scores/itw_L14_opus12_4s.txt
```

## Nguồn gốc

- Kiến trúc và checkpoint gốc: T. Liu, D.-T. Truong, R. K. Das, K. A. Lee, H. Li, *Nes2Net: A Lightweight Nested Architecture for Foundation Model Driven Speech Anti-Spoofing*, IEEE TIFS, 2025. `voiceguard/backend.py` chép từ mã của nhóm tác giả, giữ nguyên tên module để nạp checkpoint với `strict=True`.
- Encoder chạy trên Hugging Face `transformers` thay cho fairseq. Trọng số được đổi tên khi chuyển đổi, logit khớp với bản fairseq.
