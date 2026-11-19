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
pip install torch --index-url https://download.pytorch.org/whl/cu130   # GPU RTX 50 (sm_120); hoặc .../whl/cpu
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

## Hệ thống cho tổng đài

| Công việc | Màn hình (`http://<máy chủ>:8000/`) |
|---|---|
| Theo dõi cuộc gọi | **Cuộc gọi trực tiếp:** mức rủi ro và khả năng giả của từng cuộc gọi, cập nhật mỗi 2 giây; cuộc gọi rủi ro cao lên đầu, có âm báo; nút xác minh, tạm giữ giao dịch, chuyển chống gian lận |
| Điều tra | **Hàng đợi điều tra:** cuộc gọi cần xác minh hoặc rủi ro cao đã kết thúc; kết luận gian lận (tự đưa số vào blacklist) hoặc khách thật<br>**Danh sách số:** blacklist, số đã đăng ký, số được đề xuất chặn (gọi tới nhiều lần ở mức rủi ro cao) |
| Giám sát | **Tổng quan:** số cuộc gọi, cảnh báo, chặn, gian lận đã xác nhận, tỷ lệ cảnh báo nhầm, độ trễ |
| Kiểm tra tệp | Tải lên tin nhắn thoại hoặc bản ghi để kiểm tra (cũng mở riêng ở `/check`) |

- **Luồng:** tổng đài mở phiên cho mỗi cuộc gọi (`POST /v1/calls` với số gọi đến và kênh của đường dây), gửi âm thanh PCM 16 kHz theo từng đoạn và đóng phiên khi cúp máy. Mọi bước được lưu vào SQLite (`VG_DB`) và đẩy tới các màn hình qua `GET /v1/events` (server-sent events).
- **Nghe lại:** âm thanh của cuộc gọi cần xác minh hoặc rủi ro cao được giữ dạng WAV (`data/recordings/`) để chuyên viên nghe lại; trong ngăn chi tiết, thanh thời gian tô màu theo từng đoạn 4 giây, bấm để nghe đúng đoạn đó. Cuộc gọi bình thường không được lưu âm thanh. `VG_RECORD=flagged|all|none`, `VG_RECORD_DAYS` (mặc định 30 ngày).
- **Dữ liệu cá nhân:** số điện thoại chỉ được lưu dưới dạng mã băm có salt và dạng đã che; giọng nói là dữ liệu sinh trắc học nên chỉ giữ ở cuộc gọi bị gắn cờ và xóa sau thời hạn.
- **Webhook:** `VG_WEBHOOK_URL` nhận JSON khi một cuộc gọi lên mức rủi ro cao hoặc khi nhân viên tạm giữ, chuyển cuộc gọi (để CRM hoặc core banking giữ giao dịch).
- **Đăng nhập:** bảng điều khiển mở ở `/login`. Lần chạy đầu tạo tài khoản `admin` với mật khẩu `VG_ADMIN_PASSWORD` (không đặt thì mật khẩu ngẫu nhiên được ghi vào `data/admin-password.txt`). Mọi tài khoản dùng được toàn bộ màn hình; thêm, khóa tài khoản và đổi mật khẩu bằng `python -m voiceguard.users add|list|passwd|disable|enable`. Phiên đăng nhập là cookie HttpOnly hết hạn sau `VG_SESSION_HOURS` giờ (mặc định 12); sai mật khẩu 5 lần trong 5 phút thì tài khoản bị tạm chặn.
- **Kết nối tổng đài:** PBX gọi các API cuộc gọi với `Authorization: Bearer <VG_API_KEY>`; khóa này chỉ mở được luồng cuộc gọi và `/v1/analyze`, không đọc được lịch sử.
- **Mô phỏng tổng đài:** `VG_API_KEY=<khóa> VG_SIM_PASSWORD=<mật khẩu> python -m voiceguard.simulate --url http://localhost:8000 --calls 20 --user admin` mở 20 cuộc gọi đồng thời, phát âm thanh của thư viện mẫu qua codec của từng đường dây với tốc độ thật; `--user` dùng để đăng ký sẵn số khách hàng. `VG_SIM_PASSWORD` là mật khẩu của chính tài khoản `--user`; không đặt thì dùng `VG_ADMIN_PASSWORD`. Cuối lượt chạy có hai bảng đếm: mức cuối cùng (theo điểm trung bình cả cuộc) và mức cao nhất trong cuộc gọi; bảng điều khiển và `/v1/stats` đếm theo mức cao nhất, nên hai bảng có thể chênh nhau.

## Chạy dịch vụ

```bash
uvicorn voiceguard.api:app --port 8000      # mở http://localhost:8000
```

- Mặc định dùng `weights/voiceguard_L14.pt`. Mỗi kênh dùng ngưỡng hiệu chỉnh riêng trong `configs/policy/L14_<kênh>_4s.json`; kênh chưa có tệp ngưỡng dùng ngưỡng của âm thanh gốc.
- Giao diện web gồm:
  - thư viện mẫu giọng thật và giả để nghe và phân tích, có nhãn để đối chiếu kết luận;
  - tải tệp và ghi âm từ micro;
  - chọn kênh mô phỏng kèm ví dụ ứng dụng (Zalo, Messenger: Opus; di động 3G, 4G: AMR-NB; VoLTE: AMR-WB; tổng đài: G.711);
  - so sánh cùng một âm thanh qua các kênh;
  - điểm từng cửa sổ 4 s so với hai ngưỡng.
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

## Thư viện mẫu cho giao diện

Thư viện mẫu được tạo từ bản sao các bộ dữ liệu trên máy. Mỗi bộ dữ liệu lấy ngẫu nhiên với seed cố định một số đoạn thật và giả, không chọn theo điểm, nên thư viện có cả những đoạn mô hình đoán sai:

```bash
python -m voiceguard.samples --out samples --n 4 \
    --itw $ITW --itw_labels $ITW/meta.csv \
    --asv5 $ASV5 --asv5_protocol $ASV5_PROTOCOL \
    --sea $SEA/flac --sea_systems $SEA/systems.txt \
    --own my_clips --own_labels my_clips/labels.txt   # tùy chọn: mẫu tự thu, mỗi dòng "tên bonafide|spoof [hệ thống]"
```

`--itw_labels` nhận `meta.csv` của In-the-Wild (`file,speaker,label`) hoặc tệp nhãn dạng `tên nhãn` mỗi dòng; với dạng sau, thư viện mẫu không có tên người nói. `--asv5_protocol` là tệp `ASVspoof5.dev.track_1.tsv`.

`samples/` không được đưa vào git: SEA-Spoof chỉ dùng cho học thuật và không được phân phối lại. Đặt `VG_SAMPLES` nếu thư viện nằm ở chỗ khác.

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
