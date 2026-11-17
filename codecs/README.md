# Kênh truyền mô phỏng

`voiceguard/channel.py` mô phỏng kênh truyền bằng ffmpeg. Mỗi kênh mã hóa đoạn âm thanh 16 kHz bằng codec thật, giải mã lại, rồi trả về 16 kHz float32. Mô hình nhận đúng tín hiệu nó sẽ nhận sau kênh truyền. Codec áp lên toàn bộ đoạn trước khi cắt window.

| Tên | Codec | Bối cảnh | Encoder ffmpeg |
|---|---|---|---|
| `none` | Âm thanh gốc 16 kHz | Đối chứng | |
| `nb8k` | Hạ xuống 8 kHz, không codec | Tách ảnh hưởng của mất dải cao khỏi méo do codec | `pcm_s16le` |
| `g711u` | G.711 μ-law 64 kbit/s, 8 kHz | PSTN, tổng đài | `pcm_mulaw` |
| `g711a` | G.711 A-law 64 kbit/s, 8 kHz | PSTN châu Âu, châu Á | `pcm_alaw` |
| `gsm` | GSM 06.10 13 kbit/s, 8 kHz | Di động 2G | `libgsm` |
| `amrnb` | AMR-NB 12.2 kbit/s, 8 kHz | Di động 3G, 4G | `libopencore_amrnb` |
| `amrwb` | AMR-WB 12.65 kbit/s, 16 kHz | Thoại HD, VoLTE | `libvo_amrwbenc` |
| `g722` | G.722 64 kbit/s, 16 kHz | HD voice | `g722` |
| `opus12` | Opus 12 kbit/s, chế độ VoIP | Ứng dụng OTT | `libopus` |
| `mp3_32` | MP3 32 kbit/s | Tin nhắn thoại, mạng xã hội | `libmp3lame` |

Kiểm tra bản ffmpeg đang có:

```bash
python -m voiceguard.channel
ffmpeg -hide_banner -encoders | grep -E "amrnb|amrwb|libopus|libgsm|mp3lame|g722|mulaw|alaw"
```

- Các bản ffmpeg phổ biến thường thiếu hai encoder AMR, vì `libopencore_amrnb` và `libvo_amrwbenc` mang giấy phép riêng.
- Muốn có hai kênh AMR, dùng bản ffmpeg build với `--enable-libopencore-amrnb --enable-libvo-amrwbenc --enable-version3`, ví dụ bản conda-forge hoặc các bản build "full" cho Windows.
- Không có hai encoder này thì mọi kênh khác vẫn chạy. Chỉ các lần chấm có `--channel amrnb|amrwb` báo lỗi.
