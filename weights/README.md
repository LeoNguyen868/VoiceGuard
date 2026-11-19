# Checkpoint

Checkpoint không nằm trong git: các file đều vài trăm MB, và encoder là trọng số của nhóm tác giả Nes2Net.

| File | Nội dung | Cách có |
|---|---|---|
| `WavLM_Nes2Net_X_ITW.pt` | Checkpoint In-the-Wild của nhóm tác giả (wav2vec 2.0 XLS-R 300M + Nes2Net-X, tên khóa fairseq) | Tải từ repo `Liu-Tianchi/Nes2Net_ASVspoof_ITW` (bảng checkpoint In-the-Wild) |
| `nes2netx_itw_vg.pt` | Mô hình gốc 24 lớp, định dạng VoiceGuard | `python -m voiceguard.convert authors --src WavLM_Nes2Net_X_ITW.pt --dst weights/nes2netx_itw_vg.pt` |
| `L14_head.pt` | Classifier Nes2Net-X của bản 14 lớp và dấu vân tay của encoder đã dùng khi huấn luyện | Phát hành kèm repo (xem bên dưới) |
| `voiceguard_L14.pt` | Bản 14 lớp dùng để triển khai | `python -m voiceguard.convert assemble --base weights/nes2netx_itw_vg.pt --head weights/L14_head.pt --dst weights/voiceguard_L14.pt` |

Encoder của bản 14 lớp được đóng băng khi huấn luyện, nên trùng khớp với 14 lớp đầu của mô hình gốc. Vì vậy chỉ cần phát hành classifier. Bước `assemble` kiểm tra dấu vân tay SHA-256 của 14 lớp encoder. Nếu mô hình gốc khác bản đã dùng khi huấn luyện, bước này dừng và báo lỗi.

Xuất classifier từ checkpoint huấn luyện:

```bash
python -m voiceguard.convert export-head --ckpt ckpt/control_L14_head.pt --dst weights/L14_head.pt
```

| File | SHA-256 |
|---|---|
| `L14_head.pt` | `01983bbf6d56e5bd37f935c56fbbadf84f0f9d2b9716a999e60bbfcbe56dbc88` |
| `voiceguard_L14.pt` | `a0457aa6fa538be78b5e8a808e1979a787e9a85142c2665a48e7c7001172838c` |
