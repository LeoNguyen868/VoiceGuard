# Số liệu tham chiếu

EER (%), khoảng tin cậy 95% bootstrap 1000 lần trong ngoặc; thấp hơn là tốt hơn. Một lần huấn luyện (một seed). Chấm trên GPU (RTX 5060). Độ chính xác số:
- FP32: mô hình gốc trên In-the-Wild ở window 4 s, và cả hai mô hình trên In-the-Wild ở kênh gốc với toàn bộ độ dài;
- FP16: các ô còn lại.

Chênh lệch FP16 so với FP32 đã đo trên In-the-Wild ở kênh gốc, window 4 s, là dưới 0.01 điểm. Khi tái lập bằng `reproduce.sh` (FP16), sai khác nhỏ do làm tròn là bình thường.

## In-the-Wild (31,779 đoạn)

**Window 4 s**

| Kênh | Mô hình gốc (24 lớp) | Bản 14 lớp | Chênh lệch |
|---|---|---|---|
| Gốc | 7.62 (7.35–7.90) | 6.89 (6.60–7.14) | -0.73 |
| G.711 μ-law | 9.44 (9.10–9.77) | 8.05 (7.77–8.34) | -1.39 |
| Băng hẹp 8 kHz | 9.41 (9.06–9.71) | 8.03 (7.74–8.30) | -1.38 |
| GSM 06.10 | 15.47 (15.11–15.88) | 10.98 (10.60–11.35) | -4.50 |
| AMR-NB | 13.29 (12.96–13.70) | 11.14 (10.77–11.45) | -2.15 |
| AMR-WB | 11.35 (11.03–11.68) | 8.83 (8.55–9.16) | -2.52 |
| G.722 | 7.78 (7.51–8.07) | 7.53 (7.24–7.79) | -0.25 |
| Opus 12 kbit/s | 14.56 (14.18–14.95) | 10.69 (10.32–11.04) | -3.87 |
| MP3 32 kbit/s | 7.50 (7.22–7.79) | 8.28 (7.95–8.56) | +0.78 |

**Toàn bộ độ dài**

| Kênh | Mô hình gốc (24 lớp) | Bản 14 lớp | Chênh lệch |
|---|---|---|---|
| Gốc | 5.42 (5.15–5.69) | 5.36 (5.09–5.64) | -0.06 |
| GSM 06.10 | 13.69 (13.33–14.08) | 10.76 (10.39–11.13) | -2.93 |
| AMR-NB | 11.82 (11.48–12.18) | 10.26 (9.93–10.61) | -1.55 |
| Opus 12 kbit/s | 13.30 (12.95–13.71) | 10.81 (10.44–11.16) | -2.50 |
| MP3 32 kbit/s | 5.98 (5.71–6.27) | 6.42 (6.09–6.67) | +0.45 |

## ASVspoof 5 dev (tập con 20,000 đoạn, seed 2026)

**Window 4 s**

| Kênh | Mô hình gốc (24 lớp) | Bản 14 lớp | Chênh lệch |
|---|---|---|---|
| Gốc | 1.33 (1.13–1.51) | 0.81 (0.67–1.01) | -0.52 |
| AMR-NB | 5.37 (4.97–5.78) | 2.09 (1.91–2.32) | -3.28 |
| Opus 12 kbit/s | 7.96 (7.44–8.44) | 1.71 (1.49–1.91) | -6.25 |

**Toàn bộ độ dài**

| Kênh | Mô hình gốc (24 lớp) | Bản 14 lớp | Chênh lệch |
|---|---|---|---|
| Gốc | 0.83 (0.67–1.03) | 0.36 (0.29–0.51) | -0.47 |
| AMR-NB | 3.53 (3.22–3.87) | 1.42 (1.24–1.64) | -2.11 |
| Opus 12 kbit/s | 5.29 (4.84–5.72) | 1.08 (0.94–1.30) | -4.21 |

Chênh lệch tính bằng điểm phần trăm (âm là bản 14 lớp tốt hơn). Khoảng tin cậy của chênh lệch dùng paired bootstrap: `python -m voiceguard.compare`.
