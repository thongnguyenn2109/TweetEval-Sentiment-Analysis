# TweetEval Sentiment Lab — Streamlit

Ứng dụng trình bày nội dung toàn bộ đồ án NLP Nhóm 10 và hỗ trợ dự đoán cảm xúc tweet bằng mô hình Hugging Face thực tế.

## Cài đặt

Yêu cầu Python 3.10–3.12 (khuyến nghị), Internet khi tải checkpoint lần đầu.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Các trang

- Tổng quan: mục tiêu, dữ liệu, kết quả chính
- Dự đoán cảm xúc: nhập tweet, xác suất 3 nhãn, CSV hàng loạt
- Khám phá dữ liệu: phân bố lớp, các đặc trưng tweet
- So sánh mô hình: Accuracy, Macro-F1, tốc độ, seed
- Chi tiết thực nghiệm: các bảng theo mô hình/seed
- Toàn bộ báo cáo: trình đọc 7 chương, tài liệu tham khảo và phụ lục
- Giới thiệu & triển khai

## Checkpoint của nhóm

Vì file Word không đi kèm model weights, ứng dụng **không giả lập inference từ các checkpoint BERT/RoBERTa/DistilBERT đã fine-tune**. Để chạy đúng mô hình nhóm: trong trang Dự đoán, chọn "Checkpoint fine-tune" rồi nhập đường dẫn thư mục model Hugging Face đã được `save_pretrained` (gồm `config.json`, tokenizer và weights). `id2label` cần đặt nhãn `Negative`, `Neutral`, `Positive` hoặc `LABEL_0`, `LABEL_1`, `LABEL_2` tương ứng.

Phiên bản này hỗ trợ chọn và so sánh cả 5 mô hình. Đặt các artifact đã huấn luyện như sau:

```text
checkpoints/
  baseline/pipeline.joblib   # sklearn Pipeline TF-IDF + LogisticRegression đã fit
  bert/config.json           # và model.safetensors, tokenizer...
  roberta/config.json        # và model.safetensors, tokenizer...
  distilbert/config.json     # và model.safetensors, tokenizer...
```

`pipeline.joblib` cần được xuất từ đối tượng sklearn Pipeline đã train bằng `joblib.dump(pipeline, "pipeline.joblib")`. Chỉ mở file joblib đáng tin cậy do chính nhóm tạo. Các checkpoint Hugging Face cần `model.save_pretrained` và `tokenizer.save_pretrained` vào cùng thư mục. Không dùng checkpoint gốc chưa fine-tune thay cho trọng số nhóm. Zero-shot Hartmann được tải từ Hugging Face khi chạy, không cần file checkpoint nhóm.

Nếu sau khi nâng cấp torch gặp lỗi import `pipeline`, kiểm tra `python -c "from transformers import pipeline"`. TorchVision/TorchAudio cũ có thể không tương thích với torch mới; đồ án văn bản không cần hai thư viện này.

Số liệu và bảng biểu đã được giữ nguyên từ báo cáo, kể cả các trường chưa có kết quả (`[…]`). Một số hình vẽ gốc không được trích thành hình độc lập; file Word vẫn có thể tải từ giao diện.
