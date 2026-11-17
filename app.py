"""Interactive presentation and inference dashboard for the supplied TweetEval report."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import streamlit as st
from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "report.docx"
LABELS = ["Negative", "Neutral", "Positive"]
MODEL_PATHS = {
    "Baseline · TF-IDF + Logistic Regression": ROOT / "checkpoints" / "baseline" / "pipeline.joblib",
    "Zero-shot · Hartmann RoBERTa (pretrained)": "j-hartmann/sentiment-roberta-large-english-3-classes",
    "BERT-base · fine-tuned": ROOT / "checkpoints" / "bert",
    "RoBERTa-base · fine-tuned": ROOT / "checkpoints" / "roberta",
    "DistilBERT · fine-tuned": ROOT / "checkpoints" / "distilbert",
}
PAGES = [
    "Tổng quan",
    "Dự đoán cảm xúc",
    "Khám phá dữ liệu",
    "So sánh mô hình",
    "Chi tiết thực nghiệm",
    "Toàn bộ báo cáo",
    "Giới thiệu & triển khai",
]
PAGE_ICONS = {
    "Tổng quan": ":material/home:",
    "Dự đoán cảm xúc": ":material/network_intelligence:",
    "Khám phá dữ liệu": ":material/query_stats:",
    "So sánh mô hình": ":material/leaderboard:",
    "Chi tiết thực nghiệm": ":material/science:",
    "Toàn bộ báo cáo": ":material/menu_book:",
    "Giới thiệu & triển khai": ":material/info:",
}

st.set_page_config(page_title="TweetEval · Sentiment Lab", page_icon=":material/forum:", layout="wide")


@st.cache_data
def read_report():
    """Preserve original paragraphs/tables order rather than dropping appendix material."""
    doc = Document(REPORT)
    blocks = []
    table_no = 0
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            p = Paragraph(child, doc)
            txt = p.text.strip()
            if txt:
                blocks.append({"kind": "text", "style": p.style.name, "text": txt})
        elif child.tag == qn("w:tbl"):
            t = Table(child, doc)
            rows = [[c.text.strip() for c in r.cells] for r in t.rows]
            if rows:
                table_no += 1
                blocks.append({"kind": "table", "number": table_no, "rows": rows})
    return blocks


BLOCKS = read_report()
CHAPTERS = [b["text"] for b in BLOCKS if b["kind"] == "text" and b["style"] == "Heading 1"]


def report_table(number):
    tables = [b for b in BLOCKS if b["kind"] == "table"]
    rows = tables[number]["rows"]  # zero-indexed, matching python-docx order
    if len(rows) < 2:
        return pd.DataFrame()
    headers = rows[0]
    if len(set(headers)) != len(headers):
        headers = [f"{h} ({i})" for i, h in enumerate(headers)]
    return pd.DataFrame(rows[1:], columns=headers)


def number(value):
    s = str(value).split("±")[0].strip().replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def pct(value):
    """Format a 0-1 fraction as a Vietnamese-style percentage, e.g. 0.9314 -> '93,14%'."""
    if value is None:
        return "—"
    return f"{value * 100:.2f}".replace(".", ",") + "%"


SENTIMENT_BADGE_COLOR = {"Negative": "red", "Neutral": "orange", "Positive": "green"}


def clean_tweet(text):
    text = re.sub(r"https?://\S+|www\.\S+", "http", text)
    text = re.sub(r"@\w+", "@user", text)
    return re.sub(r"\s+", " ", text).strip()


@st.cache_resource(show_spinner="Đang tải mô hình lần đầu…")
def get_pipeline(model_path):
    from transformers import pipeline
    return pipeline("text-classification", model=str(model_path), tokenizer=str(model_path),
                    top_k=None, truncation=True, device=-1)


def infer(text, model_path):
    if isinstance(model_path, Path) and model_path.suffix == ".joblib":
        from joblib import load
        if not model_path.is_file():
            raise FileNotFoundError(f"Chưa có mô hình baseline: {model_path}. Cần xuất pipeline đã train bằng joblib.dump(...).")
        clf = get_baseline(str(model_path))
        probabilities = clf.predict_proba([text])[0]
        result = dict(zip(clf.classes_, probabilities))
        return {label: float(result.get(i, result.get(label, 0))) for i, label in enumerate(LABELS)}
    if isinstance(model_path, Path) and not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Chưa tìm thấy checkpoint tại {model_path}. Hãy đặt model.safetensors, config.json và tokenizer vào thư mục này.")
    outputs = get_pipeline(model_path)(text)
    if outputs and isinstance(outputs[0], list):
        outputs = outputs[0]
    raw = {r["label"].lower(): float(r["score"]) for r in outputs}
    scores = {}
    for i, label in enumerate(LABELS):
        scores[label] = raw.get(label.lower(), raw.get(f"label_{i}", 0.0))
    if not any(scores.values()):
        raise ValueError("Checkpoint không có nhãn tương ứng Negative/Neutral/Positive. Kiểm tra id2label trong config.json.")
    return scores


@st.cache_resource(show_spinner="Đang tải Baseline…")
def get_baseline(path):
    from joblib import load
    return load(path)  # Chỉ nạp file joblib do chính nhóm tạo và tin cậy.


def predict_models(tweet, selected, normalize):
    rows = []
    for name in selected:
        try:
            # Baseline dùng text_clean, Transformer dùng text_fixed trong notebook.
            # Chế độ demo sử dụng chuẩn hóa tối thiểu khi được chọn.
            text = clean_tweet(tweet) if normalize else tweet.strip()
            scores = infer(text, MODEL_PATHS[name])
            winner = max(scores, key=scores.get)
            rows.append({"Mô hình": name, "Trạng thái": "OK", "Dự đoán": winner,
                         "Độ tin cậy": scores[winner], **scores})
        except Exception as exc:
            rows.append({"Mô hình": name, "Trạng thái": str(exc), "Dự đoán": "—",
                         "Độ tin cậy": None, **{label: None for label in LABELS}})
    return pd.DataFrame(rows)


with st.sidebar:
    with st.container(key="nav_panel"):
        st.title("Sentiment Lab", icon=":material/forum:")
        st.caption("TweetEval · NLP · Nhóm 10")
        page = st.radio("Điều hướng", PAGES, format_func=lambda p: f"{PAGE_ICONS[p]} {p}")
        st.caption("Dữ liệu và điểm số từ báo cáo gốc. Dự đoán mới cần tải mô hình thực.")
    st.html("<style>.st-key-nav_panel { font-size: 1.6rem; }</style>")
    # No theme.toml option controls sidebar width; this is the documented
    # Streamlit internal testid and may need updating on a future Streamlit upgrade.
    st.html("<style>[data-testid='stSidebar'] { min-width: 400px; }</style>")

if page == "Tổng quan":
    st.title("Phân tích cảm xúc tweet bằng NLP", icon=":material/forum:")
    st.markdown(":blue-badge[:material/sell: TweetEval · Three-class sentiment classification]")
    st.write("Hệ thống minh họa đề tài: tiền xử lý tweet, khám phá dữ liệu, đánh giá và so sánh TF-IDF + Logistic Regression, Zero-shot, BERT-base, RoBERTa-base và DistilBERT.")
    cols = st.columns(4)
    for col, (title, value) in zip(cols, [("Train", "45.615"), ("Validation", "2.000"), ("Test", "12.284"), ("Nhãn cảm xúc", "3")]):
        with col.container(border=True):
            st.metric(title, value)
    st.subheader("Quy trình hệ thống", icon=":material/account_tree:")
    st.info("Tweet → Chuẩn hóa văn bản → Tokenizer / TF-IDF → Mô hình → Negative / Neutral / Positive → Đánh giá")
    st.subheader("Kết quả tổng hợp trên tập test", icon=":material/table_chart:")
    df = report_table(51)
    st.dataframe(df, width="stretch", hide_index=True, alt="Accuracy và Macro-F1 trên tập test cho từng mô hình")
    st.caption("Giá trị ghi trong báo cáo; một số mô hình chưa chạy đủ 3 seed.")

elif page == "Dự đoán cảm xúc":
    st.title("Dự đoán cảm xúc tweet", icon=":material/network_intelligence:")
    st.write("Nhập tweet tiếng Anh để phân loại thành 3 nhóm. Mô hình tải từ Hugging Face ở lần chạy đầu (cần Internet), hoặc chọn checkpoint fine-tune của nhóm.")
    selection_mode = st.segmented_control("Chế độ", ["Một mô hình", "So sánh 5 mô hình"], default="Một mô hình")
    if selection_mode == "Một mô hình":
        model_choice = st.selectbox("Nguồn mô hình", list(MODEL_PATHS))
        selected_models = [model_choice]
    else:
        model_choice = None
        selected_models = st.multiselect("Các mô hình cần so sánh", list(MODEL_PATHS), default=list(MODEL_PATHS))
    st.markdown("**Trạng thái checkpoint**")
    st.markdown(" ".join(
        f":green-badge[:material/check_circle: {name.split(' · ')[0]}]"
        if (not isinstance(path, Path) or path.exists())
        else f":gray-badge[:material/pending: {name.split(' · ')[0]}]"
        for name, path in MODEL_PATHS.items()))
    sample = st.selectbox("Ví dụ nhanh", ["Tự nhập", "I absolutely love this update!", "The event starts at 8 pm tomorrow.", "This is the worst customer service ever."])
    tweet = st.text_area("Nội dung tweet", value="" if sample == "Tự nhập" else sample, height=125, placeholder="Type an English tweet here...")
    normalize = st.checkbox("Chuẩn hóa URL và mention", value=True)
    if st.button("Phân tích cảm xúc", type="primary", icon=":material/send:", width="stretch"):
        if not tweet.strip():
            st.warning("Vui lòng nhập nội dung tweet.")
        else:
            processed = clean_tweet(tweet) if normalize else tweet.strip()
            st.caption(f"Văn bản đưa vào mô hình: {processed}")
            with st.spinner("Đang suy luận…"):
                results = predict_models(tweet, selected_models, normalize)
            valid = results[results["Trạng thái"] == "OK"]
            failed = results[results["Trạng thái"] != "OK"]
            if not valid.empty:
                chart_col, card_col = st.columns([3, 2])
                with chart_col:
                    st.bar_chart(valid.set_index("Mô hình")[LABELS], horizontal=True,
                                 height=90 + 70 * len(valid), alt="Xác suất Negative/Neutral/Positive theo mô hình")
                with card_col:
                    for _, row in valid.iterrows():
                        with st.container(border=True):
                            st.markdown(f"**{row['Mô hình']}**")
                            color = SENTIMENT_BADGE_COLOR[row["Dự đoán"]]
                            st.markdown(f":{color}-badge[:material/check_circle: {row['Dự đoán']}] · **{pct(row['Độ tin cậy'])}**")
                detail = valid.drop(columns=["Trạng thái"]).copy()
                for col in ["Độ tin cậy"] + LABELS:
                    detail[col] = detail[col].map(pct)
                st.dataframe(detail, hide_index=True, width="stretch", row_height=44)
            if not failed.empty:
                st.dataframe(failed[["Mô hình", "Trạng thái"]], hide_index=True, width="stretch", row_height=44)
            if valid.empty:
                st.warning("Chưa có mô hình nào dự đoán thành công. Xem cột Trạng thái để biết lý do.")
            st.caption("Điểm tin cậy là đầu ra của mô hình đang chọn, không phải độ chính xác của hệ thống.")
    st.subheader("Phân tích hàng loạt từ CSV", icon=":material/table_view:")
    upload = st.file_uploader("CSV có cột text", type="csv")
    if upload is not None:
        try:
            batch = pd.read_csv(upload)
            st.dataframe(batch.head(10), width="stretch")
            if "text" not in batch.columns:
                st.warning("CSV cần chứa cột tên là text.")
            elif selection_mode != "Một mô hình":
                st.info("Dự đoán CSV chỉ hỗ trợ chế độ Một mô hình để tránh tải đồng thời nhiều checkpoint lớn.")
            elif st.button("Dự đoán và xuất CSV", icon=":material/play_arrow:"):
                if len(batch) > 1000:
                    st.warning("Mỗi lượt hỗ trợ tối đa 1.000 tweet để tránh quá tải.")
                else:
                    try:
                        preds, confidences = [], []
                        progress = st.progress(0)
                        for i, value in enumerate(batch["text"].fillna("")):
                            if not str(value).strip():
                                preds.append(""); confidences.append(None)
                            else:
                                scores = infer(clean_tweet(str(value)) if normalize else str(value), MODEL_PATHS[model_choice])
                                lbl = max(scores, key=scores.get)
                                preds.append(lbl); confidences.append(scores[lbl])
                            progress.progress((i + 1) / len(batch))
                        out = batch.assign(predicted_label=preds, confidence=confidences)
                        st.dataframe(out, width="stretch")
                        st.download_button("Tải kết quả CSV", out.to_csv(index=False).encode("utf-8-sig"), "sentiment_predictions.csv", "text/csv", icon=":material/download:")
                    except Exception as exc:
                        st.error(f"Lỗi khi dự đoán hàng loạt: {exc}")
        except Exception as exc:
            st.error(f"CSV không hợp lệ: {exc}")

elif page == "Khám phá dữ liệu":
    st.title("Khám phá dữ liệu TweetEval", icon=":material/query_stats:")
    st.caption("Phân bố nhãn theo 3 tập, lấy trực tiếp từ bảng trong báo cáo.")
    split_tables = {"Train": 4, "Validation": 5, "Test": 6}
    split = st.selectbox("Chọn tập dữ liệu", list(split_tables))
    df = report_table(split_tables[split])
    vals = [7093, 20673, 17849] if split == "Train" else ([312, 869, 819] if split == "Validation" else [3972, 5937, 2375])
    left, right = st.columns([2, 1])
    with left:
        st.bar_chart(pd.DataFrame({"Số tweet": vals}, index=LABELS))
    with right:
        st.dataframe(df, hide_index=True, width="stretch")
    st.subheader("Độ dài tweet, hashtag, mention, emoji", icon=":material/straighten:")
    st.dataframe(report_table(7), width="stretch", hide_index=True)
    st.subheader("Khám phá tweet của bạn", icon=":material/travel_explore:")
    input_text = st.text_area("Thử tiền xử lý", "@Alex I love this product 😍 https://example.com #great")
    st.code(clean_tweet(input_text), language="text")
    st.caption("Đây là minh họa chuẩn hóa URL/mention/khoảng trắng; pipeline tiền xử lý thực tế phụ thuộc mô hình.")

elif page == "So sánh mô hình":
    st.title("So sánh hiệu năng mô hình", icon=":material/leaderboard:")
    df = report_table(51)
    metrics = pd.DataFrame({"Mô hình": df.iloc[:, 0], "Accuracy": df.iloc[:, 1].map(number), "Macro-F1": df.iloc[:, 2].map(number)})
    metric = st.segmented_control("Chỉ số", ["Macro-F1", "Accuracy"], default="Macro-F1")
    st.bar_chart(metrics.set_index("Mô hình")[[metric or "Macro-F1"]], horizontal=True,
                 height=70 + 50 * len(metrics))
    st.dataframe(df, width="stretch", hide_index=True)
    st.subheader("Transformer: tốc độ và chi phí", icon=":material/speed:")
    st.dataframe(report_table(50), width="stretch", hide_index=True)
    st.subheader("Độ ổn định qua các random seed", icon=":material/scatter_plot:")
    st.dataframe(report_table(53), width="stretch", hide_index=True)
    st.warning("BERT-base seed 2/3 và phần Multi-task Learning vẫn có ô […] trong báo cáo. Không suy diễn các kết quả chưa được thực nghiệm.")

elif page == "Chi tiết thực nghiệm":
    st.title("Kết quả thực nghiệm chi tiết", icon=":material/science:")
    st.write("Bảng đầy đủ kết quả từng seed, chỉ số theo lớp và cấu hình huấn luyện được trích trực tiếp từ đồ án.")
    model = st.selectbox("Nhóm thực nghiệm", ["Baseline", "Zero-shot", "BERT-base", "RoBERTa-base", "DistilBERT", "Multi-task", "Tất cả các seed"])
    mapping = {"Baseline": [8, 11, 12, 13], "Zero-shot": [14, 15, 16], "BERT-base": [17, 18], "RoBERTa-base": list(range(19, 39)), "DistilBERT": list(range(39, 50)), "Multi-task": [55, 56], "Tất cả các seed": [58]}
    for idx in mapping[model]:
        with st.expander(f"Bảng {idx + 1} · dữ liệu báo cáo", icon=":material/table_chart:", expanded=(idx == mapping[model][0])):
            st.dataframe(report_table(idx), hide_index=True, width="stretch")
    st.subheader("Cấu hình mô hình", icon=":material/tune:")
    with st.expander("Checkpoint và hyperparameters", icon=":material/settings:"):
        st.dataframe(report_table(9), hide_index=True, width="stretch")
        st.dataframe(report_table(10), hide_index=True, width="stretch")

elif page == "Toàn bộ báo cáo":
    st.title("Nội dung toàn bộ đồ án", icon=":material/menu_book:")
    st.write("Trình đọc nội dung theo chương, bao gồm cả bảng biểu và phụ lục từ tài liệu Word được cung cấp.")
    chapter = st.selectbox("Đi đến mục", CHAPTERS)
    search = st.text_input("Lọc nội dung trong mục theo từ khóa (tùy chọn)")
    active = False
    shown = 0
    for b in BLOCKS:
        if b["kind"] == "text" and b["style"] == "Heading 1":
            if active:
                break
            active = b["text"] == chapter
        if not active:
            continue
        if b["kind"] == "text":
            t = b["text"]
            if search and search.casefold() not in t.casefold():
                continue
            style = b["style"]
            level = int(style[-1]) if re.match(r"Heading [1-4]$", style) else None
            if level:
                st.markdown("#" * min(level + 1, 5) + " " + t)
            else:
                st.write(t)
            shown += 1
        elif not search or any(search.casefold() in c.casefold() for row in b["rows"] for c in row):
            st.caption(f"Bảng dữ liệu #{b['number']}")
            rows = b["rows"]
            st.dataframe(pd.DataFrame(rows[1:], columns=[f"{x} ({i})" for i, x in enumerate(rows[0])]), width="stretch", hide_index=True)
            shown += 1
    if not shown:
        st.info("Không tìm thấy nội dung phù hợp.")
    st.download_button("Tải báo cáo Word gốc", REPORT.read_bytes(), REPORT.name, "application/vnd.openxmlformats-officedocument.wordprocessingml.document", icon=":material/download:")

else:
    st.title("Giới thiệu & triển khai", icon=":material/info:")
    st.markdown("""**Đề tài:** Phân tích cảm xúc tweet sử dụng các phương pháp NLP và Transformer.
**Nguồn dữ liệu:** [TweetEval trên Hugging Face](https://huggingface.co/datasets/cardiffnlp/tweet_eval).
**Các nhóm mô hình:** Baseline TF-IDF + Logistic Regression, mô hình sentiment pretrained (Zero-shot trong cách gọi báo cáo), BERT-base, RoBERTa-base, DistilBERT.

**Lưu ý minh bạch:** Báo cáo Word không chứa file trọng số huấn luyện, vectorizer, classifier hay tập test đầy đủ. Vì vậy điểm số trong dashboard là kết quả *đã ghi trong báo cáo*, còn chức năng dự đoán phải tải checkpoint có thật. Mô hình Hartmann chưa fine-tune trên TweetEval và không phải zero-shot NLI theo nghĩa nghiêm ngặt.

**Chạy trên máy tính:**
```bash
pip install -r requirements.txt
streamlit run app.py
```
Khi chạy checkpoint nhóm, hãy đặt thư mục mô hình đã lưu vào `checkpoints/best_model` hoặc nhập đường dẫn của checkpoint. Checkpoint Hugging Face cần Internet và đủ RAM ở lượt tải đầu.
""")
    st.subheader("Các thành viên", icon=":material/group:")
    st.dataframe(report_table(59), hide_index=True, width="stretch")
