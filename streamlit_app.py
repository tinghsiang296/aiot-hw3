import os
import json
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import subprocess
try:
    import seaborn as sns
except Exception:
    sns = None
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
)


MODEL_DIR = "models"


@st.cache_data
def load_dataset(path):
    df = pd.read_csv(path, header=None)
    return df


@st.cache_data
def load_model(model_dir=MODEL_DIR):
    model_path = os.path.join(model_dir, "model.pkl")
    le_path = os.path.join(model_dir, "label_encoder.pkl")
    test_split = os.path.join(model_dir, "test_split.joblib")
    model = None
    le = None
    test = None
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    if os.path.exists(le_path):
        le = joblib.load(le_path)
    if os.path.exists(test_split):
        test = joblib.load(test_split)
    return model, le, test


def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots()
    if sns is not None:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    else:
        im = ax.imshow(cm, cmap="Blues")
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, int(val), ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    return fig


def plot_roc_pr(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr, label=f"ROC AUC={roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].legend()

    axes[1].plot(recall, precision, label=f"PR AUC={pr_auc:.3f}")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()

    fig.tight_layout()
    return fig


def main():
    st.title("垃圾郵件分類器 — Streamlit Demo")

    st.sidebar.header("資料與模型設定")
    data_path = st.sidebar.text_input("Dataset CSV path", value="data/sms_spam_no_header.csv")
    model_dir = st.sidebar.text_input("Models directory", value=MODEL_DIR)
    text_col = st.sidebar.number_input("Text column index (0-based)", value=1, min_value=0)
    label_col = st.sidebar.number_input("Label column index (0-based)", value=0, min_value=0)

    st.sidebar.markdown("---")
    st.sidebar.header("預測閾值")
    threshold = st.sidebar.slider("Threshold", 0.0, 1.0, 0.5, 0.01)

    st.sidebar.markdown("---")
    st.sidebar.header("快速測試範例")
    # 多樣化範例集
    spam_examples = [
        "Congratulations! You've won a free ticket. Call now to claim.",
        "You have been selected for a $1000 gift card. Click the link to redeem.",
        "URGENT: Your account will be suspended. Verify immediately: http://phishy.example",
        "Free trial available! No credit card required. Sign up here.",
        "Lowest price on meds - order now and save 70!",
    ]
    ham_examples = [
        "Hey, are we still meeting for lunch tomorrow?",
        "Can you send the report by Friday? Thanks!",
        "Happy birthday! Let's celebrate this weekend.",
        "Here's the meeting notes from today. Please review.",
        "Are you available for a quick call later today?",
    ]

    # keep examples in sidebar info only
    st.sidebar.write(f"Spam 範例數: {len(spam_examples)} | Ham 範例數: {len(ham_examples)}")

    model, le, test = load_model(model_dir)

    # Show model summary / metrics if available
    metrics_json = None
    metrics_path = os.path.join(model_dir, "metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as fh:
                metrics_json = json.load(fh)
        except Exception:
            metrics_json = None

    if metrics_json is not None:
        st.subheader("模型摘要")
        m = metrics_json.get("metrics", {})
        cols = st.columns(3)
        cols[0].metric("Accuracy", f"{m.get('accuracy', 0):.3f}")
        cols[1].metric("Precision", f"{m.get('precision', 0):.3f}")
        cols[2].metric("Recall", f"{m.get('recall', 0):.3f}")
        st.write("訓練時間戳記:", metrics_json.get("timestamp"))

    st.header("1. 資料集與欄位選擇")
    if os.path.exists(data_path):
        df = load_dataset(data_path)
        st.write("資料預覽 (前 5 列)")
        st.dataframe(df.head())

        cols = df.columns.tolist()
        chosen_text_col = st.selectbox("選擇文字欄位 (index)", cols, index=text_col)
        chosen_label_col = st.selectbox("選擇標籤欄位 (index)", cols, index=label_col)
    else:
        st.error(f"找不到資料: {data_path}")
        return

    st.header("2. 班級分佈與熱門詞條")
    label_counts = df.iloc[:, chosen_label_col].value_counts()
    st.bar_chart(label_counts)

    # top terms per class: simple TF-IDF top-n
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf = TfidfVectorizer(stop_words="english", max_features=2000)
    texts = df.iloc[:, chosen_text_col].astype(str).values
    X = tfidf.fit_transform(texts)
    feature_names = np.array(tfidf.get_feature_names_out())

    st.subheader("熱門詞條 (各班級)")
    for lbl in label_counts.index.tolist():
        st.write(f"Class: {lbl}")
        mask = df.iloc[:, chosen_label_col] == lbl
        if mask.sum() == 0:
            st.write("(no examples)")
            continue
        class_tfidf = X[mask.values].mean(axis=0).A1
        top_idx = np.argsort(class_tfidf)[-10:][::-1]
        top_terms = feature_names[top_idx]
        st.write(", ".join(top_terms))

    st.header("3. 混淆矩陣 / ROC / PR (需先訓練 models/)")
    if model is None or test is None:
        st.warning("找不到已訓練的模型或測試切分。請先執行 `scripts/train_model.py` 來產生 models/。")
        col_train = st.columns(1)
        def _train_now():
            # Run training script synchronously and capture output
            try:
                res = subprocess.run(["python3", "scripts/train_model.py"], capture_output=True, text=True, check=False)
                st.session_state["train_log"] = res.stdout + "\n" + res.stderr
                st.session_state["trained"] = True
            except Exception as e:
                st.session_state["train_log"] = f"Training failed: {e}"
                st.session_state["trained"] = False

        if col_train[0].button("一鍵訓練模型 (Train now)"):
            # Call training and then reload model/test if successful
            _train_now()
            # try to reload
            model, le, test = load_model(model_dir)
            if st.session_state.get("trained"):
                st.success("訓練完成，模型已儲存到 models/。請繼續操作。")
            else:
                st.error("訓練失敗，請查看 train_log 欄位。")
        if "train_log" in st.session_state:
            with st.expander("Training log"):
                st.text(st.session_state.get("train_log", ""))
    else:
        X_test = np.array(test["X_test"])
        y_test = np.array(test["y_test"])
        probs = np.array(test["probs"])

        # Confusion matrix at current threshold
        preds = (probs >= threshold).astype(int)
        cm = confusion_matrix(y_test, preds)
        fig_cm = plot_confusion_matrix(cm, labels=["ham", "spam"])
        st.pyplot(fig_cm)

        fig_rocpr = plot_roc_pr(y_test, probs)
        st.pyplot(fig_rocpr)

    st.header("4. 閾值滑桿即時指標")
    if model is not None and test is not None:
        # compute precision/recall/f1 at threshold
        from sklearn.metrics import precision_score, recall_score, f1_score

        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        st.metric("Precision", f"{prec:.3f}")
        st.metric("Recall", f"{rec:.3f}")
        st.metric("F1", f"{f1:.3f}")

    st.header("5. 即時推理")
    # Use session_state to track the input so buttons can update it without experimental_rerun
    st.session_state.setdefault("input_text", "")

    # Text area bound to session_state key; streamlit reruns on button press so value updates
    st.text_area("輸入訊息以進行預測", key="input_text", height=150)
    col1, col2 = st.columns(2)
    # Use on_click callbacks to safely update session_state linked widget
    def _set_input_text(val):
        st.session_state["input_text"] = val

    # Single merged button: generate a random example (spam or ham)
    def _generate_random_example():
        # pick spam or ham randomly
        if random.random() < 0.5:
            txt = random.choice(spam_examples)
            true_label = "spam"
        else:
            txt = random.choice(ham_examples)
            true_label = "ham"
        st.session_state["input_text"] = txt
        st.session_state["generated_true"] = true_label
        # compute probability preview if model available
        if model is not None:
            try:
                p = model.predict_proba([txt])[0]
                prob = float(p[1]) if len(p) > 1 else float(p.max())
            except Exception:
                prob = None
        else:
            prob = None
        st.session_state["generated_prob"] = prob

    col1.button("生成隨機範例", on_click=_generate_random_example)

    # Render probability bar under the example selector (show current example prob if model available)
    def render_prob_bar(prob: float | None, thr: float = 0.5):
        # create ticks from 0.0 to 1.0 step 0.1
        ticks_html = "".join([f"<span style='display:inline-block;width:9%;text-align:center;font-size:12px'>{i/10:.1f}</span>" for i in range(11)])
        thr_pct = max(0.0, min(1.0, float(thr))) * 100
        if prob is None:
            bar_inner = ""
            label = "(模型未準備好)"
        else:
            pct = max(0.0, min(1.0, float(prob))) * 100
            color = "#e74c3c" if prob >= 0.5 else "#2ecc71"
            # absolute-positioned inner bar for overlaying threshold marker
            bar_inner = f"<div style='position:absolute;left:0;top:0;bottom:0;background:{color};width:{pct}%;border-radius:4px;'></div>"
            label = f"機率: {prob:.3f}"

        # container with relative positioning to place threshold marker
        html = f"""
        <div style='width:100%;'>
          <div style='position:relative;border:1px solid #ccc;border-radius:4px;padding:2px;background:#f7f7f7;height:24px;'>
            <div style='position:relative;height:18px;'>
              {bar_inner}
              <div style='position:absolute;left:{thr_pct}%;top:0;bottom:0;width:2px;background:#000;transform:translateX(-1px);opacity:0.9;'></div>
            </div>
          </div>
          <div style='display:flex;justify-content:space-between;margin-top:6px'>{ticks_html}</div>
          <div style='margin-top:6px;font-size:13px'>閾值: {thr:.2f} &nbsp;&nbsp; {label}</div>
        </div>
        """
        return html

    # Show a preview area for the generated example and its probability bar
    st.subheader("隨機範例預覽與預測")
    gen_text = st.session_state.get("input_text", "")
    gen_true = st.session_state.get("generated_true", None)
    gen_prob = st.session_state.get("generated_prob", None)

    if gen_text:
        box = st.container()
        with box:
            st.markdown("<div style='padding:10px;border-radius:8px;background:#f5f7fb'>", unsafe_allow_html=True)
            st.markdown("**範例內容：**")
            st.write(gen_text)
            st.markdown(render_prob_bar(gen_prob, threshold), unsafe_allow_html=True)
            # show prediction and (if known) true label
            if gen_prob is None:
                st.info("模型尚未準備好；請先訓練模型或按一鍵訓練。")
            else:
                pred_label = "spam" if gen_prob >= 0.5 else "ham"
                match = None
                if gen_true is not None:
                    match = (pred_label == gen_true)
                cols_pred = st.columns([1, 1, 2])
                cols_pred[0].metric("預測", pred_label.upper())
                cols_pred[1].metric("機率", f"{gen_prob:.3f}")
                if match is None:
                    cols_pred[2].write("")
                else:
                    if match:
                        cols_pred[2].success(f"與真實標籤一致：{gen_true}")
                    else:
                        cols_pred[2].error(f"與真實標籤不符（真實：{gen_true}）")
            st.markdown("</div>", unsafe_allow_html=True)

    if st.button("預測"):
        if model is None:
            st.error("模型尚未準備好，請先執行訓練腳本。")
        else:
            # use the value from session_state (ensures buttons updated the text area)
            text_for_pred = st.session_state.get("input_text", "")
            try:
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba([text_for_pred])[0]
                    if len(prob) == 2:
                        spam_prob = float(prob[1])
                    else:
                        spam_prob = float(prob.max())
                else:
                    # fallback: use decision_function and sigmoid
                    df = model.decision_function([text_for_pred])[0]
                    spam_prob = 1.0 / (1.0 + np.exp(-float(df)))
            except Exception as e:
                st.error(f"預測失敗: {e}")
                spam_prob = None

            if spam_prob is not None:
                pred_label = int(spam_prob >= threshold)
                # If label encoder present, map label
                if le is not None and hasattr(le, "classes_"):
                    try:
                        label_name = le.classes_[pred_label]
                    except Exception:
                        label_name = ("ham", "spam")[pred_label]
                else:
                    label_name = ("ham", "spam")[pred_label]

                st.write(f"預測標籤: **{label_name}**")
                st.progress(int(spam_prob * 100))
                st.write(f"垃圾郵件機率: {spam_prob:.3f}")
                # show threshold marker
                st.write(f"當前閾值: {threshold:.2f}")


if __name__ == "__main__":
    main()
