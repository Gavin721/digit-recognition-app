import os
import re
import io
import zipfile
import tempfile
import numpy as np
import tensorflow as tf
import gdown
import streamlit as st

# ════════════════════════════════════════════
# ⚙️ 預設密碼設定
# ════════════════════════════════════════════
DEFAULT_PASSWORD = "1234"

# Windows 自然排序
def windows_sort_key(filename):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

# 從 Google Drive 網址提取 File ID
def extract_gdrive_file_id(url):
    match = re.search(r'(?:file/d/|id=)([a-zA-Z0-9_-]+)', url)
    return match.group(1) if match else url

st.set_page_config(page_title="手寫數字 AI 盲測系統", layout="centered")

# 🔒 密碼門禁
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "current_password" not in st.session_state:
    st.session_state["current_password"] = DEFAULT_PASSWORD

if not st.session_state["authenticated"]:
    st.title("🔒 存取受限：請輸入密碼")
    user_password = st.text_input("輸入密碼：", type="password")
    if st.button("🔓 驗證登入", type="primary"):
        if user_password == st.session_state["current_password"]:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("❌ 密碼錯誤！")
    st.stop()

# 🚀 系統主介面
st.title("🔢 手寫數字模型 AI 盲測系統")

with st.sidebar:
    st.header("⚙️ 安全設定")
    new_pwd = st.text_input("修改當前密碼", value=st.session_state["current_password"], type="password")
    if new_pwd != st.session_state["current_password"]:
        st.session_state["current_password"] = new_pwd
        st.toast("🔑 密碼已更新！", icon="✅")
    st.write("---")
    if st.button("🔒 登出系統"):
        st.session_state["authenticated"] = False
        st.rerun()

# ─── 1. 模型檔案輸入 ───
st.subheader("1. 提供模型檔 (.h5 / .keras)")
model_input_type = st.radio("模型來源：", ["網頁上傳", "Google Drive 連結"], horizontal=True)

uploaded_model_file = None
model_url_input = ""
if model_input_type == "網頁上傳":
    uploaded_model_file = st.file_uploader("選擇模型檔案 (.h5 / .keras)", type=["h5", "keras"])
else:
    model_url_input = st.text_input("模型檔 Google Drive 分享連結：", placeholder="https://drive.google.com/file/d/...")

# ─── 2. 預測集輸入 ───
st.subheader("2. 提供預測集 (predict.zip)")
dataset_input_type = st.radio("預測集來源：", ["網頁上傳 predict.zip", "Google Drive predict.zip 連結"], horizontal=True)

uploaded_zip_file = None
zip_url_input = ""
if dataset_input_type == "網頁上傳 predict.zip":
    uploaded_zip_file = st.file_uploader("選擇 predict.zip 壓縮檔", type=["zip"])
else:
    zip_url_input = st.text_input("predict.zip Google Drive 分享連結：", placeholder="https://drive.google.com/file/d/...")

# ─── 3. 開始預測 ───
if st.button("🚀 開始執行預測", type="primary"):
    if model_input_type == "網頁上傳" and not uploaded_model_file:
        st.error("❌ 請上傳模型檔案！"); st.stop()
    elif model_input_type == "Google Drive 連結" and not model_url_input:
        st.error("❌ 請輸入模型的 Google Drive 連結！"); st.stop()

    if dataset_input_type == "網頁上傳 predict.zip" and not uploaded_zip_file:
        st.error("❌ 請上傳 predict.zip 檔案！"); st.stop()
    elif dataset_input_type == "Google Drive predict.zip 連結" and not zip_url_input:
        st.error("❌ 請輸入 predict.zip 的 Google Drive 連結！"); st.stop()

    with st.spinner("🤖 正在載入模型與進行盲測預測，請稍候..."):
        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. 載入模型檔案
            model_path = os.path.join(temp_dir, "model.h5")
            if model_input_type == "網頁上傳":
                with open(model_path, "wb") as f:
                    f.write(uploaded_model_file.getbuffer())
            else:
                st.info("🌐 正在下載雲端模型檔...")
                file_id = extract_gdrive_file_id(model_url_input)
                gdown.download(id=file_id, output=model_path, quiet=True)

            model = tf.keras.models.load_model(model_path)

            # 2. 解壓預測集
            zip_path = os.path.join(temp_dir, "data.zip")
            if dataset_input_type == "網頁上傳 predict.zip":
                with open(zip_path, "wb") as f:
                    f.write(uploaded_zip_file.getbuffer())
            else:
                st.info("🌐 正在下載雲端預測集...")
                file_id = extract_gdrive_file_id(zip_url_input)
                gdown.download(id=file_id, output=zip_path, quiet=True)

            extract_path = os.path.join(temp_dir, "extracted")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)

            inner_path = os.path.join(extract_path, "predict")
            predict_dir = inner_path if os.path.exists(inner_path) else extract_path

            # 檢查 0~9 資料夾結構
            missing_folders = [str(i) for i in range(10) if not os.path.exists(os.path.join(predict_dir, str(i)))]
            if missing_folders:
                st.error(f"❌ 預測集結構錯誤！目錄下缺少以下子資料夾 (0~9)：{', '.join(missing_folders)}")
                st.stop()

            # 3. 盲測推論與統計
            total_images = 0
            correct_predictions = 0
            zip_buffer = io.BytesIO()

            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for folder_name in sorted(os.listdir(predict_dir)):
                    folder_path = os.path.join(predict_dir, folder_name)

                    if os.path.isdir(folder_path) and folder_name.isdigit() and 0 <= int(folder_name) <= 9:
                        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG')
                        raw_images = [f for f in os.listdir(folder_path) if f.endswith(image_extensions)]
                        images = sorted(raw_images, key=windows_sort_key)

                        folder_records = []
                        for filename in images:
                            img_path = os.path.join(folder_path, filename)
                            try:
                                img_raw = tf.io.read_file(img_path)
                                img_tensor = tf.image.decode_image(img_raw, channels=3, expand_animations=False)
                                img_resized = tf.image.resize(img_tensor, (224, 224), method='bilinear')
                                img_array = tf.expand_dims(img_resized, 0)

                                predictions = model.predict(img_array, verbose=0)
                                predicted_class = np.argmax(predictions[0])

                                true_label = int(folder_name)
                                is_correct = (predicted_class == true_label)

                                total_images += 1
                                if is_correct:
                                    correct_predictions += 1

                                folder_records.append(f"{filename},{1 if is_correct else 0}\n")
                            except Exception as e:
                                st.warning(f"圖片 {filename} 處理失敗: {e}")

                        zip_file.writestr(f"predict/{folder_name}/statistics.txt", "".join(folder_records))

            final_acc = (correct_predictions / total_images * 100) if total_images > 0 else 0
            st.success(f"🏆 預測完成！共測試 {total_images} 張圖片，實戰正確率為：{final_acc:.2f}%")

            st.download_button(
                label="📥 下載所有子資料夾統計檔 (statistics.zip)",
                data=zip_buffer.getvalue(),
                file_name="statistics.zip",
                mime="application/zip"
            )
