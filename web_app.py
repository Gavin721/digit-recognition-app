import os
import re
import io
import zipfile
import tempfile
import numpy as np
import streamlit as st
import gdown
from PIL import Image

# ════════════════════════════════════════════
# ⚙️ 預設密碼設定
# ════════════════════════════════════════════
DEFAULT_PASSWORD = "1234"

def windows_sort_key(filename):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

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
st.title("🔢 手寫數字模型 AI 盲測系統 (雲端輕量版)")

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

st.subheader("1. 提供預測集 (predict.zip)")
dataset_input_type = st.radio("預測集來源：", ["網頁上傳 predict.zip", "Google Drive predict.zip 連結"], horizontal=True)

uploaded_zip_file = None
zip_url_input = ""
if dataset_input_type == "網頁上傳 predict.zip":
    uploaded_zip_file = st.file_uploader("選擇 predict.zip 壓縮檔", type=["zip"])
else:
    zip_url_input = st.text_input("predict.zip Google Drive 分享連結：", placeholder="https://drive.google.com/file/d/...")

if st.button("🚀 開始執行統計分析", type="primary"):
    if dataset_input_type == "網頁上傳 predict.zip" and not uploaded_zip_file:
        st.error("❌ 請上傳 predict.zip 檔案！"); st.stop()
    elif dataset_input_type == "Google Drive predict.zip 連結" and not zip_url_input:
        st.error("❌ 請輸入 predict.zip 的 Google Drive 連結！"); st.stop()

    with st.spinner("🤖 正在解析資料集並生成統計檔，請稍候..."):
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, "data.zip")
            if dataset_input_type == "網頁上傳 predict.zip":
                with open(zip_path, "wb") as f:
                    f.write(uploaded_zip_file.getbuffer())
            else:
                file_id = extract_gdrive_file_id(zip_url_input)
                gdown.download(id=file_id, output=zip_path, quiet=True)

            extract_path = os.path.join(temp_dir, "extracted")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)

            inner_path = os.path.join(extract_path, "predict")
            predict_dir = inner_path if os.path.exists(inner_path) else extract_path

            # 盲測與統計處理
            total_images = 0
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
                                # 影像基本驗證
                                with Image.open(img_path) as img:
                                    img.verify()
                                total_images += 1
                                folder_records.append(f"{filename},1\n")
                            except Exception as e:
                                st.warning(f"圖片 {filename} 讀取失敗: {e}")

                        zip_file.writestr(f"predict/{folder_name}/statistics.txt", "".join(folder_records))

            st.success(f"🏆 解析完成！已成功統計 {total_images} 張圖片。")

            st.download_button(
                label="📥 一鍵下載所有子資料夾統計檔 (statistics.zip)",
                data=zip_buffer.getvalue(),
                file_name="statistics.zip",
                mime="application/zip"
            )
