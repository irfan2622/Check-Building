import streamlit as st
import os
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
import shutil
import zipfile

# --- CONFIGURATION ---
MODEL_PATH = "C:/Users/M Irfansyah/Downloads/Project2/runs/detect/train/weights/best.pt"  # <--- Nama file model Anda yang sudah "ditanam"
TEMP_DIR = "temp_process"

# --- HELPER FUNCTIONS ---
def reset_temp_dir():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(f"{TEMP_DIR}/building", exist_ok=True)
    os.makedirs(f"{TEMP_DIR}/not_building", exist_ok=True)

def create_zip(path_to_zip):
    byte_io = BytesIO()
    with zipfile.ZipFile(byte_io, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(path_to_zip):
            for file in files:
                zf.write(os.path.join(root, file), 
                         os.path.relpath(os.path.join(root, file), path_to_zip))
    byte_io.seek(0)
    return byte_io

# --- LOAD MODEL (Cache agar tidak reload setiap saat) ---
@st.cache_resource
def load_yolo_model():
    if os.path.exists(MODEL_PATH):
        return YOLO(MODEL_PATH)
    else:
        st.error(f"Model '{MODEL_PATH}' tidak ditemukan di folder aplikasi!")
        return None

# --- STREAMLIT UI ---
# --- 1. DATABASE USER (Username: Password) ---
USER_DB = {
    "admin": "rahasia123",
    "lennox": "yolo2026",
    "analyst_1": "pass789",
    "user_test": "testing321"
}

def check_password():
    """Fungsi pengecekan login untuk banyak user"""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        st.sidebar.title("🔐 Login Sistem")
        user_input = st.sidebar.text_input("Username")
        pw_input = st.sidebar.text_input("Password", type="password")
        
        if st.sidebar.button("Login"):
            # Mengecek apakah username ada di database DAN passwordnya cocok
            if user_input in USER_DB and USER_DB[user_input] == pw_input:
                st.session_state["authenticated"] = True
                st.session_state["current_user"] = user_input # Simpan nama user yang login
                st.rerun()
            else:
                st.sidebar.error("Username atau Password salah")
        return False
    return True

# --- 2. JALANKAN APLIKASI ---
if check_password():
    st.sidebar.success(f"Login sebagai: {st.session_state['current_user']}")
    
    # Tombol Logout
    if st.sidebar.button("Logout"):
        st.session_state["authenticated"] = False
        st.rerun()

    st.set_page_config(page_title="Building Classifier AI", layout="centered")
    st.title("🏢 Building Classifier & URL Converter")

    model = load_yolo_model()

    if model:
        st.info(f"Model '{MODEL_PATH}' siap digunakan.")
    
    # Drag and Drop Excel
        uploaded_file = st.file_uploader("Upload file Excel dengan kolom 'URL'", type=['xlsx'])

        if uploaded_file:
         df = pd.read_excel(uploaded_file)
         st.write("Preview Data:", df.head(3))
        
        if st.button("🚀 Start Processing"):
            reset_temp_dir()
            data_hasil = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Buat list untuk menampung gambar agar bisa dipreview (opsional)
            building_count = 0
            
            for index, row in df.iterrows():
                url_asli = str(row['URL']).strip()

                # A. MANIPULASI URL DENGAN KONDISI
                # Jika URL sudah mengandung link API download, jangan diubah lagi
                if "ddd-file-storage-x6ueszs4dq-et.a.run.app/api/v1/download?file=" in url_asli:
                    url_download = url_asli
                else:
                    # Jika URL masih format Google Storage, lakukan konversi
                    path_bersih = url_asli.replace("https://storage.googleapis.com/ddd-file-storage/", "")
                    url_download = f"https://ddd-file-storage-x6ueszs4dq-et.a.run.app/api/v1/download?file={path_bersih}"
                
                status_text.text(f"Processing row {index+1}/{len(df)}...")
                
                # A. Manipulasi URL
                #path_bersih = url_asli.replace("https://storage.googleapis.com/ddd-file-storage/", "")
                #url_download = f"https://ddd-file-storage-x6ueszs4dq-et.a.run.app/api/v1/download?file={path_bersih}"
                #status_text.text(f"Processing row {index+1}/{len(df)}...")
                
                try:
                    response = requests.get(url_download, timeout=15)
                    if response.status_code == 200:
                        img = Image.open(BytesIO(response.content))
                        
                        # B. YOLO Prediction
                        results = model.predict(source=img, conf=0.25, save=False)
                        r = results[0]
                        
                        nama_file = f"foto_{index+1}.jpg"
                        if len(r.boxes) > 0:
                            klasifikasi = 'Building'
                            building_count += 1
                            r.save(filename=os.path.join(TEMP_DIR, "building", nama_file))
                        else:
                            klasifikasi = 'Not Building'
                            img.save(os.path.join(TEMP_DIR, "not_building", nama_file))
                        
                        data_hasil.append({'URL': url_download, 'klasifikasi': klasifikasi})
                    else:
                        data_hasil.append({'URL': url_download, 'klasifikasi': 'Error Download'})
                except Exception as e:
                    data_hasil.append({'URL': url_download, 'klasifikasi': f'Error: {e}'})
                
                progress_bar.progress((index + 1) / len(df))

            # Simpan Excel Rekap
            df_hasil = pd.DataFrame(data_hasil)
            excel_path = os.path.join(TEMP_DIR, "Laporan_Klasifikasi.xlsx")
            df_hasil.to_excel(excel_path, index=False)
            
            st.success(f"✅ Selesai! {building_count} Building terdeteksi.")

            # --- DOWNLOAD SECTION ---
            zip_data = create_zip(TEMP_DIR)
            st.download_button("📦 Download Semua (ZIP)", zip_data, "hasil_klasifikasi.zip", use_container_width=True)
    else:
        st.warning("Pastikan file 'best.pt' sudah ada di folder yang sama dengan script ini.")