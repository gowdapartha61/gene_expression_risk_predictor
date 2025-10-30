import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import sys
import base64
import hashlib
import time

# --- CONFIGURATION & FILE PATHS ---
#DOWNLOADS_PATH = "C:/Users/Partha/Downloads"
USER_DB_FILE = os.path.join('users.csv')
MODEL_FILE_PATH = os.path.join('best_parkinsons_xgb_model.joblib')
FEATURES_FILE_PATH = os.path.join('best_model_features.joblib')
LOGO_FILE_PATH = os.path.join('ExpriMind_Logo.png.jpg')
BACKGROUND_IMG_PATH = os.path.join('ImageForNews_730480_16679535270477610.webp')

# ====================================================================
# I. UI/UTILITY FUNCTIONS
# ====================================================================

def inject_circular_style():
    st.markdown("""
        <style>
        .circular-image {
            border-radius: 50%;
            width: 200px;
            height: 200px;
            object-fit: cover;
            display: block;
            margin-left: auto;
            margin-right: auto;
            margin-bottom: 20px;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
        }
        </style>
    """, unsafe_allow_html=True)

def inject_background_style():
    if os.path.exists(BACKGROUND_IMG_PATH):
        try:
            with open(BACKGROUND_IMG_PATH, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode()

            st.markdown(f"""
                <style>
                .stApp {{
                    background-image: url("data:image/webp;base64,{encoded_string}");
                    background-size: cover;
                    background-attachment: fixed;
                    background-position: center;
                    color: white !important;
                    font-size: 1.1em;
                }}
                .stApp::before {{
                    content: "";
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background-color: rgba(0, 0, 0, 0.4);
                    z-index: -1;
                }}
                .main, .login-card {{
                    background-color: rgba(0, 0, 0, 0.8);
                    border-radius: 10px;
                    box-shadow: 0px 0px 20px rgba(0, 0, 0, 0.2);
                    padding: 20px;
                    color: white !important;
                    margin-top: 20px;
                }}
                [data-testid="stSidebarContent"] {{
                    background-color: white !important;
                    color: #0F172A !important;
                }}
                h1, h2, h3, h4, h5, h6 {{
                    color: white !important;
                }}
                .stTextInput label, .stRadio label, .stFileUploader label {{
                    color: white !important;
                }}
                .stAlert p {{ color: white !important; }}
                .stAlert.error p {{
                    color: white !important;
                    font-weight: bold;
                }}
                </style>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Error reading background image: {e}. Using fallback dark theme.")
    else:
        st.warning("Background image not found. Using dark theme fallback.")

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def init_user_db():
    if not os.path.exists(USER_DB_FILE):
        df = pd.DataFrame(columns=['username', 'password_hash'])
        df.to_csv(USER_DB_FILE, index=False)
    return os.path.exists(USER_DB_FILE)

def check_login(username, password):
    if not os.path.exists(USER_DB_FILE):
        return False
    df = pd.read_csv(USER_DB_FILE)
    password_hash = hash_password(password)
    match = df[(df['username'] == username) & (df['password_hash'] == password_hash)]
    return not match.empty

def register_user(username, password):
    if not init_user_db():
        return False, "Database initialization failed."
    df = pd.read_csv(USER_DB_FILE)
    if username in df['username'].values:
        return False, "Username already exists."
    if len(username) < 4 or len(password) < 6:
        return False, "Username must be at least 4 chars. Password at least 6 chars."
    password_hash = hash_password(password)
    new_user = pd.DataFrame([{'username': username, 'password_hash': password_hash}])
    df = pd.concat([df, new_user], ignore_index=True)
    df.to_csv(USER_DB_FILE, index=False)
    return True, "Registration successful! You can now log in."

# ====================================================================
# II. MODEL LOADING
# ====================================================================
@st.cache_resource
def load_resources():
    if not os.path.exists(MODEL_FILE_PATH) or not os.path.exists(FEATURES_FILE_PATH):
        st.error("Model or feature file missing in Downloads folder.")
        sys.exit()
    try:
        model = joblib.load(MODEL_FILE_PATH)
        feature_list = joblib.load(FEATURES_FILE_PATH)
        return model, feature_list
    except Exception as e:
        st.error(f"Error loading model or features: {e}")
        sys.exit()

model, FEATURE_GENES = load_resources()

# ====================================================================
# III. PREDICTION INTERFACE
# ====================================================================
def prediction_interface():
    st.title("🔬 ExpriMind: Parkinson's Prediction")
    st.header(f"Welcome back, {st.session_state['username']}!")
    st.markdown("---")

    st.info(f"Model: **Optimized XGBoost Classifier** | Features Used: **{len(FEATURE_GENES)} Gene Probes**")
    st.subheader("Upload Patient Gene Expression Data")

    uploaded_file = st.file_uploader("Upload Expression Data (CSV/TXT)", type=['csv', 'txt'])

    if uploaded_file is not None:
        try:
            patient_data = pd.read_csv(uploaded_file, sep=r'[\t,;]', engine='python', index_col=False)
            st.success(f"File loaded — {patient_data.shape[0]} rows, {patient_data.shape[1]} columns")

            # 1️⃣ Handle transposed data automatically
            if len(set(FEATURE_GENES) & set(patient_data.columns)) < len(FEATURE_GENES) / 10:
                if len(set(FEATURE_GENES) & set(patient_data.iloc[:, 0])) > len(FEATURE_GENES) / 10:
                    st.warning("Data appears transposed — correcting orientation automatically.")
                    patient_data = patient_data.set_index(patient_data.columns[0]).T

            # 2️⃣ Drop irrelevant columns
            non_gene_cols = [c for c in patient_data.columns if c not in FEATURE_GENES]
            if non_gene_cols:
                st.info(f"Ignoring {len(non_gene_cols)} non-gene columns (e.g., {non_gene_cols[:3]}...)")
                patient_data = patient_data.drop(columns=non_gene_cols)

            # 3️⃣ Fill missing genes with median
            missing_genes = [g for g in FEATURE_GENES if g not in patient_data.columns]
            if missing_genes:
                st.warning(f"{len(missing_genes)} missing genes filled with median expression.")
                for g in missing_genes:
                    patient_data[g] = np.median(patient_data.select_dtypes(include=[np.number]).values)

            # 4️⃣ Reorder columns
            X_patient = patient_data[FEATURE_GENES]

            # 5️⃣ Predict
            prediction_proba = model.predict_proba(X_patient)
            prediction = model.predict(X_patient)

            results_df = pd.DataFrame({
                'Sample Index': range(1, len(prediction) + 1),
                'Prediction': np.where(prediction == 1, 'Parkinson\'s Disease (PD)', 'Healthy Control'),
                'PD Probability': prediction_proba[:, 1].round(4),
                'Confidence': np.where(prediction == 1,
                                       prediction_proba[:, 1].round(4),
                                       (1 - prediction_proba[:, 1]).round(4)),
            })

            st.subheader("Prediction Results")
            st.dataframe(results_df, hide_index=True)

            if len(prediction) >= 1:
                final_diagnosis = results_df['Prediction'].mode()[0]
                st.markdown(f"## 🧠 Final Diagnosis: **{final_diagnosis}**")

        except Exception as e:
            st.error(f"Error processing file: {e}")

# ====================================================================
# IV. LOGIN / REGISTER INTERFACE
# ====================================================================
def login_page():
    inject_circular_style()
    inject_background_style()
    init_user_db()

    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)

        if os.path.exists(LOGO_FILE_PATH):
            with open(LOGO_FILE_PATH, 'rb') as f:
                encoded_image = base64.b64encode(f.read()).decode()
            st.markdown(f"""
                <div style="text-align: center;">
                    <img src="data:image/jpeg;base64,{encoded_image}" class="circular-image" alt="ExpriMind Logo">
                </div>
            """, unsafe_allow_html=True)
        else:
            st.header("ExpriMind")

        st.markdown("""
            <h1 style='text-align: center; font-size: 3.0em;'>Welcome to ExpriMind</h1>
            <p style='text-align: center; font-size: 1.1em;'>Precision Diagnostics using Gene Expression and XGBoost AI.</p>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Account Access")

        choice = st.radio("Select Option:", ("Login", "Register"), horizontal=True)
        if 'logged_in' not in st.session_state:
            st.session_state['logged_in'] = False
            st.session_state['username'] = None

        if choice == "Login":
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login")
                if submitted:
                    if check_login(username, password):
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = username
                        st.rerun()
                    else:
                        st.error("Invalid Username or Password.")
        else:
            with st.form("register_form"):
                new_username = st.text_input("New Username")
                new_password = st.text_input("New Password (min 6 chars)", type="password")
                submitted = st.form_submit_button("Create Account")
                if submitted:
                    success, message = register_user(new_username, new_password)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)

        st.markdown('</div>', unsafe_allow_html=True)

# ====================================================================
# V. MAIN APP FLOW
# ====================================================================
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if st.session_state['logged_in']:
    prediction_interface()
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state['username'] = None
        st.rerun()
else:
    login_page()
