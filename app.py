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
# !!! VERIFY THIS PATH IS CORRECT for your user 'Partha' !!!
DOWNLOADS_PATH = "C:/Users/Partha/Downloads"

# Database file for storing users
USER_DB_FILE = os.path.join(DOWNLOADS_PATH, 'users.csv')

# Model Files
MODEL_FILE_PATH = os.path.join(DOWNLOADS_PATH, 'best_parkinsons_xgb_model.joblib')
FEATURES_FILE_PATH = os.path.join(DOWNLOADS_PATH, 'best_model_features.joblib')
LOGO_FILE_PATH = os.path.join(DOWNLOADS_PATH, 'ExpriMind_Logo.png.jpg')
BACKGROUND_IMG_PATH = os.path.join(DOWNLOADS_PATH, 'ImageForNews_730480_16679535270477610.webp')


# ====================================================================
# I. UI/UTILITY FUNCTIONS (Authentication, Styling, etc.)
# ====================================================================

def inject_circular_style():
    """Injects CSS to make the image circular and center it."""
    st.markdown("""
        <style>
        .circular-image {
            border-radius: 50%; /* Makes the image circular */
            width: 200px; /* Set a fixed width */
            height: 200px; /* Set a fixed height */
            object-fit: cover; /* Ensures the image covers the area without distortion */
            display: block;
            margin-left: auto;
            margin-right: auto;
            margin-bottom: 20px;
        }
        /* Custom styling for the main container to center content */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
        }
        </style>
        """, unsafe_allow_html=True)

def inject_background_style():
    """Injects CSS for background image with yellow text. Removed black overlay box."""

    if os.path.exists(BACKGROUND_IMG_PATH):
        try:
            # Read image file and encode it to base64 for embedding in CSS
            with open(BACKGROUND_IMG_PATH, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode()

            # Apply the style with the embedded image and yellow text
            st.markdown(
                f"""
                <style>
                /* Apply background image to the entire body */
                .stApp {{
                    background-image: url("data:image/webp;base64,{encoded_string}");
                    background-size: cover;
                    background-attachment: fixed;
                    background-position: center;
                    color: #FFD700 !important; /* Base text color set to yellow */
                    font-size: 1.1em; /* Increased base font size */
                }}

                /* NO dark overlay - removed ::before pseudo-element */

                /* Content Area Styling - transparent background, no black box */
                .main, .login-card {{
                    background-color: transparent !important;
                    border-radius: 10px;
                    box-shadow: none !important;
                    padding: 20px;
                    color: #FFD700 !important; /* Yellow text inside cards including login */
                    margin-top: 20px;
                }}

                /* Sidebar Styling */
                [data-testid="stSidebarContent"] {{
                    background-color: white !important;
                    color: #FFD700 !important; /* Yellow text in sidebar */
                }}

                /* All headers now yellow */
                h1, h2, h3, h4, h5, h6 {{
                    color: #FFD700 !important; /* Yellow headers */
                }}

                /* Ensure all text elements are yellow */
                .stTextInput label, .stRadio label, .stFileUploader label, .stMarkdown, p, span, div {{
                    color: #FFD700 !important;
                }}
                
                [data-testid="stSidebarContent"] h1, [data-testid="stSidebarContent"] h2, [data-testid="stSidebarContent"] h3 {{
                    color: #FFD700 !important; /* Yellow sidebar headers */
                }}
                
                .stButton button, .stDownloadButton button {{ 
                    color: #0F172A !important; /* Keep button text dark for readability */
                }}

                /* Alert text yellow */
                .stAlert p {{ color: #FFD700 !important; }}
                .stAlert.error p {{ color: #FFD700 !important; font-weight: bold; }}

                </style>
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            st.warning(f"Error reading background image: {e}. Falling back to dark theme with yellow text.")
            # Fallback code
            st.markdown(
                """
                <style>
                /* Fallback dark theme with yellow text */
                .stApp { background-color: #0F172A; color: #FFD700 !important; }
                .main, .login-card { background-color: transparent !important; color: #FFD700 !important; box-shadow: none !important; }
                [data-testid="stSidebarContent"] { background-color: white !important; color: #FFD700 !important; }
                h1, h2, h3, h4, h5, h6 { color: #FFD700 !important; }
                .stTextInput label, .stRadio label, .stFileUploader label { color: #FFD700 !important; }
                .stAlert p { color: #FFD700 !important; }
                </style>
                """, unsafe_allow_html=True
            )
    else:
        st.warning(f"Background image not found at {BACKGROUND_IMG_PATH}. Falling back to dark theme with yellow text.")
        # Fallback code
        st.markdown(
             """
             <style>
             /* Fallback dark theme with yellow text */
             .stApp { background-color: #0F172A; color: #FFD700 !important; }
             .main, .login-card { background-color: transparent !important; color: #FFD700 !important; box-shadow: none !important; }
             [data-testid="stSidebarContent"] { background-color: white !important; color: #FFD700 !important; }
             h1, h2, h3, h4, h5, h6 { color: #FFD700 !important; }
             .stTextInput label, .stRadio label, .stFileUploader label { color: #FFD700 !important; }
             .stAlert p { color: #FFD700 !important; }
             </style>
             """, unsafe_allow_html=True
         )

def hash_password(password):
    """Hashes the password for secure storage."""
    return hashlib.sha256(password.encode()).hexdigest()

def init_user_db():
    """Initializes the user database CSV file."""
    if not os.path.exists(USER_DB_FILE):
        df = pd.DataFrame(columns=['username', 'password_hash'])
        df.to_csv(USER_DB_FILE, index=False)
    return os.path.exists(USER_DB_FILE)

def check_login(username, password):
    """Checks if credentials are valid."""
    if not os.path.exists(USER_DB_FILE):
        return False

    df = pd.read_csv(USER_DB_FILE)
    password_hash = hash_password(password)

    match = df[(df['username'] == username) & (df['password_hash'] == password_hash)]
    return not match.empty

def register_user(username, password):
    """Registers a new user and adds them to the database."""
    if not init_user_db():
        return False, "Database initialization failed."

    df = pd.read_csv(USER_DB_FILE)

    if username in df['username'].values:
        return False, "Username already exists."

    if len(username) < 4 or len(password) < 6:
        return False, "Username must be at least 4 chars. Password must be at least 6 chars."

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
    """Loads the trained model and feature list once upon app start."""

    if not os.path.exists(MODEL_FILE_PATH) or not os.path.exists(FEATURES_FILE_PATH):
        st.error("FATAL ERROR: Model or features file not found.")
        st.markdown(f"Please ensure files exist at the designated path: {DOWNLOADS_PATH}")
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
# III. PREDICTION INTERFACE (Main Application View)
# ====================================================================

def prediction_interface():
    st.title("🔬 ExpriMind: Parkinson's Prediction")
    st.header(f"Welcome back, {st.session_state['username']}!")
    st.markdown("---")

    st.info(f"Model: Optimized XGBoost Classifier | Final AUC: 0.7390 | Features Used: {len(FEATURE_GENES)} Gene Probes")

    st.subheader("Upload Patient Gene Expression Data")
    st.markdown("Upload a CSV or TXT file containing the expression levels for the required gene probes.")

    uploaded_file = st.file_uploader("Upload Raw Expression Data (CSV/TXT)", type=['csv', 'txt'])

    if uploaded_file is not None:
        try:
            # Read the uploaded file with a robust separator detection
            patient_data = pd.read_csv(uploaded_file, sep=r'[\t,;]', engine='python', index_col=False)

            st.success(f"File loaded successfully. Data contains {patient_data.shape[0]} sample(s).")

            # --- Validation ---
            missing_features = [gene for gene in FEATURE_GENES if gene not in patient_data.columns]

            if missing_features:
                st.error(f"Data Error: The uploaded file is missing {len(missing_features)} required gene probe(s).")
                st.markdown(f"Missing Top 5: {missing_features[:5]}...")
            else:
                # Select and order the features exactly as the model expects
                X_patient = patient_data[FEATURE_GENES]

                # --- Make Prediction ---
                prediction_proba = model.predict_proba(X_patient)
                prediction = model.predict(X_patient)

                st.subheader("Prediction Results")

                # Format results for display
                results_df = pd.DataFrame({
                    'Sample Index': range(1, len(prediction) + 1),
                    'Prediction': np.where(prediction == 1, 'Parkinson\'s Disease (PD)', 'Healthy Control'),
                    'PD Probability': prediction_proba[:, 1].round(4),
                    'Confidence': np.where(prediction == 1,
                                           prediction_proba[:, 1].round(4),
                                           (1 - prediction_proba[:, 1]).round(4)),
                })

                st.dataframe(results_df.style.highlight_max(axis=1, subset=['Confidence'], color='#00d9701a'), hide_index=True)

                # Highlight the most common diagnosis
                if len(prediction) >= 1:
                    final_diagnosis = results_df['Prediction'].mode()[0]
                    st.markdown(f"## Patient Diagnosis: {final_diagnosis}")

        except Exception as e:
            st.exception(f"An unexpected error occurred: {e}")


# ====================================================================
# IV. LOGIN/REGISTER INTERFACE (Authentication Page)
# ====================================================================

def login_page():
    # Inject both custom styles globally
    inject_circular_style()
    inject_background_style()

    # Initialize the database file
    init_user_db()

    # Create a layout with columns for aesthetic centering
    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        # Wrap all login content in the 'login-card' div for contrast
        st.markdown('<div class="login-card">', unsafe_allow_html=True)

        # 1. Logo Display (Circular)
        if os.path.exists(LOGO_FILE_PATH):
            try:
                with open(LOGO_FILE_PATH, 'rb') as f:
                    image_data = f.read()
                encoded_image = base64.b64encode(image_data).decode()

                st.markdown(
                    f"""
                    <div style="text-align: center;">
                        <img src="data:image/jpeg;base64,{encoded_image}"
                             class="circular-image"
                             alt="ExpriMind Logo">
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Error displaying logo. Check file format/permissions. Error: {e}")
                st.header("ExpriMind")

        else:
            st.warning("Logo file not found. Displaying text header.")
            st.header("ExpriMind")

        # 2. Main Title and Tagline
        st.markdown(
            """
            <h1 style='text-align: center; font-size: 3.0em;'>
                Welcome to ExpriMind 
            </h1>
            <p style='text-align: center; font-size: 1.1em;'>
                Precision Diagnostics using Gene Expression and XGBoost AI.
            </p>
            """,
            unsafe_allow_html=True
        )

        # 3. Visual Hook
        st.markdown("---")
        # Placeholder text color now uses the default white set in CSS
        st.markdown("<p style='text-align: center; font-weight: bold; font-size: 1.1em;'>— Biometric Gene Scan Placeholder —</p>", unsafe_allow_html=True)
        st.markdown("---")

        # 4. Login/Register Form Area
        st.subheader("Account Access")

        choice = st.radio("Select Option:", ("Login", "Register"), horizontal=True)

        if 'logged_in' not in st.session_state:
            st.session_state['logged_in'] = False
            st.session_state['username'] = None

        if choice == "Login":
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")

                submitted = st.form_submit_button("Login to ExpriMind", type="primary")

                if submitted:
                    if check_login(username, password):
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = username
                        st.rerun()
                    else:
                        st.error("Invalid Username or Password.")

        elif choice == "Register":
            with st.form("register_form"):
                new_username = st.text_input("New Username")
                new_password = st.text_input("New Password (min 6 chars)", type="password")
                submitted = st.form_submit_button("Create Account", type="primary")

                if submitted:
                    success, message = register_user(new_username, new_password)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)

        # Close the login-card div
        st.markdown('</div>', unsafe_allow_html=True)


# ====================================================================
# V. MAIN APP FLOW
# ====================================================================

# Set initial session state if not present
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# Control flow based on login status
if st.session_state['logged_in']:
    prediction_interface()
    # Add a logout button to the sidebar
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state['username'] = None
        st.rerun()
else:
    login_page()