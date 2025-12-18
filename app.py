import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Frequency Domain Filtering | Project 04-01",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Styling ---
st.markdown("""
<style>
    .main-header {font-size: 32px; font-weight: bold; color: #2C3E50; margin-bottom: 20px;}
    .sub-header {font-size: 24px; font-weight: bold; color: #34495E; margin-top: 20px;}
    .metric-card {background-color: #F0F2F6; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;}
</style>
""", unsafe_allow_html=True)


# --- Helper Functions ---

def load_and_resize_image(image_path, target_size=(256, 256)):
    """Loads an image and resizes it to even dimensions to ensure FFT symmetry."""
    if not os.path.exists(image_path):
        return None
    img = cv2.imread(image_path, 0)
    if img is None:
        return None
    # Resize to specific square size or nearest even numbers for stability
    img_resized = cv2.resize(img, target_size)
    return img_resized


def process_uploaded_image(file_bytes):
    """Decodes and resizes uploaded image."""
    img = cv2.imdecode(file_bytes, 0)
    # Resize to standard 256x256 or 512x512 for consistent FFT results
    # or just ensure they are even numbers
    h, w = img.shape
    new_h = h if h % 2 == 0 else h - 1
    new_w = w if w % 2 == 0 else w - 1
    img = cv2.resize(img, (new_w, new_h))
    return img


def generate_centering_mask(M, N):
    """Generates the (-1)^(x+y) mask."""
    x = np.arange(M)
    y = np.arange(N)
    x_grid, y_grid = np.meshgrid(x, y, indexing='ij')
    return np.power(-1, x_grid + y_grid)


def create_gaussian_filter(M, N, cutoff, filter_type='Low Pass'):
    """Creates a centered Gaussian filter."""
    u = np.arange(M)
    v = np.arange(N)
    u_idx, v_idx = np.meshgrid(u, v, indexing='ij')

    # Distance from center (M/2, N/2)
    D = np.sqrt((u_idx - M / 2) ** 2 + (v_idx - N / 2) ** 2)

    # Gaussian Filter Formula
    H = np.exp(-(D ** 2) / (2 * (cutoff ** 2)))

    if filter_type == 'High Pass':
        H = 1 - H

    return H


# --- Core Processing Logic (The "Package") ---
def project_04_01_workflow(image, cutoff=30, filter_type='Low Pass'):
    """
    Implements the exact steps a-e from Project 04-01.
    """
    M, N = image.shape

    # Step (a): Multiply by (-1)^(x+y)
    centering_mask = generate_centering_mask(M, N)
    centered_spatial = image * centering_mask

    # Step (b): Compute FFT
    F_u_v = np.fft.fft2(centered_spatial)

    # Step (c): Multiply by Real Filter Function
    H = create_gaussian_filter(M, N, cutoff, filter_type)
    G_u_v = F_u_v * H

    # Step (d): Inverse FFT
    g_centered_spatial = np.fft.ifft2(G_u_v)

    # Step (e): Post-process (Multiply by (-1)^(x+y) and take Real part)
    g_spatial = np.real(g_centered_spatial * centering_mask)

    # FIX: Convert float result back to uint8 [0-255] for correct display
    # We clip values to stay in valid range
    g_spatial_display = np.clip(g_spatial, 0, 255).astype(np.uint8)

    # For visualization/Project 04-02: Compute Spectrum
    # Log transformation: 20 * log(1 + |F|)
    spectrum = 20 * np.log(1 + np.abs(F_u_v))

    return {
        "centered_spectrum": spectrum,
        "filter_mask": H,
        "filtered_image": g_spatial_display,  # The fixed image for display
        "fft_complex_raw": F_u_v
    }


# --- Sidebar ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["1. Theory & Explanation", "2. Project 04-01 Implementation",
                                   "3. Interactive Dashboard (Bonus)"])

# --- PAGE 1: THEORY ---
if page == "1. Theory & Explanation":
    st.markdown('<div class="main-header">Theory: 2D FFT & Filtering</div>', unsafe_allow_html=True)

    st.markdown("### The Workflow")
    st.write("This project implements Frequency Domain Filtering using the centered Fourier Transform.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.info("**Step 1: Centering**")
        st.write(
            "Standard FFT places the zero-frequency (DC) component at the corner $(0,0)$. To move it to the center $(M/2, N/2)$, we multiply the input image by:")
        st.latex(r"f_{centered}(x,y) = f(x,y) \cdot (-1)^{x+y}")

        st.info("**Step 2: Filtering**")
        st.write("Filtering is performed by element-wise multiplication in the frequency domain:")
        st.latex(r"G(u,v) = F(u,v) \cdot H(u,v)")

    with col2:
        img_path = "data/section_4.7.3.png"
        if os.path.exists(img_path):
            st.image(img_path, caption="Textbook Figure 4.7.3: Illustration of Centering", use_container_width=True)
        else:
            st.warning("Image 'data/section_4.7.3.png' not found.")

# --- PAGE 2: PROJECT IMPLEMENTATION ---
elif page == "2. Project 04-01 Implementation":
    st.markdown('<div class="main-header">Project 04-01 & 04-02: Results</div>', unsafe_allow_html=True)
    st.write("This section applies the \"Package\" developed in Project 04-01 to the test image `fig4.41.png`.")

    # FIX: Load AND Resize to 256x256 to solve "Odd Dimensions" Average error
    img_path = "data/fig4.41.png"
    original_img = load_and_resize_image(img_path, target_size=(256, 256))

    if original_img is None:
        st.error(f"Could not load {img_path}. Please make sure the file exists in the 'data' directory.")
    else:
        results = project_04_01_workflow(original_img, cutoff=40, filter_type='Low Pass')

        # --- Display Section 1: Project 04-02 Average Value ---
        st.markdown('<div class="sub-header">Project 04-02: Average Value Calculation</div>', unsafe_allow_html=True)

        M, N = original_img.shape

        # Robust DC extraction: Find the max magnitude in the spectrum (Peak is always DC)
        # This avoids coordinate errors if M/N are odd.
        dc_component_mag = np.max(np.abs(results["fft_complex_raw"]))

        avg_value_fft = dc_component_mag / (M * N)
        avg_value_spatial = np.mean(original_img)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""<div class="metric-card"><b>Image Dimensions</b><br>{M} x {N} pixels</div>""",
                        unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card"><b>DC Component Magnitude</b><br>{dc_component_mag:.2f}</div>""",
                        unsafe_allow_html=True)
        with col3:
            st.markdown(
                f"""<div class="metric-card"><b>Calculated Average Intensity</b><br>{avg_value_fft:.4f}</div>""",
                unsafe_allow_html=True)

        # Check match
        if abs(avg_value_fft - avg_value_spatial) < 1.0:
            st.success(
                f"Verification: NumPy mean of spatial image is {avg_value_spatial:.4f}. The values match perfectly!")
        else:
            st.warning(f"Verification: NumPy mean is {avg_value_spatial:.4f}. Discrepancy detected.")

        # --- Display Section 2: Visualizations ---
        st.markdown('<div class="sub-header">Visual Results (Project 04-01)</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        with c1:
            st.image(original_img, caption="1. Original Image (fig4.41.png)", use_container_width=True)

        with c2:
            fig, ax = plt.subplots()
            ax.imshow(results["centered_spectrum"], cmap='jet')
            ax.axis('off')
            st.pyplot(fig)
            st.caption("2. Centered Fourier Spectrum (Log Scale)")

        with c3:
            # Displays the FIXED uint8 image
            st.image(results["filtered_image"], caption="3. Result after Low Pass Filter", use_container_width=True)

# --- PAGE 3: INTERACTIVE DASHBOARD ---
elif page == "3. Interactive Dashboard (Bonus)":
    st.markdown('<div class="main-header">Interactive Frequency Filtering Dashboard</div>', unsafe_allow_html=True)
    st.write("Upload any image to apply the Project 04-01 workflow dynamically.")

    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg', 'tif'])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        user_img = process_uploaded_image(file_bytes)  # Uses the robust resize function

        st.sidebar.markdown("---")
        st.sidebar.header("Filter Settings")
        filter_mode = st.sidebar.radio("Filter Type", ["Low Pass", "High Pass"])
        cutoff_freq = st.sidebar.slider("Cutoff Frequency (D0)", 5, 100, 30)

        res = project_04_01_workflow(user_img, cutoff=cutoff_freq, filter_type=filter_mode)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Spatial Domain")
            st.image(user_img, caption="Input Image", use_container_width=True)
            st.markdown("---")
            st.image(res["filtered_image"], caption=f"Output ({filter_mode} Filter)", use_container_width=True)

        with col2:
            st.subheader("Frequency Domain")
            fig, ax = plt.subplots()
            ax.imshow(res["centered_spectrum"], cmap='jet')
            ax.axis('off')
            st.pyplot(fig)
            st.caption("Centered Magnitude Spectrum")

            st.markdown("---")
            fig2, ax2 = plt.subplots()
            ax2.imshow(res["filter_mask"], cmap='gray')
            ax2.axis('off')
            st.pyplot(fig2)
            st.caption(f"Filter Mask (D0={cutoff_freq})")

    else:
        st.info("Waiting for image upload...")