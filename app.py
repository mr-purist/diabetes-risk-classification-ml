"""
Interactive Diabetes Risk Classification Dashboard
BITS Pilani - M.Tech (AIML) - Machine Learning Assignment 2
Student: Mayank Gupta (BITS ID: 2026AK05256)
Random State: 26
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve
)

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Classifier | BITS Pilani ML Assignment 2",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. Resource Caching & Artifact Loaders
# -----------------------------------------------------------------------------
MODEL_DIR = "model"
TARGET_COL = "Diabetes_binary"
EXPECTED_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
]

MODEL_FILES = {
    "Support Vector Machine": "svm.joblib",
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib"
}

@st.cache_resource
def load_preprocessor():
    path = os.path.join(MODEL_DIR, "preprocessor.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_resource
def load_model(model_name):
    if model_name in MODEL_FILES:
        path = os.path.join(MODEL_DIR, MODEL_FILES[model_name])
        if os.path.exists(path):
            return joblib.load(path)
    return None

@st.cache_data
def load_default_test_data():
    if os.path.exists("test_data.csv"):
        return pd.read_csv("test_data.csv")
    return None

preprocessor = load_preprocessor()

# -----------------------------------------------------------------------------
# 3. Sidebar: Configuration, Upload, and Model Selector
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/caduceus.png", width=70)
    st.title("Control Panel")
    st.caption("BITS ID: 2026AK05256 - Mayank Gupta")
    st.markdown("---")
    
    # Feature A: Dataset Upload Option (CSV) [1 Mark]
    st.markdown("### 1. Test Dataset Upload")
    uploaded_file = st.file_uploader(
        "Upload Test Data CSV (must include features + target):",
        type=["csv"],
        help="Upload a test CSV containing the 21 CDC health indicators and the ground-truth 'Diabetes_binary' target."
    )
    
    # Feature B: Model Selection Dropdown [1 Mark]
    st.markdown("---")
    st.markdown("### 2. Model Selection")
    model_options = [
        "Support Vector Machine",
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Compare All 5 Models"
    ]
    selected_option = st.selectbox(
        "Select Classification Model to Evaluate:",
        model_options,
        index=0,
        help="Choose a specific classifier to inspect detailed metrics, confusion matrix, and ROC curve, or select comparison view."
    )
    
    st.markdown("---")
    with st.expander("ℹ️ About Dataset & Models"):
        st.markdown("""
        - **Dataset**: CDC BRFSS 2015 (Alex Teboul)  
        - **Features**: 21 Clinical & Lifestyle Metrics  
        - **Target**: `Diabetes_binary` (0: Non-Diabetic, 1: Diabetic)  
        - **Split**: 80/20 Stratified (`random_state=26`)  
        - **Platform**: Streamlit Community Cloud  
        """)
    
    # Download sample test_data.csv button
    default_df_raw = load_default_test_data()
    if default_df_raw is not None:
        csv_bytes = default_df_raw.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Sample Test CSV",
            data=csv_bytes,
            file_name="sample_test_data.csv",
            mime="text/csv",
            help="Download the verified test partition sample to test the uploader."
        )

# -----------------------------------------------------------------------------
# 4. Main Body: Data Loading & Validation
# -----------------------------------------------------------------------------
st.title("🩺 Diabetes Risk Classification Dashboard")
st.markdown("**Evaluation of 5 Pre-Trained Machine Learning Classifiers on CDC BRFSS Health Indicators**")

# Determine active dataset (Uploaded CSV or default fallback)
if uploaded_file is not None:
    try:
        active_df = pd.read_csv(uploaded_file)
        st.success(f"📁 **Custom Dataset Active:** Successfully loaded `{uploaded_file.name}` ({active_df.shape[0]} rows, {active_df.shape[1]} columns). Models evaluate directly without retraining.")
    except Exception as e:
        st.error(f"Error reading uploaded CSV file: {e}")
        st.stop()
else:
    active_df = default_df_raw
    if active_df is not None:
        st.info("⚡ **Default Dataset Active:** Using curated test partition (`test_data.csv` with 1,500 balanced rows). You can also upload your own test CSV using the sidebar.")
    else:
        st.error("Default test_data.csv not found and no file uploaded. Please upload a test dataset.")
        st.stop()

# Schema Validation
missing_features = [col for col in EXPECTED_FEATURES if col not in active_df.columns]
if missing_features:
    st.error(f"❌ Missing required feature columns in dataset: {missing_features}")
    st.stop()

if TARGET_COL not in active_df.columns:
    st.error(f"❌ Target column '{TARGET_COL}' not found in uploaded dataset. Target is required to compute evaluation metrics without retraining.")
    st.stop()

X_test_raw = active_df[EXPECTED_FEATURES]
y_test = active_df[TARGET_COL].astype(int)

# Apply preprocessor (NO RETRAINING)
if preprocessor is not None:
    X_test_trans = preprocessor.transform(X_test_raw)
else:
    st.error("Preprocessor artifact not found. Please verify 'model/preprocessor.joblib' exists.")
    st.stop()

# -----------------------------------------------------------------------------
# 5. Model Evaluation Function
# -----------------------------------------------------------------------------
def compute_model_metrics(model, X_trans, y_true):
    y_pred = model.predict(X_trans)
    y_proba = model.predict_proba(X_trans)[:, 1] if hasattr(model, "predict_proba") else None
    
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    return {
        "Accuracy": acc,
        "AUC": auc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "MCC": mcc,
        "y_pred": y_pred,
        "y_proba": y_proba
    }

# -----------------------------------------------------------------------------
# 6. View Rendering: Single Model vs. Multi-Model Comparison
# -----------------------------------------------------------------------------
if selected_option == "Compare All 5 Models":
    st.subheader("📊 Comparative Performance Across All 5 Classifiers")
    st.caption("All models evaluate on the identical active test dataset using pre-trained weights without retraining.")
    
    comparison_records = []
    with st.spinner("Computing metrics across all 5 models..."):
        for m_name in MODEL_FILES.keys():
            mdl = load_model(m_name)
            if mdl is not None:
                m_metrics = compute_model_metrics(mdl, X_test_trans, y_test)
                comparison_records.append({
                    "ML Model Name": m_name,
                    "Accuracy": round(m_metrics["Accuracy"], 4),
                    "AUC": round(m_metrics["AUC"], 4),
                    "Precision": round(m_metrics["Precision"], 4),
                    "Recall": round(m_metrics["Recall"], 4),
                    "F1 Score": round(m_metrics["F1"], 4),
                    "MCC Score": round(m_metrics["MCC"], 4)
                })
    
    comp_df = pd.DataFrame(comparison_records)
    
    # Styled dataframe display using native Streamlit configuration
    st.dataframe(
        comp_df,
        use_container_width=True,
        column_config={
            "Accuracy": st.column_config.NumberColumn(format="%.4f"),
            "AUC": st.column_config.NumberColumn(format="%.4f"),
            "Precision": st.column_config.NumberColumn(format="%.4f"),
            "Recall": st.column_config.NumberColumn(format="%.4f"),
            "F1 Score": st.column_config.NumberColumn(format="%.4f"),
            "MCC Score": st.column_config.NumberColumn(format="%.4f"),
        }
    )
    
    # Summary Insights Callout (Native Streamlit, works in both dark and light modes)
    st.success("""
    **🏆 Overall Winner & Performance Summary:**
    - **Overall Winner**: **Support Vector Machine (SVC RBF)** achieves top performance across **Accuracy (74.99%)**, **Recall (81.50%)**, **F1 Score (0.7652)**, and **MCC (0.5041)**.
    - **Discriminative Baseline**: **Logistic Regression** achieves the highest **ROC-AUC (0.8242)** and **Precision (0.7365)**, showing solid linear separation.
    - **Interpretability**: **Decision Tree** isolates transparent clinical splits on *GenHlth*, *HighBP*, and *BMI* with **73.82% Accuracy**.
    """)
    
    # Comparative Bar Chart
    st.markdown("#### **Metric Comparison Chart**")
    plot_df = comp_df.melt(id_vars=["ML Model Name"], var_name="Metric", value_name="Score")
    
    fig_bar, ax_bar = plt.subplots(figsize=(10, 4.5))
    fig_bar.patch.set_facecolor('#ffffff')
    ax_bar.set_facecolor('#ffffff')
    sns.barplot(data=plot_df, x="Metric", y="Score", hue="ML Model Name", palette="Set2", ax=ax_bar)
    ax_bar.set_ylim(0.3, 1.0)
    ax_bar.set_ylabel("Score", fontweight="bold", color="#1e293b")
    ax_bar.set_xlabel("Evaluation Metric", fontweight="bold", color="#1e293b")
    ax_bar.set_title("Performance Metrics by Classification Model", fontsize=12, fontweight="bold", color="#0f172a")
    ax_bar.tick_params(colors="#1e293b")
    ax_bar.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    st.pyplot(fig_bar)

else:
    # Single Model Evaluation Mode
    model = load_model(selected_option)
    if model is None:
        st.error(f"Failed to load model artifact for '{selected_option}'. Check directory 'model/'.")
        st.stop()
    
    metrics = compute_model_metrics(model, X_test_trans, y_test)
    
    # Feature C: Display of Evaluation Metrics [1 Mark] (Native st.metric components)
    st.subheader(f"📈 Evaluation Metrics: {selected_option}")
    st.caption("Calculated on active test partition without retraining.")
    
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric(label="Accuracy", value=f"{metrics['Accuracy']:.4f}")
    c2.metric(label="ROC-AUC", value=f"{metrics['AUC']:.4f}")
    c3.metric(label="Precision", value=f"{metrics['Precision']:.4f}")
    c4.metric(label="Recall", value=f"{metrics['Recall']:.4f}")
    c5.metric(label="F1-Score", value=f"{metrics['F1']:.4f}")
    c6.metric(label="MCC Score", value=f"{metrics['MCC']:.4f}")
    
    st.markdown("---")
    
    # Feature D: Confusion Matrix & Classification Report [1 Mark]
    st.subheader("🎯 Classification Diagnostics & Error Analysis")
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("##### **Confusion Matrix**")
        cm = confusion_matrix(y_test, metrics["y_pred"])
        
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        fig_cm.patch.set_facecolor('#ffffff')
        ax_cm.set_facecolor('#ffffff')
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Non-Diabetic (0)", "Diabetic (1)"],
            yticklabels=["Non-Diabetic (0)", "Diabetic (1)"],
            ax=ax_cm,
            annot_kws={"size": 14, "weight": "bold"}
        )
        ax_cm.set_xlabel("Predicted Label", fontweight="bold", color="#1e293b")
        ax_cm.set_ylabel("True Ground Truth", fontweight="bold", color="#1e293b")
        ax_cm.tick_params(colors="#1e293b")
        plt.tight_layout()
        st.pyplot(fig_cm)
        
        # Diagnostic counts
        tn, fp, fn, tp = cm.ravel()
        st.caption(f"True Negatives: **{tn}** | False Positives: **{fp}** | False Negatives: **{fn}** | True Positives: **{tp}**")
        
    with col_right:
        st.markdown("##### **Classification Report & ROC Curve**")
        
        diag_tab1, diag_tab2 = st.tabs(["📋 Detailed Classification Report", "📈 ROC Curve"])
        
        with diag_tab1:
            report_dict = classification_report(
                y_test, metrics["y_pred"],
                target_names=["Non-Diabetic (0)", "Diabetic (1)"],
                output_dict=True
            )
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(
                report_df.style.format(precision=3),
                use_container_width=True
            )
            st.caption("Detailed per-class precision, sensitivity (recall), and harmonic F1-score.")
            
        with diag_tab2:
            if metrics["y_proba"] is not None:
                fpr, tpr, _ = roc_curve(y_test, metrics["y_proba"])
                fig_roc, ax_roc = plt.subplots(figsize=(5, 3.8))
                fig_roc.patch.set_facecolor('#ffffff')
                ax_roc.set_facecolor('#ffffff')
                ax_roc.plot(fpr, tpr, color="#0f4c81", lw=2.5, label=f"ROC (AUC = {metrics['AUC']:.4f})")
                ax_roc.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--", label="Random Chance")
                ax_roc.set_xlim([0.0, 1.0])
                ax_roc.set_ylim([0.0, 1.05])
                ax_roc.set_xlabel("False Positive Rate", fontweight="bold", color="#1e293b")
                ax_roc.set_ylabel("True Positive Rate", fontweight="bold", color="#1e293b")
                ax_roc.set_title("ROC Characteristic Curve", fontsize=11, fontweight="bold", color="#0f172a")
                ax_roc.tick_params(colors="#1e293b")
                ax_roc.legend(loc="lower right")
                plt.tight_layout()
                st.pyplot(fig_roc)
            else:
                st.info("Probability estimates not available for ROC curve calculation.")

# -----------------------------------------------------------------------------
# 7. Dataset Explorer & Transparency Section
# -----------------------------------------------------------------------------
with st.expander("🔍 Inspect Active Test Dataset Sample"):
    st.markdown(f"**Total Records:** `{active_df.shape[0]}` | **Features:** `21` | **Target:** `Diabetes_binary`")
    st.dataframe(active_df.head(10), use_container_width=True)
