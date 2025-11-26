import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import wandb
import os
import streamlit.components.v1 as components # Required for SHAP Force Plot

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import plot_tree, DecisionTreeRegressor # Added DecisionTreeRegressor for Viz

# --- 1. PAGE CONFIG ---
st.set_page_config(
    layout="wide",
    page_title="BMW Price Analytics",
    page_icon="🚘"
)

# Set Seaborn style for better charts
sns.set_theme(style="whitegrid")

#Helper function to render SHAP JS in Streamlit
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# --- 2. DATA LOADING & PROCESSING (Cached) ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('bmw.csv')
        return df
    except FileNotFoundError:
        return None

@st.cache_resource
def build_pipeline(df):
    # Separate features and target
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Identify columns
    categorical_cols = ['model', 'transmission', 'fuelType']
    numerical_cols = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
    
    # Build Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        verbose_feature_names_out=False
    )
    
    # Transform data
    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names
    try:
        feature_names = preprocessor.get_feature_names_out()
    except:
        feature_names = [f"feat_{i}" for i in range(X_processed.shape[1])]
        
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X_processed_df, y, test_size=0.2, random_state=42)
    
    # Train Models
    models = {
        'Linear Regression': LinearRegression().fit(X_train, y_train),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42).fit(X_train, y_train),
        'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42).fit(X_train, y_train)
    }

    return models, X_train, X_test, y_train, y_test, preprocessor, feature_names

# Load Data
df_raw = load_data()

if df_raw is None:
    st.error("🚨 Error: 'bmw.csv' not found. Please upload it.")
    st.stop()

# Initialize Pipeline
with st.spinner("⚙️ Training Models & Preparing Data..."):
    models, X_train, X_test, y_train, y_test, preprocessor, feature_names = build_pipeline(df_raw)


# --- 3. SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation 🧭")
page = st.sidebar.radio("Go to", 
    ["🏠 Business Case & Data", 
     "📊 Data Visualizations", 
     "🤖 Prediction & Evaluation",
     "🔍 AI Explainability",
     "🎛️ Hyperparameter Tuning"]
)
st.sidebar.markdown("---")
st.sidebar.caption("DS Final Project 2025")


# --- 4. PAGE 1: BUSINESS CASE & DATA ---
if page == "🏠 Business Case & Data":
    st.title("BMW Valuation System 🚘")
    
    # Image Handling
    if os.path.exists("img.jpg"):
        st.image("img.jpg", caption="The Ultimate Driving Machine", width=700)
    else:
        st.warning("⚠️ 'img.jpg' not found. Please upload an image.")

    st.markdown("---")
    st.header("The Business Problem")
    st.write("""
    Used car dealerships and private sellers face a critical challenge: **Pricing Strategy**.
    
    * **Price too high:** The car sits in the lot, depreciating and costing maintenance fees.
    * **Price too low:** The seller loses potential profit immediately.
    
    **Our Solution:** A machine learning application that utilizes **Linear Regression**, **Random Forest**, and **XGBoost** to predict the *fair market value* of a BMW based on its year, mileage, and condition.
    """)

    with st.expander("Click to see Data & Methodology"):
        st.subheader("The Dataset")
        st.write(f"We are using a dataset of **{len(df_raw):,} BMW vehicles**.")
        st.dataframe(df_raw.head())
        st.markdown("Source: [Kaggle - BMW Used Car Dataset](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes)")
        
        st.subheader("Methodology: How we handle the data")
        st.markdown("##### Step 1: Categorical Encoding")
        st.write("Columns like `model` (e.g., '3 Series') and `fuelType` were converted into numbers using **One-Hot Encoding**.")
        
        st.markdown("##### Step 2: Scaling")
        st.write("Features like `mileage` (0-200k) and `year` (1996-2020) were normalized using **StandardScaler** to help the models learn faster.")
    
    st.subheader("Quick Data Stats")
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Price", f"£{df_raw['price'].mean():,.0f}")
    c2.metric("Avg Mileage", f"{df_raw['mileage'].mean():,.0f}")
    c3.metric("Avg Year", int(df_raw['year'].mean()))


# --- 5. PAGE 2: DATA VISUALIZATIONS ---
elif page == "📊 Data Visualizations":
    st.title("Visual Insights: What Drives Price? 📈")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs([
        "📉 Mileage vs. Price", 
        "🚘 Price by Model", 
        "🔥 Correlation Heatmap"
    ])
    
    with tab1:
        st.header("The 'Depreciation Curve'")
        
        st.info("""
        **Key Insight:** This chart confirms that **mileage is the strongest negative driver of price**, with a sharp drop in value occurring within the first 40,000 miles. 
        However, **newer models (lighter dots) successfully resist this trend**, maintaining significantly higher residual values even at higher mileage points compared to older equivalents.
        """)
        
        # Optimization: Sample data so the chart renders instantly
        plot_data = df_raw.sample(n=min(2000, len(df_raw)), random_state=42)
        
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=plot_data, x='mileage', y='price', hue='year', palette='viridis', alpha=0.6, ax=ax1)
        ax1.set_title("Price vs. Mileage (Sampled for Performance)")
        ax1.set_xlabel("Mileage")
        ax1.set_ylabel("Price (£)")
        st.pyplot(fig1)

    with tab2:
        st.header("Which Series is most expensive?")
        
        st.info("""
        **Key Insight:** The **X7 and 8 Series command the highest median prices**, reflecting the premium positioning of BMW's luxury SUV and Tourer segments.
        In contrast, the **1 Series and 3 Series offer the most affordable entry points**, though their wide price ranges indicate that condition and age vary heavily in these popular models.
        """)
        
        # Calculate median price to sort the chart
        order = df_raw.groupby('model')['price'].median().sort_values(ascending=False).index
        
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        sns.boxplot(data=df_raw, x='price', y='model', order=order, palette='coolwarm', ax=ax2)
        ax2.set_title("Price Distribution by BMW Model")
        ax2.set_xlabel("Price (£)")
        ax2.set_ylabel("Model")
        st.pyplot(fig2)

    with tab3:
        st.header("Feature Correlation Heatmap")
        
        st.info("""
        **Key Insight:** **Year and Engine Size show the strongest positive correlations**, meaning that newer and more powerful cars are consistently associated with higher price tags.
        Conversely, **Mileage displays a strong negative correlation**, providing mathematical confirmation of the depreciation trends observed in our scatter plots.
        """)
        
        numeric_df = df_raw.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax3)
        st.pyplot(fig3)


# --- 6. PAGE 3: PREDICTION & EVALUATION ---
elif page == "🤖 Prediction & Evaluation":
    st.title("Model Performance & Comparison 🔬")
    
    # --- 1. EVALUATION METRICS (TOP) ---
    st.write("We evaluated 3 different models. Here is how they compare on unseen data:")

    # Calculate metrics for all 3
    pred_xgb_test = models['XGBoost'].predict(X_test)
    pred_lr_test = models['Linear Regression'].predict(X_test)
    pred_rf_test = models['Random Forest'].predict(X_test)
    
    # Display Big Metrics
    m1, m2, m3 = st.columns(3)
    
    with m1:
        st.info("Model 1: Linear Regression")
        st.metric("R² Score", f"{r2_score(y_test, pred_lr_test):.3f}")
        st.metric("MAE Error", f"£{mean_absolute_error(y_test, pred_lr_test):,.0f}")
        
    with m2:
        st.success("Model 2: XGBoost (Winner)")
        st.metric("R² Score", f"{r2_score(y_test, pred_xgb_test):.3f}")
        st.metric("MAE Error", f"£{mean_absolute_error(y_test, pred_xgb_test):,.0f}")
        
    with m3:
        st.warning("Model 3: Random Forest")
        st.metric("R² Score", f"{r2_score(y_test, pred_rf_test):.3f}")
        st.metric("MAE Error", f"£{mean_absolute_error(y_test, pred_rf_test):,.0f}")

    st.markdown("---")
    
    # --- 2. ACTUAL VS PREDICTED GRAPHS ---
    st.header("Visualizing Accuracy: Actual vs. Predicted")
    
    st.info("""
    **Key Insight:** **XGBoost and Random Forest demonstrate the tightest clustering** of points along the red diagonal line, indicating they capture both the low-end and high-end market values with high precision.
    **Linear Regression struggles significantly with high-value outliers** (the points scattered far below the line), failing to capture the non-linear premium attached to luxury models like the i8 or X7.
    """)
    
    # TABS including the NEW Combined Tab
    g0, g1, g2, g3 = st.tabs(["All Models Comparison", "Linear Regression", "XGBoost", "Random Forest"])
    
    # Helper to plot - OPTIMIZED WITH SAMPLING
    def plot_actual_vs_pred(y_true, y_pred, name, color):
        temp_df = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
        if len(temp_df) > 1000:
            temp_df = temp_df.sample(1000, random_state=42)
            
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=temp_df['Actual'], y=temp_df['Predicted'], alpha=0.3, color=color, ax=ax)
        
        min_val = min(temp_df['Actual'].min(), temp_df['Predicted'].min())
        max_val = max(temp_df['Actual'].max(), temp_df['Predicted'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
        
        ax.set_title(f"{name}: Actual vs Predicted")
        ax.set_xlabel("Actual Price (£)")
        ax.set_ylabel("Predicted Price (£)")
        return fig

    # --- NEW: COMBINED GRAPH ---
    with g0:
        st.write("Comparing all models on the same data points.")
        fig_comb, ax_comb = plt.subplots(figsize=(10, 7))
        
        # Sample data once for fair comparison
        indices = np.random.choice(len(y_test), min(500, len(y_test)), replace=False)
        y_true_samp = y_test.iloc[indices]
        
        # Plot 3 models with distinct colors
        sns.scatterplot(x=y_true_samp, y=pred_lr_test[indices], color='red', alpha=0.4, label='Linear Regression', ax=ax_comb)
        sns.scatterplot(x=y_true_samp, y=pred_rf_test[indices], color='orange', alpha=0.4, label='Random Forest', ax=ax_comb)
        sns.scatterplot(x=y_true_samp, y=pred_xgb_test[indices], color='green', alpha=0.4, label='XGBoost', ax=ax_comb)
        
        # Diagonal Line
        min_v = y_true_samp.min()
        max_v = y_true_samp.max()
        ax_comb.plot([min_v, max_v], [min_v, max_v], 'k--', lw=2, label='Perfect Prediction')
        
        ax_comb.set_title("All Models: Actual vs Predicted")
        ax_comb.set_xlabel("Actual Price (£)")
        ax_comb.set_ylabel("Predicted Price (£)")
        ax_comb.legend()
        st.pyplot(fig_comb)

    with g1:
        st.pyplot(plot_actual_vs_pred(y_test, pred_lr_test, "Linear Regression", "red"))
    with g2:
        st.pyplot(plot_actual_vs_pred(y_test, pred_xgb_test, "XGBoost", "green"))
    with g3:
        st.pyplot(plot_actual_vs_pred(y_test, pred_rf_test, "Random Forest", "orange"))

    st.markdown("---")

    # --- 3. PREDICTION TOOL ---
    st.title("Price Calculator Tool 🧮")
    st.write("Configure a car below to see what each model thinks it's worth.")
    
    with st.expander("⚙️ Configure Vehicle", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            # FIX: Default index=0 for '1 Series' (usually first alphabetically)
            p_model = st.selectbox("Model", sorted(df_raw['model'].unique()), index=0)
            p_year = st.slider("Year", 2000, 2024, 2022)
        with col2:
            p_trans = st.selectbox("Transmission", df_raw['transmission'].unique())
            p_fuel = st.selectbox("Fuel", df_raw['fuelType'].unique())
        with col3:
            p_miles = st.number_input("Mileage", 0, 250000, 30000, step=1000)
            p_engine = st.number_input("Engine (L)", 0.0, 6.0, 2.0, 0.1)
        with col4:
            p_mpg = st.number_input("MPG", 0.0, 200.0, 50.0)
            p_tax = st.number_input("Tax (£)", 0, 1000, 145)

        run_pred = st.button("Predict Price (Run All Models) 🚀", type="primary")

    if run_pred:
        # Prepare Input
        input_data = pd.DataFrame({
            'model': [p_model], 'year': [p_year], 'transmission': [p_trans],
            'mileage': [p_miles], 'fuelType': [p_fuel], 'tax': [p_tax],
            'mpg': [p_mpg], 'engineSize': [p_engine]
        })
        
        # Transform
        input_proc = preprocessor.transform(input_data)
        input_df = pd.DataFrame(input_proc, columns=feature_names)
        
        # Predict
        pred_lr = models['Linear Regression'].predict(input_df)[0]
        pred_xgb = models['XGBoost'].predict(input_df)[0]
        pred_rf = models['Random Forest'].predict(input_df)[0]
        
        st.divider()
        st.subheader("Prediction Results")
        
        r1, r2, r3 = st.columns(3)
        with r1:
            st.info("Linear Regression")
            st.metric("Price Estimate", f"£{pred_lr:,.0f}")
        with r2:
            st.success("XGBoost")
            st.metric("Price Estimate", f"£{pred_xgb:,.0f}")
        with r3:
            st.warning("Random Forest")
            st.metric("Price Estimate", f"£{pred_rf:,.0f}")


# --- 7. PAGE 4: EXPLAINABILITY ---
elif page == "🔍 AI Explainability":
    st.title("🤖 Why did the model predict that?")
    st.info("Using SHAP (SHapley Additive exPlanations) to understand feature drivers.")

    # FIX: Robustly Initialize session state for SHAP
    if 'shap_values' not in st.session_state:
        st.session_state.shap_values = None
    if 'X_display' not in st.session_state:
        st.session_state.X_display = None
    if 'model_choice' not in st.session_state:
        st.session_state.model_choice = None

    model_choice = st.selectbox("Choose Model", ["XGBoost", "Random Forest", "Linear Regression"])
    
    # Store user choice in session state to detect changes
    if st.button("Generate Explanation"):
        with st.spinner("Calculating SHAP values (Optimized)..."):
            model = models[model_choice]
            X_sample = X_test.iloc[:100].copy() 
            
            # Inverse Transform for Display
            scaler = preprocessor.named_transformers_['num']
            num_cols = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
            
            X_display = X_sample.copy()
            X_display[num_cols] = scaler.inverse_transform(X_sample[num_cols])

            # Calculate SHAP
            if model_choice == "Linear Regression":
                explainer = shap.LinearExplainer(model, X_train.iloc[:100])
                shap_values = explainer.shap_values(X_sample)
            else:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
            
            # SAVE TO SESSION STATE
            st.session_state.shap_values = shap_values
            st.session_state.X_display = X_display
            st.session_state.model_choice = model_choice

    # CHECK IF DATA EXISTS IN SESSION STATE
    if st.session_state.shap_values is not None:
        
        # Retrieve from state
        shap_values = st.session_state.shap_values
        X_display = st.session_state.X_display
        current_model = st.session_state.model_choice

        # PLOT 1: Feature Importance
        st.subheader(f"1. Global Feature Importance ({current_model})")
        st.write("Which features change the price the most on average?")
        
        fig_shap1 = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_display, plot_type="bar", show=False)
        st.pyplot(fig_shap1)
        
        # PLOT 2: Dependence Plot
        st.subheader("2. Deep Dive: Feature Dependence")
        feature_options = ["mileage", "year", "engineSize", "mpg", "tax"]
        selected_feature = st.selectbox("Select feature to analyze:", feature_options, index=0)
        
        feat_col = [c for c in X_display.columns if selected_feature in c]
        if feat_col:
            fig_shap2, ax = plt.subplots(figsize=(10, 6))
            shap.dependence_plot(feat_col[0], shap_values, X_display, ax=ax, show=False)
            st.pyplot(fig_shap2)
        else:
            st.warning(f"Could not find {selected_feature} in dataset.")

        # PLOT 3: Tree Visualization (Fixed to use RAW values)
        st.markdown("---")
        st.subheader("3. Random Forest: Under the Hood")
        
        if current_model == "Random Forest":
            st.write("Comparing **SHAP (Contribution)** vs **Tree Structure (Rules)**.")
            st.write("Below is **one single tree** trained on raw data to demonstrate clear logic (e.g., Year <= 2017).")
            
            # --- CREATE A VIZ-ONLY TREE WITH RAW DATA ---
            # 1. Prepare Raw Data
            X_viz = df_raw.drop('price', axis=1)
            y_viz = df_raw['price']
            
            # 2. Simple Ordinal Encoding for text columns (so tree handles them as 1, 2, 3...)
            cat_cols = ['model', 'transmission', 'fuelType']
            oe = OrdinalEncoder()
            X_viz[cat_cols] = oe.fit_transform(X_viz[cat_cols])
            
            # 3. Train a small tree just for this picture
            viz_tree = DecisionTreeRegressor(max_depth=3, random_state=42)
            viz_tree.fit(X_viz, y_viz)
            
            # 4. Plot
            fig_tree, ax_tree = plt.subplots(figsize=(20, 10))
            plot_tree(viz_tree, 
                      feature_names=X_viz.columns, 
                      filled=True, 
                      rounded=True,
                      impurity=False, # Hide impurity (squared_error)
                      fontsize=10, 
                      ax=ax_tree)
            st.pyplot(fig_tree)
        else:
            st.info("Select 'Random Forest' in the dropdown above to see the Decision Tree visualization.")


# --- 8. PAGE 5: TUNING ---
elif page == "🎛️ Hyperparameter Tuning":
    st.title("Hyperparameter Optimization")
    st.write("Tracking model performance with **Weights & Biases**.")

    st.markdown("""
    **What is this?** We are fine-tuning the **XGBoost** model to find the perfect balance of speed and accuracy.
    * **n_estimators (Trees):** How many "decision trees" the model uses. More is usually better but slower.
    * **learning_rate:** How fast the model learns. If too fast, it misses details; if too slow, it takes forever.
    """)
    
    wb_api = st.text_input("W&B API Key (Optional)", type="password")
    
    if st.button("Start Grid Search Simulation"):
        st.info("Running Grid Search on: n_estimators (Trees) vs learning_rate...")
        
        # --- NEW: Explicit Login ---
        if wb_api:
            try:
                wandb.login(key=wb_api)
                st.success("Logged into Weights & Biases!")
            except:
                st.error("Invalid W&B Key. Running offline.")

        # Run a "Real" mini grid search
        n_estimators_list = [50, 100, 200]
        learning_rates = [0.01, 0.1, 0.3]
        
        results = []
        
        prog_bar = st.progress(0)
        idx = 0
        total_runs = len(n_estimators_list) * len(learning_rates)
        
        for n_est in n_estimators_list:
            for lr in learning_rates:
                idx += 1
                prog_bar.progress(idx / total_runs)
                
                model = XGBRegressor(n_estimators=n_est, learning_rate=lr, n_jobs=-1, random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                results.append({
                    'n_estimators': n_est,
                    'learning_rate': lr,
                    'RMSE': rmse
                })
                
                # --- NEW: Log to W&B ---
                if wb_api:
                    run = wandb.init(project="bmw-price-opt", config={'n_estimators': n_est, 'lr': lr}, reinit=True)
                    wandb.log({'RMSE': rmse})
                    run.finish()
        
        res_df = pd.DataFrame(results)
        
        st.success("Optimization Complete!")
        
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.write("### Results Table")
            st.dataframe(res_df.style.highlight_min(subset=['RMSE'], color='lightgreen'))
            
        with c2:
            st.write("### Optimization Heatmap")
            st.write("Darker colors (Purple/Blue) = Lower Error (Better).")
            
            pivot_table = res_df.pivot(index="learning_rate", columns="n_estimators", values="RMSE")
            
            fig_heat, ax = plt.subplots(figsize=(8, 6))
            # FIX: Used 'viridis' (Default) so Low Values (Good) are Dark Purple/Blue, High Values (Bad) are Yellow
            sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="viridis", ax=ax)
            ax.set_title("RMSE Error (Lower is Better)")
            st.pyplot(fig_heat)

        st.markdown("---")
        st.subheader("💡 Tuning Insights")
        st.info("""
        **What does this tell us?**
        1.  **The Goldilocks Zone:** The darkest areas (Purple/Blue) on the heatmap show where the model performs best (Lowest Error).
        2.  **Interaction Effect:** Lower learning rates (0.01) usually need **more trees** (200+) to reach peak performance.
        3.  **Optimal Config:** Look for the darkest cell – that represents the lowest error in our experiment.
        """)