import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="BMW Price Analytics", page_icon="🚘", layout="wide")

# --- 2. LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv('bmw.csv')
    return df

try:
    df = load_data()
except:
    st.error("File bmw.csv not found. Please upload it to the project folder.")
    st.stop()

# --- 3. MODEL PIPELINE BUILDER ---
def get_model_pipeline(model_name, xgb_params=None):
    # Define features types
    categorical_cols = ['model', 'transmission', 'fuelType']
    numerical_cols = ['year', 'mileage', 'tax', 'mpg', 'engineSize']

    # Create the preprocessor
    # This handles all the "Feature Engineering" automatically:
    # 1. Scaling numerical numbers (so mileage doesn't dominate year)
    # 2. One-Hot Encoding categorical text (converting 'Manual' to 0/1)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    # Select the model
    if model_name == 'XGBoost':
        if xgb_params:
            model = XGBRegressor(**xgb_params, random_state=42)
        else:
            model = XGBRegressor(random_state=42)
    else:
        model = LinearRegression()

    # Bundle preprocessor and model together
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    return pipeline

# --- 4. SIDEBAR NAVIGATION ---
st.sidebar.title("BMW Analytics Suite 🧭")
page = st.sidebar.radio("Navigate", 
    ["1. Business & Data", 
     "2. Insights & EDA", 
     "3. Price Prediction", 
     "4. Feature Importance", 
     "5. Tuning Dashboard"])

# --- 5. PAGE 1: BUSINESS & DATA ---
if page == "1. Business & Data":
    st.title("🚘 BMW Valuation System")
    
    # --- Image Loading ---
    try:
        st.image("bmw_car.jpg", width=700)
    except:
        st.warning("Image 'bmw_car.jpg' not found. Please upload it for a better look!")

    st.markdown("---")
    
    # --- Business Case ---
    st.header("The Business Problem")
    st.write("""
    Used car dealerships and private sellers face a critical challenge: **Pricing Strategy**.
    
    * **Price too high:** The car sits in the lot for months, losing value and costing money in maintenance.
    * **Price too low:** The seller loses potential profit immediately.
    
    **Our Goal:** Build a Machine Learning tool that analyzes thousands of BMW sales to predict the *exact* fair market value of any car based on its specs.
    """)

    # --- Data Presentation ---
    with st.expander("Click to see Data & Methodology"):
        st.subheader("The Dataset")
        st.write(f"We are using a dataset of **{len(df):,} BMW vehicles**. Here is a glimpse:")
        st.dataframe(df.head())
        
        st.subheader("Methodology: How we handle the data")
        st.write("""
        Before feeding data into our models, we use a **Pipeline** to transform it:
        
        1.  **Categorical Data:** Columns like `model` (e.g., "3 Series") and `fuelType` (e.g., "Diesel") are text. We convert them into binary numbers using **One-Hot Encoding**.
        2.  **Numerical Data:** Features like `mileage` (0-200,000) and `year` (1996-2020) have vastly different ranges. We use **Standard Scaling** to normalize them.
        3.  **Modeling:** We compare a baseline **Linear Regression** against a powerful **XGBoost** model to see which best captures the complex price dynamics.
        """)

    # --- Key Stats Metrics ---
    st.subheader("Quick Data Stats")
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Price", f"£{df['price'].mean():,.0f}")
    c2.metric("Avg Mileage", f"{df['mileage'].mean():,.0f}")
    c3.metric("Avg Year", int(df['year'].mean()))

# --- 6. PAGE 2: INSIGHTS & EDA ---
elif page == "2. Insights & EDA":
    st.title("📊 Market Analysis")
    st.write("Understanding the factors that drive price before modeling.")
    
    tab1, tab2, tab3 = st.tabs(["Mileage Impact", "Categorical Analysis", "Correlations"])
    
    with tab1:
        st.subheader("The 'Depreciation Curve'")
        st.write("This chart shows the strongest relationship in our data: as mileage goes up, price goes down. The curve flattens out for older cars.")
        fig = px.scatter(df, x='mileage', y='price', color='year', title="Price vs. Mileage (Colored by Year)")
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        st.subheader("Impact of Transmission & Fuel")
        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.box(df, x='transmission', y='price', title="Price by Transmission")
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            fig2 = px.box(df, x='fuelType', y='price', title="Price by Fuel Type")
            st.plotly_chart(fig2, use_container_width=True)
            
    with tab3:
        st.subheader("Correlation Heatmap")
        st.write("Which numerical features are most linked to Price?")
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        fig_corr, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig_corr)

# --- 7. PAGE 3: PREDICTION ---
elif page == "3. Price Prediction":
    st.title("🧪 Prediction Lab")
    st.write("Compare our two models and generate live price estimates.")
    
    # Prepare Data
    X = df.drop('price', axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Selection
    model_choice = st.selectbox("Select Model", ["XGBoost (Best Accuracy)", "Linear Regression (Baseline)"])
    
    if "XGBoost" in model_choice:
        pipeline = get_model_pipeline('XGBoost')
    else:
        pipeline = get_model_pipeline('Linear Regression')
        
    # Train and Evaluate
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Show Metrics
    st.subheader("Model Performance (Test Set)")
    c1, c2 = st.columns(2)
    c1.metric("R² Score", f"{r2:.3f}", help="How much of the price variance is explained by the model?")
    c2.metric("MAE Error", f"£{mae:,.0f}", help="Average error in pounds.")
    
    st.markdown("---")
    
    # User Inputs for Prediction
    st.subheader("Live Estimator: Value a Car")
    
    xc1, xc2, xc3, xc4 = st.columns(4)
    in_model = xc1.selectbox("Model", df['model'].unique())
    in_year = xc2.number_input("Year", 2000, 2025, 2019)
    in_trans = xc3.selectbox("Transmission", df['transmission'].unique())
    in_fuel = xc4.selectbox("Fuel", df['fuelType'].unique())
    
    xc5, xc6, xc7, xc8 = st.columns(4)
    in_miles = xc5.number_input("Mileage", 0, 200000, 30000)
    in_engine = xc6.number_input("Engine (L)", 0.0, 6.0, 2.0, 0.1)
    in_tax = xc7.number_input("Tax", 0, 500, 145)
    in_mpg = xc8.number_input("MPG", 0.0, 100.0, 50.0)
    
    if st.button("Estimate Price"):
        input_data = pd.DataFrame({
            'model': [in_model], 'year': [in_year], 'transmission': [in_trans],
            'mileage': [in_miles], 'fuelType': [in_fuel], 'tax': [in_tax],
            'mpg': [in_mpg], 'engineSize': [in_engine]
        })
        
        pred = pipeline.predict(input_data)[0]
        st.success(f"Estimated Market Value: **£{pred:,.2f}**")

# --- 8. PAGE 4: FEATURE IMPORTANCE ---
elif page == "4. Feature Importance":
    st.title("🔍 Driving Factors")
    st.write("What matters more? The year, the mileage, or the engine size?")
    
    X = df.drop('price', axis=1)
    y = df['price']
    
    col_choice = st.selectbox("Analyze Model", ["XGBoost", "Linear Regression"])
    
    if col_choice == "XGBoost":
        pipeline = get_model_pipeline('XGBoost')
    else:
        pipeline = get_model_pipeline('Linear Regression')
        
    pipeline.fit(X, y)
    
    # Extract feature names from preprocessor
    feature_names = (pipeline.named_steps['preprocessor']
                     .transformers_[1][1]
                     .get_feature_names_out(input_features=['model', 'transmission', 'fuelType']))
    all_features = ['year', 'mileage', 'tax', 'mpg', 'engineSize'] + list(feature_names)
    
    # Get importances
    if col_choice == "Linear Regression":
        importances = pipeline.named_steps['model'].coef_
    else:
        importances = pipeline.named_steps['model'].feature_importances_
        
    feat_df = pd.DataFrame({'Feature': all_features, 'Importance': importances})
    feat_df['Abs_Importance'] = feat_df['Importance'].abs()
    feat_df = feat_df.sort_values(by='Abs_Importance', ascending=False).head(15)
    
    fig = px.bar(feat_df, x='Importance', y='Feature', orientation='h', 
                 title=f"Top 15 Features Impacting Price ({col_choice})")
    st.plotly_chart(fig, use_container_width=True)

# --- 9. PAGE 5: TUNING DASHBOARD ---
elif page == "5. Tuning Dashboard":
    st.title("🎛️ W&B Simulation: XGBoost Tuning")
    st.write("Tracking experiments to optimize the **Number of Trees (n_estimators)** in XGBoost.")
    
    X = df.drop('price', axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    n_estimators_list = [50, 100, 200, 300, 500]
    results = []
    
    bar = st.progress(0)
    for i, n_est in enumerate(n_estimators_list):
        # Train model with specific hyperparameter
        pipeline = get_model_pipeline('XGBoost', xgb_params={'n_estimators': n_est, 'learning_rate': 0.1})
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results.append({'n_estimators': n_est, 'RMSE': rmse, 'R2': r2})
        bar.progress((i+1)/len(n_estimators_list))
        
    res_df = pd.DataFrame(results)
    
    # Visualization similar to Weights & Biases
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Parameter Space")
        fig_p = px.parallel_coordinates(res_df, dimensions=['n_estimators', 'RMSE', 'R2'], 
                                        color='RMSE', color_continuous_scale=px.colors.sequential.Bluered)
        st.plotly_chart(fig_p, use_container_width=True)
        
    with c2:
        st.subheader("Optimization Curve")
        st.write("We look for the point where RMSE (Error) is lowest.")
        fig_l = px.line(res_df, x='n_estimators', y='RMSE', markers=True)
        st.plotly_chart(fig_l, use_container_width=True)
        
    best = res_df.loc[res_df['RMSE'].idxmin()]
    st.success(f"✅ **Best Result:** {int(best['n_estimators'])} Trees achieved the lowest error (£{best['RMSE']:.2f})")