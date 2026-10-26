import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

st.set_page_config(page_title="CSV Data Explorer & Predictor", layout="wide")

st.title("Data Explorer & Preliminary Predictor")

# 2a: Let the user upload any CSV file and display a preview
st.header("Step 1. Upload Dataset")
uploaded_file = st.file_uploader("Upload any CSV file (must contain at least 3 numeric columns)", type=["csv"])

if uploaded_file is not None:
 
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Data Preview")
    st.dataframe(df.head(10))
    
    # Automatically detect numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Added line highlighting detected numeric columns during preview
    st.info(f"**Detected Numeric Columns ({len(numeric_cols)}):** {', '.join(numeric_cols)}")
    
    # Validate requirement of at least 3 numeric columns
    if len(numeric_cols) < 3:
        st.error(f"The uploaded dataset contains only {len(numeric_cols)} numeric column(s). Please upload a file with at least 3 numeric columns.")
    else:
        st.markdown("---")
        
        # 2b: Choose column and chart type (line, scatter, histogram)
        st.header("2. Interactive Data Visualization")
        col1, col2 = st.columns(2)
        
        with col1:
            chart_type = st.selectbox("Select Chart Type", ["Scatter", "Histogram", "Line"])
            
        with col2:
            if chart_type in ["Line", "Scatter"]:
                x_col = st.selectbox("Select X Axis Column", numeric_cols, index=0)
                y_col = st.selectbox("Select Y Axis Column", numeric_cols, index=1)
            else:
                x_col = st.selectbox("Select Column for Histogram", numeric_cols, index=0)
        
        # Render chart dynamically using Plotly
        if chart_type == "Histogram":
            fig = px.histogram(df, x=x_col, title=f"Histogram of {x_col}")
            st.plotly_chart(fig, use_container_width=True)
        elif chart_type == "Line":
            fig = px.line(df, x=x_col, y=y_col, title=f"Line Chart: {y_col} vs {x_col}")
            st.plotly_chart(fig, use_container_width=True)
        elif chart_type == "Scatter":
            fig = px.scatter(df, x=x_col, y=y_col, title=f"Scatter Plot: {y_col} vs {x_col}")
            st.plotly_chart(fig, use_container_width=True)
            
        st.markdown("---")
        
        # 2c: Choose two inputs + target, fit linear regression, predict user values
        st.header("3. Linear Regression Model & Custom Prediction")
        
        reg_col1, reg_col2 = st.columns(2)
        with reg_col1:
            target = st.selectbox("Select Target Variable (Output)", numeric_cols, index=2)
        with reg_col2:
            feature_1 = st.selectbox("Select Feature 1 (Input 1)", numeric_cols, index=0)
            feature_2 = st.selectbox("Select Feature 2 (Input 2)", numeric_cols, index=1)
            
        # Feature validation checks
        if feature_1 == feature_2:
            st.warning("Please select two distinct input features.")
        elif target in [feature_1, feature_2]:
            st.warning("The target output variable must be different from the selected input features.")
        else:
            # Clean missing data for selected variables
            clean_df = df[[feature_1, feature_2, target]].dropna()
            
            if len(clean_df) < 2:
                st.error("Insufficient complete data rows to fit a linear regression model.")
            else:
                X = clean_df[[feature_1, feature_2]]
                y = clean_df[target]
                
                # Fit Linear Regression Model
                model = LinearRegression()
                model.fit(X, y)
                
                # Evaluate Model
                y_pred = model.predict(X)
                r2 = r2_score(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                
                st.success(f"Model Trained! R² Score: {r2:.3f} | Root Mean Squared Error (RMSE): {rmse:.3f}")
                
                # Predict user values
                st.subheader("Type Input Values to Generate Prediction")
                p_col1, p_col2 = st.columns(2)
                
                with p_col1:
                    val1 = st.number_input(f"Enter value for '{feature_1}'", value=float(X[feature_1].median()))
                    val2 = st.number_input(f"Enter value for '{feature_2}'", value=float(X[feature_2].median()))
                 
                input_df = pd.DataFrame([[val1, val2]], columns=[feature_1, feature_2])
                prediction = model.predict(input_df)[0]
             
                with p_col2:
                    st.metric(label=f"Predicted {target}", value=f"{prediction:,.2f}")

        # 3: One sentence answering purpose and consequence of model error
        st.markdown("---")
        st.subheader("Model Usage & Risk Notice")
        st.info("This linear regression model is used to estimate unknown numerical outcomes based on mathematical relationships found between input features; if predictions are wrong, relying on them can lead to flawed decision-making, poor strategy, and unmitigated financial risk.")

else:
    st.info("Please upload a CSV file with at least 3 numeric columns to get started.")
