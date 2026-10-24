import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

st.set_page_config(page_title="Data Explorer & Predictor", layout="wide")
st.title("Interactive Data Explorer & Predictor App")

st.header("1.Upload Dataset")
uploaded_file = st.file_uploader("Upload a CSV file with at least 3 numeric columns", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if len(numeric_cols) < 3:
        st.error("Error: The uploaded CSV must contain at least 3 numeric columns.")
    else:
        st.markdown("---")
    
        # Requirement 2b: Choose column and chart type (line, scatter, histogram)
        st.header("2. Interactive Data Visualization")
        col1, col2 = st.columns(2)
        
        with col1:
            chart_type = st.selectbox("Select Chart Type", ["Histogram", "Line", "Scatter"])
            
        with col2:
            if chart_type in ["Line", "Scatter"]:
                x_col = st.selectbox("Select X Axis Column", numeric_cols, index=0)
                y_col = st.selectbox("Select Y Axis Column", numeric_cols, index=min(1, len(numeric_cols) - 1))
            else:
                x_col = st.selectbox("Select Column for Histogram", numeric_cols, index=0)
        
        # Render Selected Chart
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
        
        # Requirement 2c: Select two inputs & target, fit linear regression, predict user values
        st.header("3. Linear Regression Model & Prediction")
        
        reg_col1, reg_col2 = st.columns(2)
        with reg_col1:
            feature_1 = st.selectbox("Select Feature 1 (Input 1)", numeric_cols, index=0)
            feature_2 = st.selectbox("Select Feature 2 (Input 2)", numeric_cols, index=min(1, len(numeric_cols) - 1))
        with reg_col2:
            target = st.selectbox("Select Target Variable (Output)", numeric_cols, index=min(2, len(numeric_cols) - 1))
            
        if feature_1 == feature_2:
            st.warning("Please select two distinct feature columns.")
        elif target in [feature_1, feature_2]:
            st.warning("The target column must be different from the input features.")
        else:
            # Drop rows with missing values in selected columns
            clean_df = df[[feature_1, feature_2, target]].dropna()
            X = clean_df[[feature_1, feature_2]]
            y = clean_df[target]
            
            # Train Linear Regression Model
            model = LinearRegression()
            model.fit(X, y)
            
            # Model Evaluation
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            
            st.success(f"Model Fitted! R² Score: {r2:.3f} | Root Mean Squared Error (RMSE): {rmse:.3f}")
            
            # User Inputs for Prediction
            st.subheader("Make a Custom Prediction")
            p_col1, p_col2 = st.columns(2)
            
            with p_col1:
                val1 = st.number_input(f"Enter value for {feature_1}", value=float(X[feature_1].mean()))
            with p_col2:
                val2 = st.number_input(f"Enter value for {feature_2}", value=float(X[feature_2].mean()))
                
            input_data = np.array([[val1, val2]])
            prediction = model.predict(input_data)[0]
            
            st.metric(label=f"Predicted {target}", value=f"{prediction:,.2f}")

        # Requirement 3: One sentence answering model purpose and consequences when wrong
        st.markdown("---")
        st.subheader("Model Usage & Risk Notice")
        st.info("This linear regression model is used to estimate numerical outcomes based on historical patterns in input features; if predictions are wrong, reliance on them may result in flawed decisions such as real estate mispricing or budget misallocation.")

else:
    st.info("Upload a CSV file using the box above to get started.")  
