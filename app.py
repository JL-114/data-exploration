import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px

st.set_page_config(page_title="CSV Data Explorer & Predictor", layout="wide")

st.title("Data Explorer & Preliminary Predictor")

# Helper function to auto-clean currency/formatted columns into numeric floats
def preprocess_dataframe(df_input: pd.DataFrame) -> pd.DataFrame:
    df_clean = df_input.copy()
    for col in df_clean.select_dtypes(include=['object', 'string']).columns:
        s_raw = df_clean[col].dropna().astype(str).str.strip()
        if s_raw.empty:
            continue
            
        # Check for accounting negative format: ($1,234) -> -1234
        is_paren_neg = s_raw.str.startswith('(') & s_raw.str.endswith(')')
        
        # Strip non-numeric characters except digits, dots, and minus signs
        s_clean = s_raw.str.replace(r'[^\d.-]', '', regex=True)
        numeric_test = pd.to_numeric(s_clean, errors='coerce')
        
        # Convert column if >= 70% of non-null strings become valid numbers
        if len(s_raw) > 0 and (numeric_test.notnull().sum() / len(s_raw)) >= 0.7:
            s_full = df_clean[col].astype(str).str.strip()
            full_paren_neg = s_full.str.startswith('(') & s_full.str.endswith(')')
            full_clean = s_full.str.replace(r'[^\d.-]', '', regex=True)
            
            converted_series = pd.to_numeric(full_clean, errors='coerce')
            converted_series[full_paren_neg] = -converted_series[full_paren_neg].abs()
            
            df_clean[col] = converted_series

    return df_clean

# 2a: Upload Dataset
st.header("Upload Dataset")
uploaded_file = st.file_uploader("Upload any CSV file (must contain at least 3 numeric columns)", type=["csv"])

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    
    # Clean currency and formatted text columns immediately upon upload
    df = preprocess_dataframe(raw_df)
    
    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Automatically detect all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Highlight detected numeric columns during preview
    st.info(f"**Detected Numeric Columns ({len(numeric_cols)}):** {', '.join(numeric_cols)}")
    
    # Validate requirement of at least 3 numeric columns
    if len(numeric_cols) < 3:
        st.error(f"The uploaded dataset contains only {len(numeric_cols)} numeric column(s). Please upload a file with at least 3 numeric columns.")
    else:
        st.markdown("---")
        
        # Section 2: Interactive Data Visualization
        st.header("Interactive Data Visualization")
        col1, col2 = st.columns(2)
        
        with col1:
            chart_type = st.selectbox("Select Chart Type", ["Scatter", "Histogram", "Line"])
            
        with col2:
            if chart_type in ["Line", "Scatter"]:
                x_col = st.selectbox("Select X Axis Column", numeric_cols, index=0)
                y_col = st.selectbox("Select Y Axis Column", numeric_cols, index=1)
            else:
                x_col = st.selectbox("Select Column for Histogram", numeric_cols, index=0)
        
        # Render chart dynamically using Plotly with 650px height
        if chart_type == "Histogram":
            fig = px.histogram(df, x=x_col, title=f"Histogram of {x_col}")
        elif chart_type == "Line":
            fig = px.line(df, x=x_col, y=y_col, title=f"Line Chart: {y_col} vs {x_col}")
        elif chart_type == "Scatter":
            fig = px.scatter(df, x=x_col, y=y_col, title=f"Scatter Plot: {y_col} vs {x_col}")
            
        fig.update_layout(height=650)
        st.plotly_chart(fig, use_container_width=True)
            
        st.markdown("---")

        # Section 3: Linear Regression Model & Custom Prediction
        st.header("Linear Regression Model & Custom Prediction")
        
        def get_input_params(series: pd.Series):
            clean_s = series.dropna()
            if clean_s.empty:
                return 0.1, "%.2f", None, None
            
            unique_vals = set(clean_s.unique())
            is_binary = unique_vals.issubset({0, 1, 0.0, 1.0})
            is_int = np.issubdtype(series.dtype, np.integer) or (clean_s % 1 == 0).all()
            
            if is_binary:
                return 1.0, "%.0f", 0.0, 1.0
            elif is_int:
                return 1.0, "%.0f", None, None
            else:
                return 0.01, "%.2f", None, None

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
            
            if len(clean_df) <= 3:
                st.error("Insufficient complete data rows to fit a linear regression model.")
            else:
                X = clean_df[[feature_1, feature_2]]
                y = clean_df[target]
                
                # Fit Linear Regression Model for prediction
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                
                # Calculate evaluation metrics using scikit-learn & numpy
                r2 = r2_score(y, y_pred)
                n = len(y)
                k = X.shape[1]
                adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
                
                mae = mean_absolute_error(y, y_pred)
                mse = mean_squared_error(y, y_pred)
                rmse = np.sqrt(mse)

                # Display 5 evaluation metrics in columns
                st.subheader("Model Evaluation Metrics")
                m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
                m_col1.metric("R² Score", f"{r2:.3f}")
                m_col2.metric("Adjusted R²", f"{adj_r2:.3f}")
                m_col3.metric("MAE", f"{mae:,.3f}")
                m_col4.metric("MSE", f"{mse:,.3f}")
                m_col5.metric("RMSE", f"{rmse:,.3f}")
                
                # Predict user values with dynamic step, bounds, and formatting per column
                st.subheader("Type Input Values to Generate Prediction")
                p_col1, p_col2 = st.columns(2)
                 
                with p_col2:
                    step1, fmt1, min1, max1 = get_input_params(X[feature_1])
                    step2, fmt2, min2, max2 = get_input_params(X[feature_2])
                    
                    val1 = st.number_input(
                        f"Enter value for '{feature_1}'", 
                        value=float(X[feature_1].median()), 
                        min_value=min1,
                        max_value=max1,
                        step=step1, 
                        format=fmt1
                    )
                    val2 = st.number_input(
                        f"Enter value for '{feature_2}'", 
                        value=float(X[feature_2].median()), 
                        min_value=min2,
                        max_value=max2,
                        step=step2, 
                        format=fmt2
                    )
                 
                input_df = pd.DataFrame([[val1, val2]], columns=X.columns)
                prediction = model.predict(input_df)[0]

                with p_col1:
                    st.metric(label=f"Predicted {target}", value=f"{prediction:,.2f}")
                    
                # Dynamically format prediction as integer or continuous float based on target type
                clean_target = y.dropna()
                is_target_int = np.issubdtype(y.dtype, np.integer) or (clean_target % 1 == 0).all()

                if is_target_int:
                    formatted_prediction = f"{round(prediction):,d}"
                else:
                    formatted_prediction = f"{prediction:,.2f}"

                with p_col1:
                    st.metric(label=f"Predicted {target}", value=formatted_prediction)

                # Optional Dynamic Threshold Context
                if adj_r2 >= 0.75:
                    st.success(
                        f"**Optional Fit Context — High Model Fit.**"
                        f"This model explains **{adj_r2 * 100:.1f}%** of the variance in target output **{target}**. "
                        f"While this offers strong baseline reference context for **{target}**, remember that MAE = **{mae:,.2f}** and predictions should complement, not replace, domain expertise."
                    )
                elif adj_r2 >= 0.50:
                    st.warning(
                        f"**Optional Fit Context — Moderate Model Fit.**"
                        f"This model explains **{adj_r2 * 100:.1f}%** of the variance in target output **{target}**. "
                        f"Predictions may provide optional directional reference for **{target}**, but should be treated cautiously given the moderate error margin (MAE = **{mae:,.2f}**)."
                    )
                else:
                    st.error(
                        f"**Optional Fit Context — Low Model Fit.**"
                        f"This model explains only **{max(0.0, adj_r2 * 100):.1f}%** of the variance in target output **{target}**. "
                        f"The selected features (**{feature_1}**, **{feature_2}**) fail to reliably predict **{target}**, so predictions are low quality and should not be relied upon."
                    )
                    
                # Model Usage & Risk Notice Section
                st.markdown("---")
                st.subheader("Model Usage & Risk Notice")
                
                # 3. What model is used for and consequences of error
                st.info(
                    "This linear regression model is a simple prediction tool used to estimate unknown numerical outcomes based on mathematical relationships found between input features. "
                    "Because this is a simple two-feature model, predictions should **never be used solely for decision-making**. "
                    "If predictions are wrong, relying on them can lead to flawed decision-making, poor strategy, and unmitigated financial/operational risk."
                )


else:
    st.info("Please upload a CSV file with at least 3 numeric columns to get started.")
