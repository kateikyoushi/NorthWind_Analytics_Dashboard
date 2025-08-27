import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta, datetime
import numpy as np
from operator import attrgetter
import warnings
warnings.filterwarnings('ignore')

# Optional ML imports - will gracefully degrade if not available
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("⚠️ Scikit-learn not installed. ML features will be disabled. Install with: pip install scikit-learn")

# --- Page Config ---
st.set_page_config(page_title="Northwind Advanced Analytics", layout="wide", initial_sidebar_state="expanded")

# --- Enhanced Data Loading and Preprocessing ---
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    date_cols = ['OrderDate', 'ShipDate', 'InvoiceDueDate', 'InvoicePaidDate']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce', format='mixed')

    df['LineTotal'] = pd.to_numeric(df['LineTotal'].replace('[\$,]', '', regex=True))
    df['ShippingFee'] = pd.to_numeric(df['ShippingFee'].replace('[\$,]', '', regex=True))
    df['Taxes'] = pd.to_numeric(df['Taxes'].replace('[\$,]', '', regex=True))
    df['TotalRevenue'] = df['LineTotal'] + df['ShippingFee'] + df['Taxes']
    df['FulfillmentTime'] = (df['ShipDate'] - df['OrderDate']).dt.days
    
    # Enhanced metrics
    df['PaymentDelay'] = (df['InvoicePaidDate'] - df['InvoiceDueDate']).dt.days
    df['IsLatePayment'] = df['PaymentDelay'] > 0
    df['IsCancelled'] = df['OrderStatus'] == 'Cancelled'
    df['OrderMonth'] = df['OrderDate'].dt.to_period('M')
    df['OrderQuarter'] = df['OrderDate'].dt.to_period('Q')
    
    return df

# --- Advanced Analytics Functions ---
@st.cache_data
def perform_cohort_analysis(df):
    """Enhanced cohort analysis for customer retention"""
    try:
        # Get customer's first purchase date
        df_cohort = df.copy()
        df_cohort['OrderPeriod'] = df_cohort['OrderDate'].dt.to_period('M')
        df_cohort['CohortGroup'] = df_cohort.groupby('CustomerID')['OrderDate'].transform('min').dt.to_period('M')
        
        # Calculate period number for each transaction
        df_cohort['PeriodNumber'] = (df_cohort['OrderPeriod'] - df_cohort['CohortGroup']).apply(attrgetter('n'))
        
        # Create cohort table
        cohort_data = df_cohort.groupby(['CohortGroup', 'PeriodNumber'])['CustomerID'].nunique().reset_index()
        cohort_sizes = df_cohort.groupby('CohortGroup')['CustomerID'].nunique().reset_index()
        cohort_sizes.rename(columns={'CustomerID': 'CohortGroupSize'}, inplace=True)
        
        cohort_table = cohort_data.merge(cohort_sizes, on='CohortGroup')
        cohort_table['Retention'] = cohort_table['CustomerID'] / cohort_table['CohortGroupSize']
        
        return cohort_table.pivot(index='CohortGroup', columns='PeriodNumber', values='Retention')
    except Exception as e:
        st.error(f"Error in cohort analysis: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def predict_churn(df):
    """Machine learning model for churn prediction"""
    if not SKLEARN_AVAILABLE:
        return None, None, None
    
    try:
        # Prepare features for churn prediction
        today = df['OrderDate'].max()
        customer_features = df.groupby('CustomerID').agg({
            'OrderDate': lambda x: (today - x.max()).days,  # Recency
            'OrderID': 'count',  # Frequency
            'TotalRevenue': ['sum', 'mean'],  # Monetary
            'IsCancelled': 'sum',  # Cancellation count
            'IsLatePayment': 'mean',  # Late payment rate
            'FulfillmentTime': 'mean'  # Avg fulfillment time
        }).reset_index()
        
        customer_features.columns = ['CustomerID', 'Recency', 'Frequency', 'TotalSpend', 'AvgOrderValue', 
                                    'CancellationCount', 'LatePaymentRate', 'AvgFulfillmentTime']
        
        # Define churn (customers inactive for 90+ days)
        customer_features['IsChurned'] = (customer_features['Recency'] > 90).astype(int)
        
        # Prepare features for ML
        feature_cols = ['Recency', 'Frequency', 'TotalSpend', 'AvgOrderValue', 
                       'CancellationCount', 'LatePaymentRate', 'AvgFulfillmentTime']
        
        X = customer_features[feature_cols].fillna(0)
        y = customer_features['IsChurned']
        
        if len(X) < 10:  # Not enough data for ML
            return customer_features, None, feature_cols
        
        # Train model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict churn probability
        customer_features['ChurnProbability'] = model.predict_proba(X_scaled)[:, 1]
        
        return customer_features, model, feature_cols
    
    except Exception as e:
        st.error(f"Error in churn prediction: {str(e)}")
        return None, None, None

@st.cache_data
def advanced_inventory_analysis(df):
    """Advanced inventory turnover and ABC analysis"""
    product_metrics = df.groupby(['ProductID', 'ProductName']).agg({
        'Quantity': 'sum',
        'LineTotal': 'sum',
        'OrderID': 'nunique'
    }).reset_index()
    
    # ABC Analysis
    product_metrics = product_metrics.sort_values('LineTotal', ascending=False)
    product_metrics['CumulativeRevenue'] = product_metrics['LineTotal'].cumsum()
    total_revenue = product_metrics['LineTotal'].sum()
    product_metrics['CumulativePercent'] = product_metrics['CumulativeRevenue'] / total_revenue * 100
    
    # Classify products
    def classify_abc(cum_percent):
        if cum_percent <= 80:
            return 'A'
        elif cum_percent <= 95:
            return 'B'
        else:
            return 'C'
    
    product_metrics['ABC_Category'] = product_metrics['CumulativePercent'].apply(classify_abc)
    
    return product_metrics

try:
    df = load_data("northwind_flat.csv")
except FileNotFoundError:
    st.error("The file `northwind_flat.csv` was not found. Please upload the file to the same directory.")
    st.stop()

# --- Enhanced Sidebar Navigation ---
with st.sidebar:
    st.header("🏢 Northwind Analytics Suite")
    st.markdown("---")
    st.subheader("📊 Analytics Modules")
    section = st.radio(
        "Select Business Intelligence Module:",
        [
            "🎯 Customer Intelligence & Churn",
            "📈 Sales Performance Analytics", 
            "📦 Smart Inventory Management",
            "🤝 Supplier Performance Scorecard",
            "🚚 Logistics & Delivery Analytics",
            "💰 Financial Analytics & Forecasting",
            "👥 Workforce Analytics"
        ]
    )
    
    st.markdown("---")
    st.subheader("🔧 Global Filters")
    
    # Enhanced date filtering
    min_date = df['OrderDate'].min()
    max_date = df['OrderDate'].max()
    if pd.notna(min_date) and pd.notna(max_date):
        date_range = st.slider(
            "📅 Order Date Range",
            value=(min_date.date(), max_date.date()),
            format="YYYY-MM-DD"
        )
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
        filtered_df = df[(df['OrderDate'] >= start_date) & (df['OrderDate'] <= end_date)]
    else:
        st.warning("Date columns contain missing values.")
        filtered_df = df.copy()
    
    # Additional filters
    selected_countries = st.multiselect(
        "🌍 Filter by Countries", 
        options=df['CustomerCountry'].unique(),
        default=df['CustomerCountry'].unique()[:5]
    )
    
    if selected_countries:
        filtered_df = filtered_df[filtered_df['CustomerCountry'].isin(selected_countries)]

# Moved this import to the top with other imports

# Helper function for currency formatting
def format_currency(value):
    return f"${value:,.2f}"

def format_number(value):
    return f"{value:,.0f}"

# --- Enhanced Dashboard Layout ---
st.title(f"📊 {section}")
st.markdown("---")

# --- Section 1: Enhanced Customer Intelligence & Churn Prediction ---
if section == "🎯 Customer Intelligence & Churn":
    
    st.header("🎯 Advanced Customer Intelligence & Churn Prediction")
    
    # Section Overview
    with st.expander("ℹ️ Why Customer Intelligence Matters", expanded=False):
        st.markdown("""
        **Customer Intelligence helps you:**
        - 📊 **Understand customer behavior patterns** through RFM analysis
        - 🎯 **Identify at-risk customers** before they churn
        - 💰 **Maximize customer lifetime value** through targeted retention
        - 📈 **Optimize marketing spend** by focusing on high-value segments
        - 🔄 **Improve customer retention** through data-driven insights
        """)
    
    if filtered_df.empty:
        st.warning("No data available in the selected date range.")
    else:
        # Key Metrics Dashboard
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_customers = filtered_df['CustomerID'].nunique()
            st.metric("Total Customers", format_number(total_customers))
        with col2:
            avg_clv = filtered_df.groupby('CustomerID')['TotalRevenue'].sum().mean()
            st.metric("Avg Customer LTV", format_currency(avg_clv))
        with col3:
            repeat_customers = filtered_df.groupby('CustomerID')['OrderID'].nunique()
            repeat_rate = (repeat_customers > 1).sum() / len(repeat_customers) * 100
            st.metric("Repeat Customer Rate", f"{repeat_rate:.1f}%")
        with col4:
            churn_rate = (filtered_df.groupby('CustomerID')['OrderDate'].max() < 
                         (filtered_df['OrderDate'].max() - timedelta(days=90))).sum() / total_customers * 100
            st.metric("Churn Rate (90 days)", f"{churn_rate:.1f}%")

        st.markdown("---")
        
        # Advanced RFM Analysis with K-Means Clustering
        st.subheader("🔍 Advanced RFM Analysis with ML Clustering")
        
        with st.expander("📖 Understanding RFM Analysis", expanded=False):
            st.markdown("""
            **RFM (Recency, Frequency, Monetary) Analysis** segments customers based on:
            - **Recency**: How recently they made a purchase
            - **Frequency**: How often they make purchases  
            - **Monetary**: How much they spend
            
            **Why it matters:**
            - 🎯 **Target high-value customers** with premium offers
            - 💰 **Re-engage dormant customers** with special promotions
            - 📊 **Optimize marketing campaigns** based on customer behavior
            - 🔍 **Identify customer segments** for personalized marketing
            """)
        
        tab1, tab2, tab3 = st.tabs(["RFM Segmentation", "ML-Based Clustering", "Cohort Analysis"])
        
        with tab1:
            st.info("📊 **RFM Segmentation**: Traditional approach using quartiles to segment customers into groups like 'Champions', 'Loyal Customers', etc.")
            
            # Enhanced RFM Analysis
            today = filtered_df['OrderDate'].max() + timedelta(days=1)
            rfm_df = filtered_df.groupby('CustomerID').agg(
                Recency=('OrderDate', lambda x: (today - x.max()).days),
                Frequency=('OrderID', 'count'),
                Monetary=('TotalRevenue', 'sum')
            ).reset_index()

            # K-means clustering on RFM
            if SKLEARN_AVAILABLE:
                rfm_features = rfm_df[['Recency', 'Frequency', 'Monetary']].fillna(0)
                scaler = StandardScaler()
                rfm_scaled = scaler.fit_transform(rfm_features)
                
                kmeans = KMeans(n_clusters=5, random_state=42)
                rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)
            else:
                # Simple quartile-based clustering if sklearn not available
                rfm_df['R_Quartile'] = pd.qcut(rfm_df['Recency'], 4, labels=['1','2','3','4'])
                rfm_df['F_Quartile'] = pd.qcut(rfm_df['Frequency'], 4, labels=['1','2','3','4'])
                rfm_df['M_Quartile'] = pd.qcut(rfm_df['Monetary'], 4, labels=['1','2','3','4'])
                rfm_df['Cluster'] = (rfm_df['R_Quartile'].astype(str) + 
                                   rfm_df['F_Quartile'].astype(str) + 
                                   rfm_df['M_Quartile'].astype(str)).astype('category').cat.codes
            
            # Visualizations
            col1, col2 = st.columns(2)
            with col1:
                fig_3d = px.scatter_3d(rfm_df, x='Recency', y='Frequency', z='Monetary', 
                                      color='Cluster', title='3D RFM Customer Clusters')
                st.plotly_chart(fig_3d, use_container_width=True)
            
            with col2:
                cluster_summary = rfm_df.groupby('Cluster').agg({
                    'Recency': 'mean',
                    'Frequency': 'mean', 
                    'Monetary': 'mean',
                    'CustomerID': 'count'
                }).round(2)
                st.subheader("Cluster Characteristics")
                st.dataframe(cluster_summary)

        with tab2:
            st.info("🤖 **ML-Powered Churn Prediction**: Uses machine learning to predict which customers are likely to stop buying, enabling proactive retention strategies.")
            
            # Machine Learning Churn Prediction
            st.subheader("🤖 ML-Powered Churn Prediction")
            
            if not SKLEARN_AVAILABLE:
                st.warning("Machine Learning features require scikit-learn. Please install it to use churn prediction.")
            else:
                churn_result = predict_churn(filtered_df)
                
                if churn_result[0] is not None:
                    churn_data, model, feature_cols = churn_result
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        # Churn Risk Distribution
                        if 'ChurnProbability' in churn_data.columns:
                            fig_churn_dist = px.histogram(churn_data, x='ChurnProbability', 
                                                         title='Churn Risk Distribution',
                                                         labels={'ChurnProbability': 'Churn Probability'})
                            st.plotly_chart(fig_churn_dist, use_container_width=True)
                        else:
                            st.info("Not enough data for churn probability calculation")
                    
                    with col2:
                        # High-risk customers
                        if 'ChurnProbability' in churn_data.columns:
                            high_risk = churn_data[churn_data['ChurnProbability'] > 0.7].sort_values('ChurnProbability', ascending=False)
                            st.subheader("🚨 High-Risk Customers")
                            if not high_risk.empty:
                                st.dataframe(high_risk[['CustomerID', 'Recency', 'Frequency', 'TotalSpend', 'ChurnProbability']].head(10))
                            else:
                                st.info("No high-risk customers identified")
                        else:
                            st.info("Churn analysis not available with current data")
                    
                    # Feature Importance
                    if model is not None and feature_cols is not None:
                        st.subheader("📊 Churn Prediction Factors")
                        importance_df = pd.DataFrame({
                            'Feature': feature_cols,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        fig_importance = px.bar(importance_df, x='Importance', y='Feature', 
                                               orientation='h', title='Feature Importance for Churn Prediction')
                        st.plotly_chart(fig_importance, use_container_width=True)
                else:
                    st.error("Could not perform churn prediction with current data")


        with tab3:
            st.info("📅 **Cohort Analysis**: Groups customers by their first purchase month and tracks their retention over time to understand customer lifecycle patterns.")
            
            # Cohort Analysis
            st.subheader("📅 Customer Cohort Analysis")
            cohort_table = perform_cohort_analysis(filtered_df)
            
            if not cohort_table.empty:
                # Heatmap visualization
                try:
                    # Convert Period objects to strings for JSON serialization
                    cohort_table_display = cohort_table.reset_index()
                    cohort_table_display['CohortGroup'] = cohort_table_display['CohortGroup'].astype(str)
                    cohort_table_display = cohort_table_display.set_index('CohortGroup')
                    
                    fig_cohort = px.imshow(cohort_table_display.iloc[:, :12], 
                                          title='Customer Retention Cohort Analysis',
                                          labels=dict(x="Period Number", y="Cohort Group", color="Retention Rate"),
                                          color_continuous_scale='RdYlGn')
                    st.plotly_chart(fig_cohort, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not create cohort heatmap: {str(e)}")
                    # Fallback: display as regular dataframe
                    st.dataframe(cohort_table.head())
            else:
                st.warning("Unable to perform cohort analysis with current data")

# --- Section 2: Enhanced Sales Performance Analytics ---
elif section == "📈 Sales Performance Analytics":
    st.header("📈 Advanced Sales Performance Analytics")
    
    # Section Overview
    with st.expander("ℹ️ Why Sales Performance Matters", expanded=False):
        st.markdown("""
        **Sales Performance Analytics helps you:**
        - 💰 **Track revenue trends** and identify growth opportunities
        - 📊 **Analyze product performance** to optimize inventory
        - 🌍 **Understand regional performance** for market expansion
        - 👥 **Evaluate employee performance** to improve sales effectiveness
        - 📈 **Monitor seasonal patterns** for better planning
        """)
    
    if filtered_df.empty:
        st.warning("No data available in the selected date range.")
    else:
        # Enhanced KPIs
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_sales = filtered_df['LineTotal'].sum()
            st.metric("Total Sales", format_currency(total_sales))
        with col2:
            total_orders = filtered_df['OrderID'].nunique()
            st.metric("Total Orders", format_number(total_orders))
        with col3:
            avg_order_value = total_sales / total_orders if total_orders > 0 else 0
            st.metric("Avg. Order Value", format_currency(avg_order_value))
        with col4:
            growth_rate = 0  # Placeholder for period-over-period growth
            st.metric("Sales Growth", f"{growth_rate:.1f}%")

        st.markdown("---")

        # Multi-tab layout for different analytics
        tab1, tab2, tab3, tab4 = st.tabs(["Product Analytics", "Regional Performance", "Employee Analytics", "Trend Analysis"])
        
        with tab1:
            st.info("📦 **Product Analytics**: Analyzes product performance using ABC classification to identify your most valuable products and optimize inventory management.")
            
            col1, col2 = st.columns(2)
            with col1:
                # Enhanced product analysis with ABC classification
                st.subheader("🏆 Top Products (ABC Analysis)")
                product_analysis = advanced_inventory_analysis(filtered_df)
                
                fig_abc = px.bar(product_analysis.head(20), x='LineTotal', y='ProductName',
                               color='ABC_Category', orientation='h',
                               title='Top 20 Products by Revenue (ABC Classification)')
                st.plotly_chart(fig_abc, use_container_width=True)
            
            with col2:
                st.subheader("📊 Product Performance Matrix")
                product_matrix = filtered_df.groupby('ProductName').agg({
                    'LineTotal': 'sum',
                    'Quantity': 'sum'
                }).reset_index()
                
                fig_matrix = px.scatter(product_matrix, x='Quantity', y='LineTotal',
                                       hover_data=['ProductName'],
                                       title='Product Performance: Revenue vs Volume')
                st.plotly_chart(fig_matrix, use_container_width=True)

        with tab2:
            st.info("🌍 **Regional Performance**: Identifies your strongest and weakest markets to guide expansion decisions and resource allocation.")
            
            # Enhanced regional analysis
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🌍 Sales by Region")
                regional_sales = filtered_df.groupby('CustomerCountry')['LineTotal'].sum().reset_index()
                fig_map = px.choropleth(regional_sales, locations='CustomerCountry', 
                                       locationmode='country names', color='LineTotal',
                                       title='Global Sales Distribution')
                st.plotly_chart(fig_map, use_container_width=True)
            
            with col2:
                st.subheader("🏙️ Top Cities Performance")
                city_sales = filtered_df.groupby(['CustomerCity', 'CustomerCountry'])['LineTotal'].sum().nlargest(15).reset_index()
                fig_cities = px.bar(city_sales, x='LineTotal', y='CustomerCity',
                                   orientation='h', title='Top 15 Cities by Sales')
                st.plotly_chart(fig_cities, use_container_width=True)

        with tab3:
            st.info("👥 **Employee Analytics**: Evaluates sales team performance to identify top performers, training needs, and optimize workforce productivity.")
            
            # Enhanced employee analytics
            st.subheader("👥 Employee Performance Dashboard")
            
            employee_metrics = filtered_df.groupby(['EmployeeName', 'EmployeeJobTitle']).agg({
                'LineTotal': 'sum',
                'OrderID': 'count',
                'IsCancelled': 'sum'
            }).reset_index()
            employee_metrics['CancellationRate'] = employee_metrics['IsCancelled'] / employee_metrics['OrderID'] * 100
            
            col1, col2 = st.columns(2)
            with col1:
                fig_emp_sales = px.bar(employee_metrics.nlargest(10, 'LineTotal'), 
                                      x='EmployeeName', y='LineTotal',
                                      title='Top 10 Employees by Sales')
                st.plotly_chart(fig_emp_sales, use_container_width=True)
            
            with col2:
                fig_emp_cancel = px.scatter(employee_metrics, x='OrderID', y='CancellationRate',
                                           hover_data=['EmployeeName'],
                                           title='Employee Order Volume vs Cancellation Rate')
                st.plotly_chart(fig_emp_cancel, use_container_width=True)

        with tab4:
            st.info("📈 **Trend Analysis**: Tracks sales patterns over time to identify seasonal trends, growth trajectories, and forecast future performance.")
            
            # Time series analysis
            st.subheader("📈 Sales Trend Analysis")
            
            # Monthly trend
            monthly_sales = filtered_df.groupby(filtered_df['OrderDate'].dt.to_period('M'))['LineTotal'].sum().reset_index()
            monthly_sales['OrderDate'] = monthly_sales['OrderDate'].astype(str)
            
            fig_trend = px.line(monthly_sales, x='OrderDate', y='LineTotal',
                               title='Monthly Sales Trend')
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # Seasonal analysis
            filtered_df['Month'] = filtered_df['OrderDate'].dt.month_name()
            seasonal_sales = filtered_df.groupby('Month')['LineTotal'].sum().reset_index()
            
            fig_seasonal = px.bar(seasonal_sales, x='Month', y='LineTotal',
                                 title='Seasonal Sales Pattern')
            st.plotly_chart(fig_seasonal, use_container_width=True)

# --- Section 3: Smart Inventory Management ---
elif section == "📦 Smart Inventory Management":
    st.header("📦 Smart Inventory Management & Optimization")
    
    if filtered_df.empty:
        st.warning("No data available in the selected date range.")
    else:
        # Inventory KPIs
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            unique_products = filtered_df['ProductID'].nunique()
            st.metric("Active Products", format_number(unique_products))
        with col2:
            total_quantity = filtered_df['Quantity'].sum()
            st.metric("Total Units Sold", format_number(total_quantity))
        with col3:
            avg_turnover = total_quantity / unique_products
            st.metric("Avg Turnover Rate", f"{avg_turnover:.1f}")
        with col4:
            stockout_risk = filtered_df.groupby('ProductName')['Quantity'].sum().quantile(0.25)
            st.metric("Low Stock Threshold", format_number(stockout_risk))

        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs(["ABC Analysis", "Demand Forecasting", "Reorder Optimization"])
        
        with tab1:
            st.subheader("📊 ABC Inventory Analysis")
            inventory_analysis = advanced_inventory_analysis(filtered_df)
            
            col1, col2 = st.columns(2)
            with col1:
                abc_summary = inventory_analysis['ABC_Category'].value_counts().reset_index()
                fig_abc_pie = px.pie(abc_summary, values='count', names='ABC_Category',
                                    title='Product Distribution by ABC Category')
                st.plotly_chart(fig_abc_pie, use_container_width=True)
            
            with col2:
                st.subheader("Category Performance")
                category_abc = inventory_analysis.groupby(['ABC_Category']).agg({
                    'LineTotal': 'sum',
                    'Quantity': 'sum'
                }).reset_index()
                st.dataframe(category_abc)

        with tab2:
            st.subheader("📈 Demand Forecasting")
            
            # Simple demand forecasting using moving averages
            selected_product = st.selectbox("Select Product for Forecasting", 
                                           filtered_df['ProductName'].unique())
            
            product_demand = filtered_df[filtered_df['ProductName'] == selected_product].copy()
            if not product_demand.empty:
                daily_demand = product_demand.groupby('OrderDate')['Quantity'].sum().reset_index()
                daily_demand['MovingAvg7'] = daily_demand['Quantity'].rolling(7).mean()
                daily_demand['MovingAvg30'] = daily_demand['Quantity'].rolling(30).mean()
                
                fig_forecast = px.line(daily_demand, x='OrderDate', y='Quantity',
                                      title=f'Demand Pattern: {selected_product}')
                fig_forecast.add_scatter(x=daily_demand['OrderDate'], y=daily_demand['MovingAvg7'], 
                                        name='7-day MA')
                fig_forecast.add_scatter(x=daily_demand['OrderDate'], y=daily_demand['MovingAvg30'], 
                                        name='30-day MA')
                st.plotly_chart(fig_forecast, use_container_width=True)

        with tab3:
            st.subheader("🚨 Reorder Alerts & Optimization")
            
            # Low-performing products
            low_performers = inventory_analysis[inventory_analysis['ABC_Category'] == 'C'].head(10)
            
            col1, col2 = st.columns(2)
            with col1:
                st.warning("⚠️ Slow-Moving Products (Category C)")
                st.dataframe(low_performers[['ProductName', 'LineTotal', 'Quantity']])
            
            with col2:
                st.success("🔥 High-Performing Products (Category A)")
                high_performers = inventory_analysis[inventory_analysis['ABC_Category'] == 'A'].head(10)
                st.dataframe(high_performers[['ProductName', 'LineTotal', 'Quantity']])

elif section == "🚚 Logistics & Delivery Analytics":
    st.header("🚚 Advanced Logistics & Delivery Analytics")
    
    # Section Overview
    with st.expander("ℹ️ Why Logistics Analytics Matters", expanded=False):
        st.markdown("""
        **Logistics & Delivery Analytics helps you:**
        - 🚚 **Optimize delivery performance** and reduce fulfillment times
        - 💰 **Control shipping costs** and identify cost-saving opportunities
        - 🌍 **Improve regional delivery** efficiency and customer satisfaction
        - 📦 **Evaluate shipper performance** to choose the best partners
        - 📊 **Track on-time delivery** rates to maintain service quality
        """)
    
    if filtered_df.empty:
        st.warning("No data available in the selected date range.")
    else:
        # Logistics KPIs
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_fulfillment = filtered_df['FulfillmentTime'].mean()
            st.metric("Avg Fulfillment Time", f"{avg_fulfillment:.1f} days")
        with col2:
            total_shipping_cost = filtered_df['ShippingFee'].sum()
            st.metric("Total Shipping Cost", format_currency(total_shipping_cost))
        with col3:
            on_time_delivery = (filtered_df['FulfillmentTime'] <= 7).sum() / len(filtered_df) * 100
            st.metric("On-Time Delivery Rate", f"{on_time_delivery:.1f}%")
        with col4:
            avg_shipping_fee = filtered_df['ShippingFee'].mean()
            st.metric("Avg Shipping Fee", format_currency(avg_shipping_fee))

        st.markdown("---")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Delivery Performance", "Shipping Cost Analysis", "Regional Logistics", "Shipper Performance"])
        
        with tab1:
            st.info("📊 **Delivery Performance**: Monitors fulfillment times, delivery success rates, and identifies bottlenecks in your supply chain.")
            
            st.subheader("📊 Delivery Performance Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                # Fulfillment time distribution
                fig_fulfillment = px.histogram(filtered_df, x='FulfillmentTime', nbins=20,
                                             title='Fulfillment Time Distribution')
                st.plotly_chart(fig_fulfillment, use_container_width=True)
            
            with col2:
                # Delivery status analysis
                status_counts = filtered_df['OrderStatus'].value_counts().reset_index()
                fig_status = px.pie(status_counts, values='count', names='OrderStatus',
                                   title='Order Status Distribution')
                st.plotly_chart(fig_status, use_container_width=True)
            
            # Fulfillment time trends
            monthly_fulfillment = filtered_df.groupby(filtered_df['OrderDate'].dt.to_period('M'))['FulfillmentTime'].mean().reset_index()
            monthly_fulfillment['OrderDate'] = monthly_fulfillment['OrderDate'].astype(str)
            fig_trend = px.line(monthly_fulfillment, x='OrderDate', y='FulfillmentTime',
                               title='Average Fulfillment Time Trend')
            st.plotly_chart(fig_trend, use_container_width=True)

        with tab2:
            st.info("💰 **Shipping Cost Analysis**: Breaks down shipping expenses by product category and fulfillment time to identify cost optimization opportunities.")
            
            st.subheader("💰 Shipping Cost Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                # Shipping cost distribution
                fig_shipping_dist = px.histogram(filtered_df, x='ShippingFee', nbins=20,
                                                title='Shipping Fee Distribution')
                st.plotly_chart(fig_shipping_dist, use_container_width=True)
            
            with col2:
                # Shipping cost vs fulfillment time
                fig_cost_time = px.scatter(filtered_df, x='FulfillmentTime', y='ShippingFee',
                                          title='Shipping Cost vs Fulfillment Time')
                st.plotly_chart(fig_cost_time, use_container_width=True)
            
            # Shipping cost by category
            shipping_by_category = filtered_df.groupby('Category').agg({
                'ShippingFee': 'mean',
                'FulfillmentTime': 'mean',
                'OrderID': 'count'
            }).reset_index()
            
            fig_category = px.bar(shipping_by_category, x='Category', y='ShippingFee',
                                 title='Average Shipping Cost by Product Category')
            st.plotly_chart(fig_category, use_container_width=True)

        with tab3:
            st.info("🌍 **Regional Logistics**: Compares delivery performance across different regions to optimize logistics strategies and identify market-specific improvements.")
            
            st.subheader("🌍 Regional Logistics Performance")
            
            col1, col2 = st.columns(2)
            with col1:
                # Delivery performance by country
                regional_performance = filtered_df.groupby('ShipCountry').agg({
                    'FulfillmentTime': 'mean',
                    'ShippingFee': 'mean',
                    'OrderID': 'count'
                }).reset_index()
                
                fig_regional = px.bar(regional_performance.nlargest(10, 'OrderID'), 
                                     x='ShipCountry', y='FulfillmentTime',
                                     title='Avg Fulfillment Time by Country (Top 10)')
                st.plotly_chart(fig_regional, use_container_width=True)
            
            with col2:
                # Shipping cost by region
                fig_regional_cost = px.bar(regional_performance.nlargest(10, 'OrderID'),
                                          x='ShipCountry', y='ShippingFee',
                                          title='Avg Shipping Cost by Country (Top 10)')
                st.plotly_chart(fig_regional_cost, use_container_width=True)
            
            # Global delivery heatmap (if enough data)
            if len(regional_performance) > 5:
                fig_map = px.choropleth(regional_performance, locations='ShipCountry',
                                       locationmode='country names', color='FulfillmentTime',
                                       title='Global Fulfillment Time Heatmap')
                st.plotly_chart(fig_map, use_container_width=True)

        with tab4:
            st.info("🚛 **Shipper Performance**: Analyzes shipping partner effectiveness by comparing delivery speed, cost efficiency, and overall performance scores.")
            
            st.subheader("🚛 Shipper Performance Analysis")
            
            # Shipper performance metrics
            shipper_metrics = filtered_df.groupby('ShipperName').agg({
                'FulfillmentTime': 'mean',
                'ShippingFee': 'mean',
                'OrderID': 'count',
                'LineTotal': 'sum'
            }).reset_index()
            
            shipper_metrics['Avg_Order_Value'] = shipper_metrics['LineTotal'] / shipper_metrics['OrderID']
            
            col1, col2 = st.columns(2)
            with col1:
                # Shipper by volume
                fig_shipper_volume = px.bar(shipper_metrics.nlargest(10, 'OrderID'),
                                           x='ShipperName', y='OrderID',
                                           title='Orders by Shipper (Top 10)')
                st.plotly_chart(fig_shipper_volume, use_container_width=True)
            
            with col2:
                # Shipper performance matrix
                fig_shipper_matrix = px.scatter(shipper_metrics, x='FulfillmentTime', y='ShippingFee',
                                               size='OrderID', hover_name='ShipperName',
                                               title='Shipper Performance: Speed vs Cost')
                st.plotly_chart(fig_shipper_matrix, use_container_width=True)
            
            # Shipper efficiency analysis
            st.subheader("📈 Shipper Efficiency Metrics")
            shipper_efficiency = shipper_metrics.copy()
            shipper_efficiency['Efficiency_Score'] = (shipper_efficiency['Avg_Order_Value'] / 
                                                    (shipper_efficiency['FulfillmentTime'] * shipper_efficiency['ShippingFee']))
            
            fig_efficiency = px.bar(shipper_efficiency.nlargest(10, 'Efficiency_Score'),
                                   x='ShipperName', y='Efficiency_Score',
                                   title='Shipper Efficiency Scores (Top 10)')
            st.plotly_chart(fig_efficiency, use_container_width=True)

elif section == "👥 Workforce Analytics":
    st.header("👥 Advanced Workforce Analytics")
    
    # Section Overview
    with st.expander("ℹ️ Why Workforce Analytics Matters", expanded=False):
        st.markdown("""
        **Workforce Analytics helps you:**
        - 👥 **Optimize employee performance** through data-driven insights
        - 📊 **Identify training opportunities** and skill gaps
        - 💰 **Improve productivity** by understanding workload distribution
        - 🎯 **Enhance job role effectiveness** through comparative analysis
        - 📈 **Track employee development** and career progression
        """)
    
    if filtered_df.empty:
        st.warning("No data available in the selected date range.")
    else:
        # Workforce KPIs
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_employees = filtered_df['EmployeeID'].nunique()
            st.metric("Active Employees", format_number(total_employees))
        with col2:
            avg_orders_per_employee = filtered_df['OrderID'].nunique() / total_employees
            st.metric("Avg Orders per Employee", f"{avg_orders_per_employee:.1f}")
        with col3:
            total_sales_workforce = filtered_df['LineTotal'].sum()
            st.metric("Total Sales by Workforce", format_currency(total_sales_workforce))
        with col4:
            avg_employee_sales = total_sales_workforce / total_employees
            st.metric("Avg Sales per Employee", format_currency(avg_employee_sales))

        st.markdown("---")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Employee Performance", "Job Role Analysis", "Workforce Efficiency", "Employee Trends"])
        
        with tab1:
            st.info("👤 **Employee Performance**: Evaluates individual employee contributions, sales performance, and order handling effectiveness.")
            
            st.subheader("👤 Individual Employee Performance")
            
            # Employee performance metrics
            employee_metrics = filtered_df.groupby(['EmployeeID', 'EmployeeName', 'EmployeeJobTitle']).agg({
                'LineTotal': 'sum',
                'OrderID': 'count',
                'Quantity': 'sum',
                'FulfillmentTime': 'mean',
                'IsCancelled': 'sum'
            }).reset_index()
            
            employee_metrics['CancellationRate'] = employee_metrics['IsCancelled'] / employee_metrics['OrderID'] * 100
            employee_metrics['Avg_Order_Value'] = employee_metrics['LineTotal'] / employee_metrics['OrderID']
            
            col1, col2 = st.columns(2)
            with col1:
                # Top performers by sales
                fig_top_sales = px.bar(employee_metrics.nlargest(10, 'LineTotal'),
                                      x='EmployeeName', y='LineTotal',
                                      title='Top 10 Employees by Sales')
                st.plotly_chart(fig_top_sales, use_container_width=True)
            
            with col2:
                # Top performers by order volume
                fig_top_orders = px.bar(employee_metrics.nlargest(10, 'OrderID'),
                                       x='EmployeeName', y='OrderID',
                                       title='Top 10 Employees by Order Volume')
                st.plotly_chart(fig_top_orders, use_container_width=True)
            
            # Employee performance matrix
            fig_employee_matrix = px.scatter(employee_metrics, x='OrderID', y='Avg_Order_Value',
                                            size='LineTotal', hover_name='EmployeeName',
                                            color='EmployeeJobTitle',
                                            title='Employee Performance Matrix')
            st.plotly_chart(fig_employee_matrix, use_container_width=True)

        with tab2:
            st.info("🏢 **Job Role Analysis**: Compares performance across different job positions to understand role effectiveness and identify optimization opportunities.")
            
            st.subheader("🏢 Job Role Analysis")
            
            # Job role performance
            job_role_metrics = filtered_df.groupby('EmployeeJobTitle').agg({
                'LineTotal': 'sum',
                'OrderID': 'count',
                'EmployeeID': 'nunique',
                'FulfillmentTime': 'mean',
                'IsCancelled': 'mean'
            }).reset_index()
            
            job_role_metrics['Avg_Sales_per_Employee'] = job_role_metrics['LineTotal'] / job_role_metrics['EmployeeID']
            job_role_metrics['CancellationRate'] = job_role_metrics['IsCancelled'] * 100
            
            col1, col2 = st.columns(2)
            with col1:
                # Sales by job role
                fig_job_sales = px.bar(job_role_metrics, x='EmployeeJobTitle', y='LineTotal',
                                      title='Total Sales by Job Role')
                st.plotly_chart(fig_job_sales, use_container_width=True)
            
            with col2:
                # Average sales per employee by role
                fig_job_avg = px.bar(job_role_metrics, x='EmployeeJobTitle', y='Avg_Sales_per_Employee',
                                    title='Average Sales per Employee by Role')
                st.plotly_chart(fig_job_avg, use_container_width=True)
            
            # Job role efficiency
            fig_job_efficiency = px.scatter(job_role_metrics, x='FulfillmentTime', y='CancellationRate',
                                           size='OrderID', hover_name='EmployeeJobTitle',
                                           title='Job Role Efficiency: Fulfillment vs Cancellation')
            st.plotly_chart(fig_job_efficiency, use_container_width=True)

        with tab3:
            st.info("⚡ **Workforce Efficiency**: Measures employee productivity through efficiency scoring and identifies opportunities for performance optimization.")
            
            st.subheader("⚡ Workforce Efficiency Metrics")
            
            # Calculate efficiency metrics
            workforce_efficiency = employee_metrics.copy()
            workforce_efficiency['Efficiency_Score'] = (workforce_efficiency['LineTotal'] / workforce_efficiency['OrderID']) / (workforce_efficiency['FulfillmentTime'] + 1)
            
            col1, col2 = st.columns(2)
            with col1:
                # Efficiency distribution
                fig_efficiency_dist = px.histogram(workforce_efficiency, x='Efficiency_Score', nbins=20,
                                                  title='Employee Efficiency Distribution')
                st.plotly_chart(fig_efficiency_dist, use_container_width=True)
            
            with col2:
                # Top efficient employees
                fig_top_efficient = px.bar(workforce_efficiency.nlargest(10, 'Efficiency_Score'),
                                          x='EmployeeName', y='Efficiency_Score',
                                          title='Top 10 Most Efficient Employees')
                st.plotly_chart(fig_top_efficient, use_container_width=True)
            
            # Efficiency vs performance
            fig_eff_perf = px.scatter(workforce_efficiency, x='Efficiency_Score', y='LineTotal',
                                     hover_name='EmployeeName', color='EmployeeJobTitle',
                                     title='Efficiency vs Total Sales')
            st.plotly_chart(fig_eff_perf, use_container_width=True)

        with tab4:
            st.info("📈 **Employee Trends**: Tracks performance changes over time and analyzes employee tenure to understand career progression and retention patterns.")
            
            st.subheader("📈 Employee Performance Trends")
            
            # Monthly employee performance
            monthly_employee = filtered_df.groupby([filtered_df['OrderDate'].dt.to_period('M'), 'EmployeeName']).agg({
                'LineTotal': 'sum',
                'OrderID': 'count'
            }).reset_index()
            
            monthly_employee['OrderDate'] = monthly_employee['OrderDate'].astype(str)
            
            # Select top employees for trend analysis
            top_employees = employee_metrics.nlargest(5, 'LineTotal')['EmployeeName'].tolist()
            monthly_top = monthly_employee[monthly_employee['EmployeeName'].isin(top_employees)]
            
            fig_trend_sales = px.line(monthly_top, x='OrderDate', y='LineTotal', color='EmployeeName',
                                     title='Sales Trends for Top 5 Employees')
            st.plotly_chart(fig_trend_sales, use_container_width=True)
            
            # Employee hiring/retention analysis (based on order dates)
            employee_timeline = filtered_df.groupby('EmployeeName').agg({
                'OrderDate': ['min', 'max', 'count']
            }).reset_index()
            employee_timeline.columns = ['EmployeeName', 'First_Order', 'Last_Order', 'Total_Orders']
            employee_timeline['Tenure_Days'] = (employee_timeline['Last_Order'] - employee_timeline['First_Order']).dt.days
            
            fig_tenure = px.scatter(employee_timeline, x='Tenure_Days', y='Total_Orders',
                                   hover_name='EmployeeName',
                                   title='Employee Tenure vs Order Volume')
            st.plotly_chart(fig_tenure, use_container_width=True)

elif section == "🤝 Supplier Performance Scorecard":
    st.header("🤝 Advanced Supplier Performance Analytics")
    
    if filtered_df.empty:
        st.warning("No data available in the selected date range.")
    else:
        # Supplier KPIs
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_suppliers = filtered_df['SupplierID'].nunique()
            st.metric("Active Suppliers", format_number(total_suppliers))
        with col2:
            total_procurement = filtered_df['LineTotal'].sum()
            st.metric("Total Procurement", format_currency(total_procurement))
        with col3:
            avg_discount = filtered_df['Discount'].mean() * 100
            st.metric("Avg Discount Rate", f"{avg_discount:.1f}%")
        with col4:
            avg_supplier_rating = 4.2  # Placeholder - would need supplier rating data
            st.metric("Avg Supplier Rating", f"{avg_supplier_rating:.1f}/5")

        st.markdown("---")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Supplier Performance", "Procurement Analysis", "Quality Metrics", "Supplier Trends"])
        
        with tab1:
            st.subheader("🏆 Supplier Performance Overview")
            
            # Comprehensive supplier metrics
            supplier_metrics = filtered_df.groupby(['SupplierID', 'SupplierName']).agg({
                'LineTotal': 'sum',
                'Quantity': 'sum',
                'OrderID': 'count',
                'Discount': 'mean',
                'FulfillmentTime': 'mean',
                'IsCancelled': 'mean',
                'ProductID': 'nunique'
            }).reset_index()
            
            supplier_metrics['CancellationRate'] = supplier_metrics['IsCancelled'] * 100
            supplier_metrics['Avg_Order_Value'] = supplier_metrics['LineTotal'] / supplier_metrics['OrderID']
            supplier_metrics['Products_Offered'] = supplier_metrics['ProductID']
            
            col1, col2 = st.columns(2)
            with col1:
                # Top suppliers by procurement volume
                fig_top_suppliers = px.bar(supplier_metrics.nlargest(10, 'LineTotal'),
                                          x='SupplierName', y='LineTotal',
                                          title='Top 10 Suppliers by Procurement Volume')
                st.plotly_chart(fig_top_suppliers, use_container_width=True)
            
            with col2:
                # Supplier performance matrix
                fig_supplier_matrix = px.scatter(supplier_metrics, x='FulfillmentTime', y='CancellationRate',
                                                size='LineTotal', hover_name='SupplierName',
                                                title='Supplier Performance: Fulfillment vs Quality')
                st.plotly_chart(fig_supplier_matrix, use_container_width=True)
            
            # Supplier efficiency analysis
            supplier_metrics['Efficiency_Score'] = (supplier_metrics['Avg_Order_Value'] * supplier_metrics['Products_Offered']) / (supplier_metrics['FulfillmentTime'] + 1)
            
            fig_efficiency = px.bar(supplier_metrics.nlargest(10, 'Efficiency_Score'),
                                   x='SupplierName', y='Efficiency_Score',
                                   title='Top 10 Most Efficient Suppliers')
            st.plotly_chart(fig_efficiency, use_container_width=True)

        with tab2:
            st.subheader("📊 Procurement Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                # Procurement by category
                procurement_by_category = filtered_df.groupby('Category').agg({
                    'LineTotal': 'sum',
                    'SupplierID': 'nunique'
                }).reset_index()
                
                fig_category_procurement = px.bar(procurement_by_category, x='Category', y='LineTotal',
                                                 title='Procurement by Product Category')
                st.plotly_chart(fig_category_procurement, use_container_width=True)
            
            with col2:
                # Supplier concentration analysis
                supplier_concentration = supplier_metrics.sort_values('LineTotal', ascending=False)
                supplier_concentration['Cumulative_Share'] = supplier_concentration['LineTotal'].cumsum() / supplier_concentration['LineTotal'].sum() * 100
                
                fig_concentration = px.line(supplier_concentration.head(20), x=supplier_concentration.head(20).index + 1, 
                                          y='Cumulative_Share', title='Supplier Concentration (Pareto Analysis)')
                fig_concentration.update_xaxes(title='Number of Suppliers')
                fig_concentration.update_yaxes(title='Cumulative Procurement Share (%)')
                st.plotly_chart(fig_concentration, use_container_width=True)
            
            # Monthly procurement trends
            monthly_procurement = filtered_df.groupby(filtered_df['OrderDate'].dt.to_period('M')).agg({
                'LineTotal': 'sum',
                'SupplierID': 'nunique'
            }).reset_index()
            monthly_procurement['OrderDate'] = monthly_procurement['OrderDate'].astype(str)
            
            fig_procurement_trend = px.line(monthly_procurement, x='OrderDate', y='LineTotal',
                                           title='Monthly Procurement Trends')
            st.plotly_chart(fig_procurement_trend, use_container_width=True)

        with tab3:
            st.subheader("🎯 Supplier Quality Metrics")
            
            # Quality metrics analysis
            quality_metrics = supplier_metrics.copy()
            
            col1, col2 = st.columns(2)
            with col1:
                # Cancellation rate by supplier
                fig_cancellation = px.bar(quality_metrics.nsmallest(10, 'CancellationRate'),
                                         x='SupplierName', y='CancellationRate',
                                         title='Lowest Cancellation Rates by Supplier')
                st.plotly_chart(fig_cancellation, use_container_width=True)
            
            with col2:
                # Fulfillment time distribution
                fig_fulfillment_dist = px.histogram(filtered_df, x='FulfillmentTime', nbins=20,
                                                   title='Supplier Fulfillment Time Distribution')
                st.plotly_chart(fig_fulfillment_dist, use_container_width=True)
            
            # Supplier quality scorecard
            st.subheader("📋 Supplier Quality Scorecard")
            
            # Calculate quality scores (weighted average of multiple factors)
            quality_metrics['Quality_Score'] = (
                (1 - quality_metrics['CancellationRate']/100) * 0.4 +  # 40% weight on low cancellation
                (1 / (quality_metrics['FulfillmentTime'] + 1)) * 0.3 +  # 30% weight on fast fulfillment
                (quality_metrics['Discount'] / quality_metrics['Discount'].max()) * 0.2 +  # 20% weight on discount
                (quality_metrics['Products_Offered'] / quality_metrics['Products_Offered'].max()) * 0.1  # 10% weight on product variety
            ) * 100
            
            quality_scorecard = quality_metrics[['SupplierName', 'Quality_Score', 'CancellationRate', 'FulfillmentTime', 'Discount', 'Products_Offered']].copy()
            quality_scorecard = quality_scorecard.sort_values('Quality_Score', ascending=False)
            
            fig_scorecard = px.bar(quality_scorecard.head(15), x='SupplierName', y='Quality_Score',
                                  title='Supplier Quality Scorecard (Top 15)')
            st.plotly_chart(fig_scorecard, use_container_width=True)
            
            # Display detailed scorecard
            st.dataframe(quality_scorecard.head(10).round(2))

        with tab4:
            st.subheader("📈 Supplier Performance Trends")
            
            # Supplier performance over time
            supplier_trends = filtered_df.groupby([filtered_df['OrderDate'].dt.to_period('M'), 'SupplierName']).agg({
                'LineTotal': 'sum',
                'OrderID': 'count',
                'FulfillmentTime': 'mean'
            }).reset_index()
            
            supplier_trends['OrderDate'] = supplier_trends['OrderDate'].astype(str)
            
            # Select top suppliers for trend analysis
            top_suppliers = supplier_metrics.nlargest(5, 'LineTotal')['SupplierName'].tolist()
            trends_top = supplier_trends[supplier_trends['SupplierName'].isin(top_suppliers)]
            
            fig_supplier_trends = px.line(trends_top, x='OrderDate', y='LineTotal', color='SupplierName',
                                         title='Procurement Trends for Top 5 Suppliers')
            st.plotly_chart(fig_supplier_trends, use_container_width=True)
            
            # Supplier onboarding/retention analysis
            supplier_timeline = filtered_df.groupby('SupplierName').agg({
                'OrderDate': ['min', 'max', 'count']
            }).reset_index()
            supplier_timeline.columns = ['SupplierName', 'First_Order', 'Last_Order', 'Total_Orders']
            supplier_timeline['Relationship_Days'] = (supplier_timeline['Last_Order'] - supplier_timeline['First_Order']).dt.days
            
            fig_relationship = px.scatter(supplier_timeline, x='Relationship_Days', y='Total_Orders',
                                        hover_name='SupplierName', size='Total_Orders',
                                        title='Supplier Relationship Duration vs Order Volume')
            st.plotly_chart(fig_relationship, use_container_width=True)

elif section == "💰 Financial Analytics & Forecasting":
    st.header("💰 Advanced Financial Analytics")
    
    if filtered_df.empty:
        st.warning("No data available in the selected date range.")
    else:
        # Financial KPIs
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_revenue = filtered_df['TotalRevenue'].sum()
            st.metric("Total Revenue", format_currency(total_revenue))
        with col2:
            outstanding_amount = filtered_df[filtered_df['IsLatePayment']]['TotalRevenue'].sum()
            st.metric("Outstanding Receivables", format_currency(outstanding_amount))
        with col3:
            on_time_rate = ((~filtered_df['IsLatePayment']).sum() / len(filtered_df)) * 100
            st.metric("On-Time Payment Rate", f"{on_time_rate:.1f}%")
        with col4:
            avg_payment_delay = filtered_df[filtered_df['IsLatePayment']]['PaymentDelay'].mean()
            st.metric("Avg Payment Delay", f"{avg_payment_delay:.0f} days")

        # Payment analysis
        st.subheader("💳 Payment Analysis")
        tab1, tab2 = st.tabs(["Payment Methods", "Aging Analysis"])
        
        with tab1:
            payment_analysis = filtered_df.groupby('PaymentType')['TotalRevenue'].sum().reset_index()
            fig_payment = px.pie(payment_analysis, values='TotalRevenue', names='PaymentType',
                                title='Revenue by Payment Method')
            st.plotly_chart(fig_payment, use_container_width=True)
        
        with tab2:
            if filtered_df['IsLatePayment'].any():
                late_payments = filtered_df[filtered_df['IsLatePayment']].copy()
                fig_aging = px.histogram(late_payments, x='PaymentDelay', nbins=20,
                                       title='Aging Analysis: Days Overdue Distribution')
                st.plotly_chart(fig_aging, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("### 📈 Northwind Advanced Analytics Dashboard")
st.markdown("*Powered by James Andrew Dorado - Ako nagid Ni faket*")