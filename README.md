# Northwind Advanced Analytics Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📊 Overview

A comprehensive business intelligence dashboard built with Streamlit for analyzing Northwind Traders data. This interactive application provides advanced analytics across multiple business domains including customer intelligence, sales performance, inventory management, logistics, workforce analytics, supplier performance, and financial forecasting.

## 🚀 Features

### 🎯 Customer Intelligence & Churn Prediction
- **RFM Analysis**: Recency, Frequency, Monetary segmentation with ML clustering
- **Churn Prediction**: Machine learning-powered customer churn risk assessment
- **Cohort Analysis**: Customer retention tracking by acquisition cohorts

### 📈 Sales Performance Analytics
- **Product Analytics**: ABC classification for inventory optimization
- **Regional Performance**: Geographic sales distribution and market analysis
- **Employee Analytics**: Sales team performance evaluation
- **Trend Analysis**: Time-series sales patterns and forecasting

### 📦 Smart Inventory Management
- **ABC Analysis**: Product classification for inventory prioritization
- **Demand Forecasting**: Moving average-based demand prediction
- **Reorder Optimization**: Low-stock alerts and optimization recommendations

### 🚚 Logistics & Delivery Analytics
- **Delivery Performance**: Fulfillment time analysis and bottleneck identification
- **Shipping Cost Analysis**: Cost optimization by category and region
- **Regional Logistics**: Global delivery performance mapping
- **Shipper Performance**: Partner evaluation and efficiency scoring

### 👥 Workforce Analytics
- **Employee Performance**: Individual contribution and productivity metrics
- **Job Role Analysis**: Position effectiveness and comparative analysis
- **Workforce Efficiency**: Productivity scoring and optimization insights
- **Employee Trends**: Performance tracking and tenure analysis

### 🤝 Supplier Performance Scorecard
- **Supplier Performance**: Procurement volume and efficiency metrics
- **Procurement Analysis**: Category-wise spending and supplier concentration
- **Quality Metrics**: Cancellation rates, fulfillment times, and quality scoring
- **Supplier Trends**: Relationship duration and performance tracking

### 💰 Financial Analytics & Forecasting
- **Revenue Analysis**: Total revenue, outstanding receivables, and payment patterns
- **Payment Methods**: Revenue distribution by payment type
- **Aging Analysis**: Overdue payment tracking and risk assessment

## 📋 Prerequisites

- Python 3.8 or higher
- Required packages listed in [`requirements.txt`](requirements.txt)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/northwind-analytics-dashboard.git
   cd northwind-analytics-dashboard
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure data file is present**:
   - Place `northwind_flat.csv` in the same directory as the script
   - The CSV should contain the Northwind dataset with columns like OrderID, OrderDate, CustomerID, etc.

## ▶️ Running the Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run st_deploy.py
   ```

2. **Access the dashboard**:
   - Open your web browser and navigate to the URL displayed (typically `http://localhost:8501`)
   - Use the sidebar to select different analytics modules
   - Apply filters for date ranges and countries as needed

## 📊 Data Requirements

The dashboard requires a CSV file named `northwind_flat.csv` with the following key columns:
- Order information: `OrderID`, `OrderDate`, `OrderStatus`
- Customer data: `CustomerID`, `CustomerName`, `CustomerCountry`
- Product details: `ProductID`, `ProductName`, `Category`
- Financial data: `LineTotal`, `ShippingFee`, `Taxes`
- Logistics: `ShipDate`, `ShipperName`, `FulfillmentTime`
- Employee data: `EmployeeID`, `EmployeeName`, `EmployeeJobTitle`
- Supplier information: `SupplierID`, `SupplierName`

## 🔧 Configuration

### Optional ML Features
- The dashboard includes machine learning capabilities for customer clustering and churn prediction
- If `scikit-learn` is not installed, these features will gracefully degrade with a warning message
- For full functionality, ensure `scikit-learn>=1.3.0` is installed

### Performance Optimization
- The app uses Streamlit's caching (`@st.cache_data`) for optimal performance
- Large datasets are processed efficiently with pandas operations

## 📁 Project Structure

```
northwind-analytics-dashboard/
│
├── st_deploy.py              # Main Streamlit application
├── requirements.txt          # Python dependencies
├── northwind_flat.csv        # Dataset (not included in repo)
├── README.md                 # Project documentation
└── .gitignore               # Git ignore file
```

## 🎯 Key Analytics Insights

### Customer Intelligence
- **RFM Segmentation**: Automatically categorizes customers into Champions, Loyal, At-Risk, etc.
- **Churn Prediction**: Identifies customers likely to stop purchasing within 90 days
- **Cohort Analysis**: Tracks retention rates by customer acquisition month

### Sales Performance
- **ABC Classification**: Prioritizes products based on revenue contribution (A=80%, B=15%, C=5%)
- **Regional Analysis**: Identifies top-performing markets and growth opportunities
- **Trend Forecasting**: Predicts future sales patterns using historical data

### Inventory Management
- **Demand Forecasting**: Uses moving averages to predict product demand
- **Stock Optimization**: Identifies slow-moving inventory and recommends actions
- **Reorder Alerts**: Automatically flags products below optimal stock levels

### Logistics & Delivery
- **Performance Metrics**: Tracks on-time delivery rates and fulfillment times
- **Cost Analysis**: Identifies shipping cost optimization opportunities
- **Shipper Evaluation**: Compares shipping partners by speed, cost, and reliability

### Workforce Analytics
- **Performance Scoring**: Measures employee productivity and contribution
- **Role Optimization**: Identifies strengths and gaps in job positions
- **Retention Analysis**: Tracks employee tenure and performance trends

### Supplier Performance
- **Quality Scoring**: Comprehensive supplier evaluation based on multiple factors
- **Procurement Optimization**: Identifies best-value suppliers and concentration risks
- **Relationship Tracking**: Monitors supplier performance over time

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to improve the dashboard.

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and test thoroughly
4. Commit your changes: `git commit -m 'Add amazing feature'`
5. Push to the branch: `git push origin feature/amazing-feature`
6. Open a Pull Request

### Guidelines
- Follow PEP 8 style guidelines for Python code
- Add docstrings to new functions
- Update README.md for any new features
- Test your changes with different data scenarios

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Northwind Database**: Sample database provided by Microsoft for demonstration purposes
- **Streamlit**: Amazing framework for building data applications
- **Plotly**: Powerful visualization library for interactive charts
- **Scikit-learn**: Machine learning library for advanced analytics

## 📞 Support

For questions or support:
- Create an issue in the GitHub repository
- Check the code comments for implementation details
- Review the Streamlit documentation for customization options

---

**Developed by**: James Andrew Dorado  
**Powered by**: Streamlit, Plotly, Pandas, and Scikit-learn  
**Data Source**: Northwind Traders Sample Database

⭐ Star this repository if you find it useful! 🚀</content>
<parameter name="filePath">d:\User\Downloads\DiaTrack\DFU_Healing_ML\DiaTrack_DFU_Detection_Models\DS_Class\SHEM_BA\README.md
