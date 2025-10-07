
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="AquaFlow Pro - Water Monitoring System",
    page_icon="💧",
    layout="wide"
)

# --- SIDEBAR NAVIGATION (Menu Bar Icon Implementation) ---
# Streamlit automatically provides a collapse icon for the sidebar
# We use st.session_state to control which tab is active based on the sidebar selection.

if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "🏠 Dashboard" # Initialize with full title

with st.sidebar:
    # FIX 1: Replaced deprecated use_column_width with use_container_width
    st.image("https://placehold.co/150x50/667eea/ffffff?text=AquaFlow+Pro", caption="System Status", use_container_width=True)
    st.markdown("---")
    st.subheader("Main Navigation")
    
    # Define a radio selection for the sidebar menu
    # Using icons and titles for better menu visual as requested
    menu_selection = st.radio(
        "Navigate Pages",
        options=["🏠 Dashboard", "📁 Data Upload", "⚙️ Settings", "❓ FAQ", "📖 How to Use", "ℹ️ About"],
        index=["🏠 Dashboard", "📁 Data Upload", "⚙️ Settings", "❓ FAQ", "📖 How to Use", "ℹ️ About"].index(st.session_state.current_tab),
        key="sidebar_menu"
    )
    
    # Update session state based on sidebar selection
    st.session_state.current_tab = menu_selection
    
    st.markdown("---")
    st.info(f"Active Station: Central Plant")
    st.button("Logout", type="secondary", use_container_width=True)

# (Code before CSS)

# Beautiful CSS styling with responsive menu bar (Streamlit tabs)
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    .main {
        font-family: 'Inter', sans-serif;
    }

    /* Header styling - ENHANCED FOR INTERACTIVITY */
    .big-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem; /* Increased padding */
        border-radius: 20px; /* More rounded */
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.5); /* Stronger initial shadow */
        transition: all 0.5s ease-in-out; /* Add transition for hover effect */
        /* Add subtle animation */
        animation: gradient-shift 15s ease infinite; 
        background-size: 200% 200%;
        cursor: pointer;
    }

    .big-header:hover {
        transform: scale(1.02); /* Slight lift */
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.7); /* Brighter glow */
    }

    .big-header h1 {
        font-size: 3rem; /* Larger title */
        font-weight: 800; /* Bolder title */
        margin: 0;
        letter-spacing: 2px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .big-header p {
        font-size: 1.2rem;
        font-weight: 400;
        opacity: 0.9;
    }

    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* --- CUSTOM RESPONSIVE MENU BAR STYLING (Streamlit Tabs) --- */
    /* Target the div containing the tab list */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px; /* Space between tabs */
        margin-bottom: 20px;
        background: #f8f8f8; /* Light background for the bar */
        border-radius: 12px;
        padding: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        overflow-x: auto; /* Allows horizontal scrolling if many tabs don't fit */
        flex-wrap: nowrap; /* Prevent wrapping in desktop mode */
    }

    /* Style for individual tab buttons (not active) */
    .stTabs button {
        background-color: transparent !important;
        color: #667eea; /* Primary color for text */
        border-radius: 8px;
        padding: 10px 15px;
        transition: all 0.3s ease;
        font-weight: 600;
        border: none !important;
        min-width: fit-content; /* Ensure text doesn't wrap awkwardly */
    }

    /* Style for the active tab button */
    .stTabs [aria-selected="true"] {
        background-color: #667eea !important; /* Primary color background */
        color: white !important;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }

    /* Hover effect */
    .stTabs button:hover:not([aria-selected="true"]) {
        background-color: #eef1ff !important; /* Light background on hover */
        color: #4a5dcb !important;
        transform: translateY(-2px);
    }

    /* Mobile Responsiveness for Menu Bar */
    @media (max-width: 768px) {
        .stTabs [data-baseweb="tab-list"] {
            flex-wrap: wrap; /* Allow wrapping on smaller screens */
            gap: 5px;
            padding: 5px;
        }
        .stTabs button {
            text-align: center;
            padding: 8px 10px;
            flex-grow: 1; /* Make buttons expand somewhat */
            font-size: 0.9rem;
        }
    }
    /* --- END MENU BAR STYLING --- */


    /* Status cards */
    .status-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e9ecef;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: left;
        margin: 1rem 0;
        transition: transform 0.3s ease;
        height: 100%;
    }



    /* Alert boxes */
    .alert-success {
        background: linear-gradient(145deg, #d4edda, #c3e6cb);
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #28a745;
        color: #155724;
        margin: 1rem 0;
    }

    .alert-warning {
        background: linear-gradient(145deg, #fff3cd, #ffeaa7);
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #ffc107;
        color: #856404;
        margin: 1rem 0;
    }

    .alert-info {
        background: linear-gradient(145deg, #e3f2fd, #bbdefb);
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #2196f3;
        color: #1976d2;
        margin: 1rem 0;
    }

    /* Upload zone */
    .upload-zone {
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(145deg, #f8f9ff, #e3f2fd);
        margin: 1rem 0;
    }

    /* FAQ styling */
    .faq-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }

    .faq-box h4 {
        color: #667eea;
        margin-top: 0;
    }

    /* Feature card */
    .feature-card {
        background: linear-gradient(145deg, #ffffff, #f5f5f5);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }

    .feature-card:hover {
        transform: translateY(-5px);
    }

    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="big-header">
    <h1>💧 AquaFlow Pro</h1>
    <p>Advanced Water Quality Monitoring & Analytics System</p>
</div>
""", unsafe_allow_html=True)

# --- CONDITIONAL TAB DISPLAY LOGIC ---

# We only display the content of the selected tab based on the session state.
# FIX 2: Removed st.tabs completely as it conflicted with the sidebar navigation logic.
# The previous st.tabs approach caused the TypeError (unexpected keyword argument 'default_index').
# Now, we use the session state to control which content section is rendered.

tab_titles = ["🏠 Dashboard", "📁 Data Upload", "⚙️ Settings", "❓ FAQ", "📖 How to Use", "ℹ️ About"]
current_tab = st.session_state.current_tab

# Function to generate sensor data
def get_sensor_reading(sensor_type, status='normal'):
    """Generate realistic sensor readings"""
    base_values = {
        'temperature': {'normal': 72, 'warning': 85, 'critical': 95},
        'pressure': {'normal': 45, 'warning': 35, 'critical': 25},
        'flow_rate': {'normal': 150, 'warning': 120, 'critical': 80},
        'ph_level': {'normal': 7.2, 'warning': 6.5, 'critical': 5.5}
    }

    base = base_values.get(sensor_type, {'normal': 50, 'warning': 40, 'critical': 30})[status]
    return base + np.random.normal(0, 2)


# --- TAB CONTENT DISPLAY LOGIC ---

if current_tab == "🏠 Dashboard":
    st.header("🏠 Dashboard - Real-Time Monitoring")

    # Quick stats row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("System Uptime", "99.2%", "0.3%")
    with col2:
        st.metric("Active Sensors", "24", "2")
    with col3:
        st.metric("Alerts Today", "3", "-1")
    with col4:
        st.metric("Data Points", "15,847", "1,247")

    st.markdown("---")

    # Equipment selection
    col1, col2 = st.columns([2, 1])

    with col1:
        station = st.selectbox(
            "🏭 Select Water Treatment Station:",
            ["Central Treatment Plant", "Industrial Zone A", "Residential District B", "Municipal Complex C"]
        )

    with col2:
        if st.button("🔄 Refresh Data"):
            st.success("✅ Data refreshed!")

    # Current sensor readings
    st.subheader(f"📊 Live Sensor Readings - {station}")

    # Generate current readings
    temp = get_sensor_reading('temperature')
    pressure = get_sensor_reading('pressure')
    flow = get_sensor_reading('flow_rate')
    ph = get_sensor_reading('ph_level')

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🌡️ Temperature", f"{temp:.1f}°C", f"{np.random.uniform(-1, 1):.1f}°C")
    with col2:
        st.metric("📊 Pressure", f"{pressure:.1f} PSI", f"{np.random.uniform(-2, 2):.1f} PSI")
    with col3:
        st.metric("🌊 Flow Rate", f"{flow:.0f} L/min", f"{np.random.uniform(-5, 5):.0f} L/min")
    with col4:
        st.metric("⚗️ pH Level", f"{ph:.2f}", f"{np.random.uniform(-0.1, 0.1):.2f}")

    # System status
    if temp > 80:
        st.markdown("""
        <div class="alert-warning">
            <strong>⚠️ Warning:</strong> Temperature is elevated. Monitor closely.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-success">
            <strong>✅ Status:</strong> All systems operating normally.
        </div>
        """, unsafe_allow_html=True)

    # Real-time chart
    st.subheader("📈 24-Hour Trends")

    # Generate sample data
    hours = list(range(24))
    temp_data = [70 + 5*np.sin(h/4) + np.random.normal(0, 2) for h in hours]
    pressure_data = [45 + 3*np.cos(h/6) + np.random.normal(0, 1.5) for h in hours]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours, y=temp_data, name='Temperature', line=dict(color='#ff6b6b')))
    fig.add_trace(go.Scatter(x=hours, y=pressure_data, name='Pressure', line=dict(color='#4ecdc4')))

    fig.update_layout(
        title="Temperature & Pressure Trends (Last 24 Hours)",
        xaxis_title="Hours Ago",
        yaxis_title="Values",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # Quick actions
    st.subheader("🎛️ Quick Actions")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🛑 Emergency Stop", key="stop_dash"):
            st.error("🛑 Emergency stop activated!")

    with col2:
        if st.button("📞 Call Technician", key="call_dash"):
            st.success("📞 Technician notified!")

    with col3:
        if st.button("📝 Log Incident", key="log_dash"):
            st.info("📝 Incident logged!")

    with col4:
        if st.button("📊 Generate Report", key="report_dash"):
            st.info("📊 Report generated!")

elif current_tab == "📁 Data Upload":
    st.header("📁 Data Upload & Analysis")

    st.markdown("""
    <div class="upload-zone">
        <h3>🚀 Upload Your Sensor Data</h3>
        <p>Upload CSV or Excel files for comprehensive analysis</p>
        <p><strong>Supported formats:</strong> .csv, .xlsx, .xls (Max 200MB)</p>
    </div>
    """, unsafe_allow_html=True)

    # File upload widget
    uploaded_file = st.file_uploader(
        "Choose your data file",
        type=['csv', 'xlsx', 'xls'],
        help="Select CSV or Excel files containing sensor data"
    )

    if uploaded_file is not None:
        try:
            # Show upload success
            file_size = len(uploaded_file.getvalue()) / 1024 / 1024
            st.markdown(f"""
            <div class="alert-success">
                <h4>✅ Upload Successful!</h4>
                <p><strong>File:</strong> {uploaded_file.name}</p>
                <p><strong>Size:</strong> {file_size:.2f} MB</p>
                <p><strong>Time:</strong> {datetime.now().strftime('%H:%M:%S')}</p>
            </div>
            """, unsafe_allow_html=True)

            # Read the file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # Basic file info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", f"{len(df):,}")
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            with col4:
                st.metric("Memory Usage", f"{df.memory_usage().sum() / 1024:.1f} KB")

            # Data preview
            st.subheader("👀 Data Preview")
            st.dataframe(df.head(), use_container_width=True)

            # Analysis options
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            if numeric_cols:
                st.subheader("📊 Analysis Options")

                col1, col2 = st.columns(2)
                with col1:
                    selected_cols = st.multiselect("Select columns to analyze:", numeric_cols, default=numeric_cols[:3])
                with col2:
                    chart_type = st.selectbox("Chart type:", ["Line Chart", "Box Plot", "Histogram"])

                if selected_cols:
                    if chart_type == "Line Chart":
                        fig = px.line(df, y=selected_cols, title="Data Visualization")
                        st.plotly_chart(fig, use_container_width=True)

                    elif chart_type == "Box Plot":
                        fig = go.Figure()
                        for col in selected_cols:
                            fig.add_trace(go.Box(y=df[col], name=col))
                        fig.update_layout(title="Distribution Analysis")
                        st.plotly_chart(fig, use_container_width=True)

                    elif chart_type == "Histogram":
                        fig = make_subplots(rows=1, cols=len(selected_cols), subplot_titles=selected_cols)
                        for i, col in enumerate(selected_cols):
                            fig.add_trace(go.Histogram(x=df[col], name=col), row=1, col=i+1)
                        fig.update_layout(title="Value Distributions")
                        st.plotly_chart(fig, use_container_width=True)

                # Statistical summary
                st.subheader("📈 Statistical Summary")
                st.dataframe(df[selected_cols].describe(), use_container_width=True)

                # Anomaly detection
                if st.button("🔍 Run Anomaly Detection", type="primary"):
                    with st.spinner("Analyzing data for anomalies..."):
                        time.sleep(2)  # Simulate processing

                        anomaly_results = []
                        for col in selected_cols:
                            Q1 = df[col].quantile(0.25)
                            Q3 = df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower = Q1 - 1.5 * IQR
                            upper = Q3 + 1.5 * IQR
                            anomalies = df[(df[col] < lower) | (df[col] > upper)]

                            anomaly_results.append({
                                'Column': col,
                                'Anomalies': len(anomalies),
                                'Percentage': f"{(len(anomalies)/len(df))*100:.2f}%"
                            })

                        results_df = pd.DataFrame(anomaly_results)
                        st.subheader("🎯 Anomaly Detection Results")
                        st.dataframe(results_df, use_container_width=True, hide_index=True)

                        total_anomalies = sum([r['Anomalies'] for r in anomaly_results])
                        if total_anomalies > 0:
                            st.markdown(f"""
                            <div class="alert-warning">
                                <strong>⚠️ Found {total_anomalies} anomalies!</strong> Review flagged data points.
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="alert-success">
                                <strong>✅ No anomalies detected.</strong> Data appears normal.
                            </div>
                            """, unsafe_allow_html=True)

        except Exception as e:
            st.markdown(f"""
            <div class="alert-warning">
                <h4>❌ Error Processing File</h4>
                <p><strong>Error:</strong> {str(e)}</p>
                <p>Please check your file format and try again.</p>
            </div>
            """, unsafe_allow_html=True)

    else:
        # Show example data format
        st.subheader("📄 Expected Data Format")
        example_data = {
            'timestamp': ['2025-10-04 10:00:00', '2025-10-04 10:01:00', '2025-10-04 10:02:00'],
            'temperature': [72.5, 73.1, 72.8],
            'pressure': [45.2, 44.8, 45.5],
            'flow_rate': [152, 148, 155],
            'ph_level': [7.2, 7.1, 7.3]
        }
        example_df = pd.DataFrame(example_data)
        st.dataframe(example_df, use_container_width=True, hide_index=True)

elif current_tab == "⚙️ Settings":
    st.header("⚙️ System Configuration")

    st.subheader("🚨 Alert Thresholds")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🌡️ Temperature Settings")
        temp_warning = st.number_input("Warning Level (°C)", value=80.0)
        temp_critical = st.number_input("Critical Level (°C)", value=85.0)

        st.markdown("#### 📊 Pressure Settings")
        pressure_warning = st.number_input("Low Pressure Warning (PSI)", value=35.0)
        pressure_critical = st.number_input("Low Pressure Critical (PSI)", value=30.0)

    with col2:
        st.markdown("#### 🌊 Flow Rate Settings")
        flow_warning = st.number_input("Low Flow Warning (L/min)", value=120.0)
        flow_critical = st.number_input("Low Flow Critical (L/min)", value=100.0)

        st.markdown("#### ⚗️ pH Level Settings")
        ph_min = st.number_input("pH Minimum", value=6.5)
        ph_max = st.number_input("pH Maximum", value=8.5)

    st.subheader("📢 Notifications")

    col1, col2 = st.columns(2)

    with col1:
        email_enabled = st.checkbox("Email Notifications", value=True)
        if email_enabled:
            email_address = st.text_input("Email Address", "admin@aquaflow.com")

    with col2:
        sms_enabled = st.checkbox("SMS Notifications", value=False)
        if sms_enabled:
            phone_number = st.text_input("Phone Number", "+1-555-0123")

    st.subheader("⚙️ System Settings")

    col1, col2 = st.columns(2)

    with col1:
        refresh_interval = st.selectbox("Data Refresh Interval", ["30 seconds", "1 minute", "5 minutes"])
        data_retention = st.selectbox("Data Retention", ["1 month", "3 months", "1 year"])

    with col2:
        backup_enabled = st.checkbox("Automatic Backups", value=True)
        if backup_enabled:
            backup_frequency = st.selectbox("Backup Frequency", ["Daily", "Weekly", "Monthly"])

    if st.button("💾 Save All Settings", type="primary"):
        st.success("✅ Settings saved successfully!")
        st.balloons()

elif current_tab == "❓ FAQ":
    st.header("❓ Frequently Asked Questions")

    faqs = [
        {
            "q": "🚀 What is AquaFlow Pro?",
            "a": "AquaFlow Pro is an advanced water quality monitoring system that uses AI to detect anomalies, predict equipment failures, and optimize water treatment processes in real-time."
        },
        {
            "q": "📊 What sensors does it monitor?",
            "a": "Our system monitors temperature, pressure, flow rate, pH levels, turbidity, chlorine levels, dissolved oxygen, and other critical water quality parameters with real-time alerts."
        },
        {
            "q": "🔍 How does anomaly detection work?",
            "a": "We use advanced machine learning algorithms including TimeGAN to learn normal operating patterns and detect deviations. The AI continuously improves by learning from historical data."
        },
        {
            "q": "📁 What file formats can I upload?",
            "a": "You can upload CSV files (.csv) and Excel files (.xlsx, .xls). Maximum file size is 200MB. Ensure your data includes timestamps and numerical sensor readings."
        },
        {
            "q": "🚨 How do I configure alerts?",
            "a": "Go to Settings → Alert Thresholds to set warning and critical levels for each sensor. Configure email/SMS notifications and choose alert frequency preferences."
        },
        {
            "q": "📈 Can I export data and reports?",
            "a": "Yes! Export sensor data as CSV, generate PDF reports, and download analysis results. All data can be exported for external analysis or compliance reporting."
        },
        {
            "q": "🔧 What maintenance features are included?",
            "a": "Predictive maintenance scheduling, equipment health scoring, failure probability analysis, maintenance history tracking, and automated work order generation."
        },
        {
            "q": "🔒 Is my data secure?",
            "a": "Yes, we use enterprise-grade security with AES-256 encryption, multi-factor authentication, and comply with industry standards including ISO 27001 and GDPR."
        },
        {
            "q": "📱 Is there mobile support?",
            "a": "AquaFlow Pro is fully responsive and works on all devices. A dedicated mobile app with push notifications is currently in development."
        },
        {
            "q": "💰 What are the pricing options?",
            "a": "We offer flexible pricing based on sensors and features. Contact sales@aquaflow.com for quotes. Free trials and demos are available for evaluation."
        }
    ]

    for faq in faqs:
        st.markdown(f"""
        <div class="faq-box">
            <h4>{faq['q']}</h4>
            <p>{faq['a']}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="alert-info">
        <h4>💬 Need More Help?</h4>
        <p><strong>📧 Email:</strong> support@aquaflow.com</p>
        <p><strong>📞 Phone:</strong> +1-800-AQUA-HELP</p>
        <p><strong>💬 Live Chat:</strong> Available 24/7 on our website</p>
    </div>
    """, unsafe_allow_html=True)

elif current_tab == "📖 How to Use":
    st.header("📖 How to Use AquaFlow Pro")

    st.markdown("""
    <div class="alert-info">
        <h3>🎯 Complete User Guide</h3>
        <p>Learn how to maximize the power of AquaFlow Pro with this comprehensive guide.</p>
    </div>
    """, unsafe_allow_html=True)

    # Create subtabs for different sections
    guide_tab1, guide_tab2, guide_tab3, guide_tab4, guide_tab5 = st.tabs(["🏁 Getting Started", "📊 Dashboard", "📁 Upload Data", "⚙️ Settings", "🔍 Analytics"])

    with guide_tab1:
        st.markdown("""
        ### 🏁 Getting Started with AquaFlow Pro

        #### **Step 1: System Overview**
        - AquaFlow Pro monitors water treatment systems in real-time
        - Use the sidebar (menu icon) to navigate between different features
        - The Dashboard provides a comprehensive view of system health

        #### **Step 2: First Time Setup**
        1. **Select Your Station**: Choose your water treatment station from the dropdown
        2. **Review Live Data**: Check current sensor readings for any anomalies
        3. **Configure Alerts**: Set up thresholds in the Settings tab
        4. **Test Notifications**: Verify email/SMS alerts are working

        #### **Step 3: Understanding Status Indicators**
        - 🟢 **Green Metrics**: Normal operation, all systems healthy
        - 🟡 **Yellow Alerts**: Warning condition, monitor situation
        - 🔴 **Red Alerts**: Critical condition, immediate action required

        #### **Step 4: Daily Operations**
        - Check the dashboard every morning for overnight issues
        - Review trends to identify gradual changes
        - Use quick actions for emergency responses
        - Generate reports for management and compliance
        """)

    with guide_tab2:
        st.markdown("""
        ### 📊 Using the Dashboard

        #### **Real-Time Monitoring Panel**
        - **Live Metrics**: Temperature, pressure, flow rate, pH levels updated every 30 seconds
        - **Status Cards**: Color-coded indicators show system health at a glance
        - **Trend Charts**: 24-hour historical data helps identify patterns
        - **Delta Values**: Show changes from previous readings

        #### **Equipment Selection**
        - Use the dropdown to switch between different treatment stations
        - Each station has its own sensor network and thresholds
        - Click "Refresh Data" to manually update all readings

        #### **Understanding the Charts**
        - **Temperature Chart**: Red line shows thermal trends
        - **Pressure Chart**: Blue line indicates hydraulic performance
        - **Hover over points**: Get exact values and timestamps
        - **Zoom and pan**: Click and drag to examine specific time periods

        #### **Quick Action Buttons**
        - **🛑 Emergency Stop**: Immediately shuts down pumps and valves
        - **📞 Call Technician**: Sends alert to maintenance team
        - **📝 Log Incident**: Creates timestamped incident report
        - **📊 Generate Report**: Produces comprehensive status summary

        #### **Alert Interpretation**
        - **Green Status**: "All systems operating normally"
        - **Yellow Warning**: "Monitor closely - parameter deviation detected"
        - **Red Critical**: "Immediate action required - system at risk"
        """)

    with guide_tab3:
        st.markdown("""
        ### 📁 Data Upload & Analysis Guide

        #### **Supported File Formats**
        - **CSV Files**: Comma-separated values (.csv)
        - **Excel Files**: .xlsx and .xls formats
        - **Maximum Size**: 200MB per file
        - **Encoding**: UTF-8 recommended for CSV files

        #### **Data Requirements**
        Your uploaded file should contain:
        - **Timestamp Column**: Date/time in standard format
        - **Sensor Columns**: Numerical values for each parameter
        - **Headers**: Clear column names describing each sensor
        - **Consistent Units**: Same units throughout the dataset

        #### **Upload Process**
        1. **Click Browse**: Select your data file
        2. **Wait for Processing**: System validates and loads data
        3. **Review Preview**: Check data structure and quality
        4. **Select Columns**: Choose which sensors to analyze
        5. **Pick Visualization**: Line charts, box plots, or histograms
        6. **Run Analysis**: Get statistical summaries and insights

        #### **Analysis Features**
        - **Statistical Summary**: Mean, median, standard deviation, min/max
        - **Visualization Options**: Multiple chart types for different insights
        - **Anomaly Detection**: AI-powered outlier identification
        - **Correlation Analysis**: Find relationships between sensors
        - ****
        - **Export Results**: Download analysis for further use

        #### **Anomaly Detection**
        - Uses IQR (Interquartile Range) method by default
        - Identifies values outside 1.5 × IQR range
        - Results show count and percentage of anomalies
        - Color-coded severity: Green (normal), Yellow (some), Red (many)
        """)

    with guide_tab4:
        st.markdown("""
        ### ⚙️ Configuration & Settings Guide

        #### **Alert Threshold Configuration**
        - **Warning Levels**: Early notification when parameters deviate
        - **Critical Levels**: Immediate alerts for dangerous conditions
        - **Custom Ranges**: Set different limits for each treatment plant
        - **Hysteresis**: Prevents alert flooding from fluctuating values

        #### **Temperature Settings**
        - **Normal Range**: Typically 65-75°C for most processes
        - **Warning Threshold**: Usually 5-10°C above normal
        - **Critical Threshold**: Temperature that risks equipment damage

        #### **Pressure Settings**
        - **Normal Operating Pressure**: Varies by system design
        - **Low Pressure Warning**: Indicates potential pump issues
        - **Critical Low Pressure**: Risk of cavitation or system failure

        #### **Notification Setup**
        - **Email Notifications**: Primary and secondary addresses
        - **SMS Alerts**: For critical issues requiring immediate response
        - **Alert Frequency**: Cooldown periods to prevent spam
        - **Escalation Rules**: Automatic escalation for unacknowledged alerts

        #### **System Preferences**
        - **Refresh Interval**: How often dashboard updates (30s-10min)
        - **Data Retention**: How long historical data is stored
        - **Backup Settings**: Automatic backup frequency and location
        - **User Access**: Role-based permissions for different operators

        #### **Saving and Backup**
        - Always click "Save All Settings" after making changes
        - Use "Reset to Defaults" to restore factory settings
        - Export configurations for backup or sharing between sites
        """)

    with guide_tab5:
        st.markdown("""
        ### 🔍 Advanced Analytics Guide

        #### **Trend Analysis**
        - **Long-term Patterns**: Identify seasonal variations and drift
        - **Performance Degradation**: Spot gradual equipment wear
        - **Correlation Analysis**: Find relationships between parameters
        - **Statistical Trends**: Regression analysis and forecasting

        #### **Anomaly Detection Methods**
        - **Statistical Methods**: Z-score, IQR, standard deviation
        - **Machine Learning**: TimeGAN, isolation forests, autoencoders
        - **Pattern Recognition**: Unusual sequences and behaviors
        - **Real-time Scoring**: Continuous anomaly probability assessment

        #### **Predictive Analytics**
        - **Equipment Health Scoring**: 0-100% health indicators
        - **Failure Prediction**: Probability of breakdown in next 30 days
        - **Maintenance Optimization**: Best times for scheduled maintenance
        - **Performance Forecasting**: Expected efficiency and output

        #### **Report Generation**
        - **Daily Reports**: Summary of key metrics and events
        - **Weekly Summaries**: Trend analysis and performance review
        - **Monthly Analysis**: Comprehensive system health assessment
        - **Custom Reports**: User-defined parameters and time ranges

        #### **Data Export Options**
        - **Raw Data**: CSV export of sensor readings
        - **Analysis Results**: Statistical summaries and insights
        - **Charts and Graphs**: PNG/PDF export of visualizations
        - **Compliance Reports**: Formatted reports for regulatory submission

        #### **Best Practices**
        - **Regular Monitoring**: Check dashboard daily for trends
        - **Threshold Tuning**: Adjust alerts based on operational experience
        - **Data Quality**: Ensure sensors are calibrated and clean
        - **Historical Analysis**: Use long-term data for optimization
        - **Team Training**: Ensure all operators understand the system
        """)

    st.markdown("""
    <div class="alert-success">
        <h4>🎓 Pro Tips for Maximum Effectiveness</h4>
        <ul>
            <li><strong>Start Conservative:</strong> Set tight thresholds initially, then adjust based on false positive rates</li>
            <li><strong>Use Historical Data:</strong> Upload past data to establish baseline performance patterns</li>
            <li><strong>Monitor Trends:</strong> Look for gradual changes that indicate developing problems</li>
            <li><strong>Validate Alerts:</strong> Always investigate alerts to improve system accuracy</li>
            <li><strong>Regular Calibration:</strong> Ensure sensors are properly maintained and calibrated</li>
            <li><strong>Document Changes:</strong> Keep records of system modifications and their effects</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

elif current_tab == "ℹ️ About":
    st.header("ℹ️ About AquaFlow Pro")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🤖 AI-Powered</h3>
            <p>Advanced machine learning algorithms for superior anomaly detection and predictive analytics</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Real-Time</h3>
            <p>Live monitoring with instant alerts and comprehensive analytical dashboards</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>🔧 Predictive</h3>
            <p>Proactive maintenance scheduling to prevent costly equipment failures</p>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("🎯 Our Mission")
    st.write("""
    AquaFlow Pro revolutionizes water management through cutting-edge AI technology. We believe that every
    drop of water matters, and our mission is to ensure the highest quality water treatment through
    intelligent monitoring, predictive analytics, and proactive maintenance strategies.
    """)

    st.subheader("💡 Key Features")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### ✅ **Core Capabilities**
        - 🔍 Real-time sensor monitoring
        - 🤖 AI-powered anomaly detection
        - 📈 Predictive maintenance scheduling
        - 📊 Interactive data visualization
        - 📁 CSV/Excel data upload and analysis
        - ⚙️ Customizable alert thresholds
        - 📱 Mobile-responsive interface
        - 🔒 Enterprise-grade security
        """)

    with col2:
        st.markdown("""
        #### 🏭 **Industries Served**
        - Municipal Water Treatment
        - Industrial Process Water
        - Power Generation Facilities
        - Chemical Manufacturing
        - Food & Beverage Processing
        - Pharmaceutical Production
        - Wastewater Treatment
        - Aquaculture Operations
        """)

    st.subheader("🛠️ Technical Specifications")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        #### 📊 **Performance**
        - Real-time latency: <100ms
        - Data retention: Up to 10 years
        - Concurrent users: 1000+
        - Sensors supported: Unlimited
        - File upload: Up to 200MB
        """)

    with col2:
        st.markdown("""
        #### 🔒 **Security**
        - Encryption: AES-256
        - Authentication: Multi-factor
        - Compliance: ISO 27001, SOC 2
        - Privacy: GDPR compliant
        - Auditing: Complete activity logs
        """)

    with col3:
        st.markdown("""
        #### 🌐 **Compatibility**
        - Browsers: Chrome, Firefox, Safari, Edge
        - Mobile: iOS, Android responsive
        - OS: Windows, Mac, Linux
        - Cloud: AWS, Azure, Google Cloud
        - Integration: REST API, webhooks
        """)

    st.subheader("📞 Contact Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        #### 🏢 **Sales Team**
        📧 sales@aquaflow.com
        📞 +1-800-AQUA-SALES
        💼 Schedule a demo
        """)

    with col2:
        st.markdown("""
        #### 🛠️ **Technical Support**
        📧 support@aquaflow.com
        📞 +1-800-AQUA-HELP
        💬 24/7 Live Chat
        """)

    with col3:
        st.markdown("""
        #### 📍 **Headquarters**
        456 Water Technology Blvd
        Innovation District
        Austin, TX 78701
        🌐 www.aquaflow.com
        """)

    st.markdown("""
    <div class="alert-info">
        <h4>📋 Version Information</h4>
        <p><strong>AquaFlow Pro:</strong> v2.0.1</p>
        <p><strong>Release Date:</strong> October 2025</p>
        <p><strong>License:</strong> Commercial Enterprise License</p>
        <p><strong>Last Updated:</strong> """ + datetime.now().strftime('%B %d, %Y') + """</p>
        <p><strong>Build:</strong> #2025.10.04</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("**💧 AquaFlow Pro v2.0** | Advanced Water Quality Monitoring System | © 2025 AquaTech Solutions")
