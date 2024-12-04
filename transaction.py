import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load Main Data
@st.cache
def load_data():
    try:
        data = pd.read_excel("C:/Users/gigik/OneDrive/Desktop/combined_data_with_predictions.xlsx")
        transaction_data = pd.read_excel("C:/Users/gigik/OneDrive/Desktop/df (1).xlsx")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

    # Merge data
    data = pd.merge(
        data,
        transaction_data[['CUSTOMERNAME', 'COUNTRYNAME', 'YEAR', 'AMOUNT', 'ITEMGROUPDESCRIPTION']],
        on='CUSTOMERNAME',
        how='left'
    )

    # Fill missing years
    all_years = pd.DataFrame({'YEAR': range(2020, 2024 + 1)})
    data = data.merge(all_years, on='YEAR', how='outer').fillna({'AMOUNT': 0})

    return data, transaction_data

data, transaction_data = load_data()
if data is None:
    st.stop()

# Sidebar Filters
st.sidebar.header("Filters")
segment_filter = st.sidebar.selectbox("Select Segment:", ["All"] + data['Segment'].dropna().unique().tolist())
country_filter = st.sidebar.selectbox("Select Country:", ["All"] + data['COUNTRYNAME'].dropna().unique().tolist())
item_group_filter = st.sidebar.selectbox("Select Item Group:", ["All"] + data['ITEMGROUPDESCRIPTION'].dropna().unique().tolist())
customer_filter = st.sidebar.selectbox("Select Customer:", ["All"] + data['CUSTOMERNAME'].dropna().unique().tolist())
metric_filter = st.sidebar.selectbox("Select Metric:", ["Sales", "Refunds"])

# Filter Data
filtered_data = data.copy()
if segment_filter != "All":
    filtered_data = filtered_data[filtered_data['Segment'] == segment_filter]
if country_filter != "All":
    filtered_data = filtered_data[filtered_data['COUNTRYNAME'] == country_filter]
if item_group_filter != "All":
    filtered_data = filtered_data[filtered_data['ITEMGROUPDESCRIPTION'] == item_group_filter]
if customer_filter != "All":
    filtered_data = filtered_data[filtered_data['CUSTOMERNAME'] == customer_filter]

# Add Metrics
filtered_data['Sales'] = filtered_data['AMOUNT'].apply(lambda x: x if x > 0 else 0)
filtered_data['Refunds'] = filtered_data['AMOUNT'].apply(lambda x: -x if x < 0 else 0)

# Display Metrics
st.header("Key Metrics")
st.metric("Total Customers", filtered_data['CUSTOMERNAME'].nunique())
st.metric("Total Sales", f"{filtered_data['Sales'].sum():,.2f} AED")
st.metric("Total Refunds", f"{filtered_data['Refunds'].sum():,.2f} AED")

# Trends Chart
st.header(f"{metric_filter} Trends")
trends = filtered_data.groupby('YEAR')[metric_filter].sum().reset_index()
fig = px.line(trends, x='YEAR', y=metric_filter, title=f"{metric_filter} Trends (2020-2024)")
st.plotly_chart(fig)

# Download Button
st.download_button(
    label="Download Filtered Data",
    data=filtered_data.to_csv(index=False),
    file_name='filtered_data.csv',
    mime='text/csv'
)


