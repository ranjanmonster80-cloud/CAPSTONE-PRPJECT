import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Crypto Analytics Pro", layout="wide")

@st.cache_data
def load_data():
 df = pd.read_csv('cleaned_crypto_data.csv')
 return df

df = load_data()

st.sidebar.title("Dashboard Controls")
user_role = st.sidebar.radio("Access Level (RLS)", ["Standard", "Premium"])
category_filter = st.sidebar.multiselect(
    "Filter by Tier:", 
    options=df['market_cap_category'].unique(),
    default=df['market_cap_category'].unique())

mask = df['market_cap_category'].isin(category_filter)
filtered_df = df[mask].copy()

st.title("Crypto Market Intelligence Dashboard")
st.markdown("---")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Assets", len(filtered_df))
kpi2.metric("Avg Market Cap", f"${filtered_df['market_cap'].mean():,.0f}")
kpi3.metric("Max Price", f"${filtered_df['current_price'].max():,.2f}")
kpi4.metric("Avg Vol/MC Ratio", f"{filtered_df['vol_mc_ratio'].mean():.4f}")

# Row 1: Market Hierarchy & Distribution 
st.markdown("### 🏛️ Market Structure & Distribution")
col1, col2 = st.columns(2)

with col1:
    fig_tree = px.treemap(
        filtered_df.nlargest(50, 'market_cap'), 
        path=['market_cap_category', 'name'], 
        values='market_cap',
        color='market_cap',
        color_continuous_scale='RdBu',
        title="Top 50 Assets Hierarchy (Treemap)")
    st.plotly_chart(fig_tree, use_container_width=True)

with col2:
    fig_pie = px.pie(
        filtered_df, 
        names='market_cap_category', 
        title="Market Cap Tier Distribution",
        hole=0.4,color_discrete_sequence=px.colors.sequential.Tealgrn)
    st.plotly_chart(fig_pie, use_container_width=True)

#Row 2: Correlation & Volume Dynamics 
st.markdown("Correlation & Trading Dynamics")
col3, col4 = st.columns(2)

with col3:
    corr = filtered_df[['current_price', 'market_cap', 'total_volume', 'vol_mc_ratio']].corr()
    fig_heat = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='Viridis',title="Feature Correlation Heatmap")
    st.plotly_chart(fig_heat, use_container_width=True)

with col4:
    fig_scatter = px.scatter(
        filtered_df, 
        x="market_cap", y="total_volume", 
        size="current_price", color="market_cap_category",
        log_x=True, log_y=True,
        hover_name="name",
        title="Volume vs. Market Cap (Log Scale)",
        trendline="ols" if user_role == "Premium" else None)
    st.plotly_chart(fig_scatter, use_container_width=True)

# Row 3: AI Clustering (Model Outcome) 
st.markdown("AI Market Segmentation")
X = filtered_df[['log_current_price', 'log_market_cap', 'vol_mc_ratio']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
filtered_df['Cluster'] = kmeans.labels_.astype(str)

fig_cluster = px.scatter_3d(filtered_df, 
    x='log_market_cap', y='log_current_price', z='vol_mc_ratio',
    color='Cluster', hover_name='name',title="3D AI Clustering Visualization")
st.plotly_chart(fig_cluster, use_container_width=True)

#Footer: Data View & Drill Through 
with st.expander("🔍 View Raw Data & Search"):
    search_query = st.text_input("Search for a specific coin:")
    if search_query:
        st.dataframe(filtered_df[filtered_df['name'].str.contains(search_query, case=False)])
    else:
        st.dataframe(filtered_df.sort_values(by='market_cap', ascending=False))