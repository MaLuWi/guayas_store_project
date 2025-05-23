import sys
import os
import datetime

import pandas as pd
import streamlit as st
import plotly.express as px

# Configure Streamlit layout and page title
st.set_page_config(layout="wide", page_title="Sales Forecasting Dashboard")
# set project’s root folder
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.data_utils import load_data
from model.model_utils import load_model, forecast_timeseries

@st.cache_data(show_spinner=False)
def get_data():
    df_stores, df_items, df_train = load_data()
    df_train['date'] = pd.to_datetime(df_train['date'])
    return df_stores, df_items, df_train

@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()
#main app entry
def main():
    st.title("🛒 Sales Forecasting Dashboard")

    # Load data and model
    df_stores, df_items, df_train = get_data()
    model = get_model()

    # Sidebar: user inputs for store, item, start date
    st.sidebar.header("Forecast Inputs")

    store_id = st.sidebar.selectbox(
        "Choose Store",
        sorted(df_stores['store_nbr'].unique())
    )
    # pick forecast start date
    start_date = st.sidebar.date_input(
        "Forecast Start Date",
        value=datetime.date(2014, 1, 1),
        min_value=df_train['date'].min().date(),
        max_value=df_train['date'].max().date()
    )

    # Compute total sales per item for the selected store
    item_sales = (
        df_train[
            (df_train['store_nbr'] == store_id) &
            (df_train['date'] <= pd.to_datetime(start_date))
        ]
        .groupby('item_nbr')['unit_sales']
        .sum()
        .reset_index()
    )

    # Filter out items with zero sales
    item_sales = item_sales[item_sales['unit_sales'] > 0]

    # Sort by total sales descending
    item_sales = item_sales.sort_values(by='unit_sales', ascending=False)

    # Final list of item_nbrs sorted by sales
    valid_items = item_sales['item_nbr'].tolist()

    item_id = st.sidebar.selectbox("Choose Item", valid_items)
    # how many days ahead to forecast
    horizon = st.sidebar.number_input(
        "Days to Forecast",
        min_value=1,
        max_value=60,
        value=14
    )

    # Run forecast
    if st.sidebar.button("Run Forecast"):
        with st.spinner("Running forecast…"):
            fc = forecast_timeseries(
                model=model,
                store_id=store_id,
                item_id=item_id,
                start_date=start_date,
                horizon=horizon,
                df_train=df_train,
                df_stores=df_stores,
                df_items=df_items
            )

        if fc.empty:
            st.error("No historical data available for that store/item.")
            return

        # actual sales in the prediction window
        actual = (
            df_train[
                (df_train.store_nbr == store_id) &
                (df_train.item_nbr == item_id) &
                (df_train.date >= pd.to_datetime(start_date)) &
                (df_train.date < pd.to_datetime(start_date) + pd.Timedelta(days=horizon))
            ]
            .groupby('date')['unit_sales']
            .sum()
            .reset_index()
            .rename(columns={'unit_sales': 'actual'})
        )

        # merge actuals and predictions
        df_plot = pd.merge(actual, fc, on='date', how='outer').sort_values('date')
        df_plot['prediction'] = df_plot['prediction'].ffill()

        # Plot the result
        fig = px.line(
            df_plot,
            x='date',
            y=['actual', 'prediction'],
            labels={'value': 'Unit Sales', 'variable': 'Series', 'date': 'Date'},
            title=f"Store {store_id} • Item {item_id}: Actual vs Predicted"
        )
        fig.update_layout(
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            xaxis_title="Date"
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.sidebar.info("Configure your inputs and click **Run Forecast**.")


if __name__ == '__main__':
    main()

