# app/main.py
import sys
import os
import datetime

import pandas as pd
import streamlit as st
import plotly.express as px

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.data_utils import load_data
from model.model_utils import load_model, forecast_timeseries

def main():
    st.set_page_config(layout="wide")
    st.title("Sales Forecasting Dashboard")

    # 1) Load data & model
    df_stores, df_items, df_train = load_data()
    df_train['date'] = pd.to_datetime(df_train['date'])
    model = load_model()

    # 2) Sidebar inputs
    st.sidebar.header("Forecast Inputs")

    store_id = st.sidebar.selectbox(
        "Choose Store",
        df_stores['store_nbr'].unique()
    )

    start_date = st.sidebar.date_input(
        "Forecast Start Date",
        value=datetime.date(2014, 1, 1),
        min_value=df_train['date'].min().date(),
        max_value=df_train['date'].max().date()
    )

    # filter items for that store AND up to the selected date
    valid_items = (
        df_train[
            (df_train.store_nbr == store_id) &
            (df_train.date <= pd.to_datetime(start_date))
        ]['item_nbr']
        .unique()
        .tolist()
    )

    if not valid_items:
        st.sidebar.error(
            f"No historical items for store {store_id} by {start_date}. "
            "Try an earlier start date."
        )
        st.stop()

    item_id = st.sidebar.selectbox("Choose Item", valid_items)

    horizon = st.sidebar.number_input(
        "Days to Forecast",
        min_value=1,
        max_value=60,
        value=30
    )

    # 3) Run forecast when button clicked
    if st.sidebar.button("Run Forecast"):
        # generate forecast frame
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

        # pull actuals for the same date window
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

        # merge forecast + actual
        df_plot = pd.merge(actual, fc, on='date', how='outer').sort_values('date')
        df_plot['prediction'] = df_plot['prediction'].fillna(method='ffill')

        # 4) Plotly interactive chart
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
