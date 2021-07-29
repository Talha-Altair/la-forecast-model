'''
Created on 

Course work: 

@author: Tact Team

Source:
    https://discuss.streamlit.io/t/how-do-i-change-the-app-title-and-favicon/1654/5

    https://docs.streamlit.io/en/stable/api.html#streamlit.set_page_config

Questions:

    1. Can we use different model and check the accuracy?

    2. 
'''

# Import necessary modules
import warnings

from kats.consts import TimeSeriesData
from kats.models.holtwinters import HoltWintersParams, HoltWintersModel
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(
    page_title              = 'Learning Entries Prediction 2.0', 
    # page_icon = favicon, 
    layout                  = 'wide', 
    initial_sidebar_state   = 'auto'
)

CSV_FILEPATH = 'tact_intern_cleaned.csv'

warnings.simplefilter(action = 'ignore')

def get_base_data():

  df = pd.read_csv(CSV_FILEPATH, 
    usecols = [
        'User', 
        'Date', 
        'Count'
    ]
  )

  return df

learning_heatmap_df = get_base_data()

def get_predicted_df_for_user(username):

    global learning_heatmap_df

    indi_new = learning_heatmap_df[learning_heatmap_df['User'] == username]

    # print(f'entries for {username}')
    # print(len(indi_new))

    if(len(indi_new) == 0):
        empty_df = pd.DataFrame()
        return empty_df

    indi_new.plot()

    indi_new_graph = indi_new.groupby(['Date']).sum()
    indi_new_graph.reset_index(level = 0, inplace = True)

    indi_new_graph.columns  = ['time', 'value']
    indi_new_graph          = TimeSeriesData(indi_new_graph)

    params = HoltWintersParams(
                trend             = "add",
                damped            = False,
                seasonal          = "add",
                seasonal_periods  = 12,
            )

    model = HoltWintersModel(
        data    = indi_new_graph, 
        params  = params
    )

    model.fit()

    fcst = model.predict(steps = 10, alpha = 0.1)

    fcst = fcst.drop(['fcst_lower', 'fcst_upper'], axis = 1)
    fcst = fcst.rename(columns = {"fcst" : "value"})

    indi_new_graph = indi_new_graph.to_dataframe()
    data = indi_new_graph.append(fcst)

    # data_type = type(data)
    # print(f'data : {data_type}')

    # print(data)

    return data


def tact_start():

    st.title('Learning Entries Prediction 2.0')

    fpr_name = st.text_input('Enter Featureprenuer Name')

    if st.button('Forecast'):
        # try:
        st.info(f'Forecast Started for {fpr_name}')
        df = get_predicted_df_for_user(fpr_name)

        print(f'trap12 : df : {df}')

        if(df.empty):
            st.warning("Intern Not Found")
            return

        basic_chart = alt.Chart(df).mark_line().encode(
            x   = 'time',
            y   = 'value'
        )
        st.altair_chart(basic_chart)
        # except:
        #     st.warning("Something went horribly wrong")

if __name__ == '__main__':
    tact_start()
