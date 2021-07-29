'''
Created on 

Course work: 

@author: Tact Team

Source:
    
'''

# Import necessary modules
from kats.consts import TimeSeriesData
from kats.models.holtwinters import HoltWintersParams, HoltWintersModel
import warnings
import pandas as pd
import streamlit as st
import altair as alt

warnings.simplefilter(action = 'ignore')


def intern(username):

  learning_heatmap_df = pd.read_csv('tact_intern_cleaned.csv', usecols = ['User', 'Date', 'Count'])

  indi_new = learning_heatmap_df[learning_heatmap_df['User'] == username]
  indi_new.plot()

  indi_new_graph = indi_new.groupby(['Date']).sum()
  indi_new_graph.reset_index(level = 0, inplace = True)

  indi_new_graph.columns  = ['time','value']
  indi_new_graph          = TimeSeriesData(indi_new_graph)

  params = HoltWintersParams(
              trend             = "add",
              damped            = False,
              seasonal          = "add",
              seasonal_periods  = 12,
          )

  model = HoltWintersModel(
      data = indi_new_graph, 
      params = params)

  model.fit()

  fcst = model.predict(steps = 10, alpha = 0.1)

  fcst = fcst.drop(['fcst_lower', 'fcst_upper'], axis = 1)
  fcst = fcst.rename(columns = {"fcst" : "value"})
  
  indi_new_graph = indi_new_graph.to_dataframe()
  data = indi_new_graph.append(fcst)

  return data

st.title('Learning Analytics Prediction')

name = st.text_input('Enter Featureprenuer Name')

if st.button('Forecast'):
    try:
      st.info(f'Forecast Started for {name}')
      df = intern(name)
      basic_chart = alt.Chart(df).mark_line().encode(
        x = 'time',
        y = 'value'
      )
      st.altair_chart(basic_chart)
    except:
      st.warning("Intern Not Found")


