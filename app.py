import streamlit as st
import pickle
import pandas as pd
import sklearn
st.title("IPL Win Predictor")
pipe=pickle.load(open('pipe.pkl','rb'))
teams=['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']
cities=['Bangalore', 'Kolkata', 'Ahmedabad', 'Delhi', 'Durban', 'Chennai',
       'Chandigarh', 'Cuttack', 'Hyderabad', 'Johannesburg', 'Mohali',
       'Mumbai', 'Nagpur', 'Jaipur', 'Port Elizabeth', 'Centurion',
       'Abu Dhabi', 'Cape Town', 'Dharamsala', 'Pune', 'Ranchi',
       'Kimberley', 'Visakhapatnam', 'Indore', 'Sharjah', 'Bengaluru',
       'Raipur', 'Bloemfontein', 'East London']
col1,col2=st.columns(2)

with col1:
    batting_team=st.selectbox("Select the batting team",sorted(teams))

with col2:
    bowling_team=st.selectbox("Select the bowling team",sorted(teams))

selected_city=st.selectbox("Select the host city",sorted(cities))

target=st.number_input('Target')

col3,col4,col5=st.columns(3)
with col3:
    score=st.number_input('Score')
with col4:
    overs=st.number_input('Overs Completed')
with col5:
    wickets=st.number_input('Wickets Out')

if st.button("Predict Probability"):
    run_left=target-score
    ball_left=120-(overs*6)
    wickets=10-wickets
    crr=score/overs
    rrr=(run_left*6)/ball_left

    input_df=pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[selected_city],
                  'runs_left':[run_left],'ball_left':[ball_left],'wickets':[wickets],'total_runs_x':[target],
                  'crr': [crr],'rrr':[rrr]
                  })
    result=pipe.predict_proba(input_df)
    loss=result[0][0]
    win=result[0][1]
    st.header("The Winning Probablity for the teams are")
    st.text(batting_team+"- "+ str(round(win*100))+"%")
    st.text(bowling_team+"- "+str(round(loss*100))+"%")