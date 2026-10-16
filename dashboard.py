import dash
from dash import dcc,html,Input,Output
import pandas as pd
import plotly.express as px
from pathlib import Path
df=pd.read_csv(Path(__file__).with_name('data.csv'))
app=dash.Dash(__name__)
app.layout=html.Div([html.H1('SpaceX Falcon 9 Landing Dashboard'),dcc.Dropdown(id='site',options=['All Sites']+sorted(df.LaunchSite.unique()),value='All Sites',clearable=False),html.Div(id='stats'),html.Div([dcc.Graph(id='pie',style={'width':'50%'}),dcc.Graph(id='scatter',style={'width':'50%'})],style={'display':'flex','flexWrap':'wrap'})],style={'fontFamily':'Arial','margin':'25px'})
@app.callback(Output('pie','figure'),Output('scatter','figure'),Output('stats','children'),Input('site','value'))
def update(site):
 d=df if site=='All Sites' else df[df.LaunchSite==site]
 pie=px.pie(d.assign(Landing=d.Class.map({1:'Positive outcome',0:'Other outcome'})),names='Landing',title='First-stage recovery outcome share')
 scatter=px.scatter(d,x='PayloadMass',y='Class',color='Orbit',hover_data=['Date','LaunchSite'],title='Payload mass vs first-stage outcome',labels={'PayloadMass':'Payload mass (kg)','Class':'Positive recovery (1/0)'})
 return pie,scatter,f'{len(d)} launches · {d.Class.sum()} positive outcomes · success share {d.Class.mean():.1%}'
if __name__=='__main__':app.run(debug=True)
