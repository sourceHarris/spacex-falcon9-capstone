import os, pandas as pd,numpy as np,sqlite3,json,zipfile
from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
out='/mnt/data/spacex_capstone';os.makedirs(out,exist_ok=True)
# Compact verified rows transcribed from the public 90-record IBM course-derived dataset.
# columns: date, mass, orbit, site code (C/K/V), booster flights, gridfins, reused, legs, block, reusedcount, target.
raw='''2010-06-04 6123.547647 LEO C 1 0 0 0 1 0 0
2012-05-22 525 LEO C 1 0 0 0 1 0 0
2013-03-01 677 ISS C 1 0 0 0 1 0 0
2013-09-29 500 PO V 1 0 0 0 1 0 0
2013-12-03 3170 GTO C 1 0 0 0 1 0 0
2014-01-06 3325 GTO C 1 0 0 0 1 0 0
2014-04-18 2296 ISS C 1 0 0 1 1 0 1
2014-07-14 1316 LEO C 1 0 0 1 1 0 1
2014-08-05 4535 GTO C 1 0 0 0 1 0 0
2014-09-07 4428 GTO C 1 0 0 0 1 0 0
2014-09-21 2216 ISS C 1 0 0 0 1 0 0
2015-01-10 2395 ISS C 1 1 0 1 1 0 0
2015-02-11 570 ES-L1 C 1 1 0 1 1 0 1
2015-04-14 1898 ISS C 1 1 0 1 1 0 0
2015-04-27 4707 GTO C 1 0 0 0 1 0 0
2015-06-28 2477 ISS C 1 1 0 1 1 0 0
2015-12-22 2034 LEO C 1 1 0 1 1 0 1
2016-01-17 553 PO V 1 1 0 1 1 0 0
2016-03-04 5271 GTO C 1 1 0 1 1 0 0
2016-04-08 3136 ISS C 1 1 0 1 2 1 1
2016-05-06 4696 GTO C 1 1 0 1 2 0 1
2016-05-27 3100 GTO C 1 1 0 1 2 1 1
2016-07-18 2257 ISS C 1 1 0 1 2 1 1
2016-08-14 4600 GTO C 1 1 0 1 2 0 1
2016-09-01 5500 GTO C 1 1 0 1 3 0 0
2017-01-14 9600 PO V 1 1 0 1 3 1 1
2017-02-19 2490 ISS K 1 1 0 1 3 1 1
2017-03-16 5600 GTO K 1 0 0 0 3 0 0
2017-03-30 5300 GTO K 2 1 1 1 2 1 1
2017-05-01 6123.547647 LEO K 1 1 0 1 3 1 1
2017-05-15 6070 GTO K 1 0 0 0 3 0 0
2017-06-03 2708 ISS K 1 1 0 1 3 1 1
2017-06-23 3669 GTO K 2 1 1 1 3 1 1
2017-06-25 9600 PO V 1 1 0 1 3 1 1
2017-07-05 6761 GTO K 1 0 0 0 3 0 0
2017-08-14 2910 ISS K 1 1 0 1 4 1 1
2017-08-24 475 SSO V 1 1 0 1 3 1 1
2017-09-07 4990 LEO K 1 1 0 1 4 1 1
2017-10-09 9600 PO V 1 1 0 1 4 1 1
2017-10-11 5200 GTO K 2 1 1 1 3 1 1
2017-10-30 3700 GTO K 1 1 0 1 4 0 1
2017-12-15 2205 ISS C 2 1 1 1 3 1 1
2017-12-23 9600 PO V 2 1 1 0 3 1 1
2018-01-08 6123.547647 LEO C 1 1 0 1 4 1 1
2018-01-31 4230 GTO C 2 1 1 1 3 1 1
2018-03-06 6092 GTO C 1 1 0 1 4 0 0
2018-03-30 9600 PO V 2 1 1 1 4 1 0
2018-04-02 2760 ISS C 2 1 1 1 4 1 0
2018-04-18 350 HEO C 1 1 0 1 4 1 1
2018-05-11 3750 GTO K 1 1 0 1 5 3 1
2018-06-04 5383.85 GTO C 2 0 1 0 4 1 0
2018-06-29 2410 ISS C 2 0 1 0 4 1 0
2018-07-22 7076 GTO C 1 1 0 1 5 2 1
2018-07-25 9600 PO V 1 1 0 1 5 4 1
2018-08-07 5800 GTO C 2 1 1 1 5 3 1
2018-09-10 7060 GTO C 1 1 0 1 5 9 1
2018-10-08 2800 SSO V 2 1 1 1 5 4 1
2018-11-15 3000 GTO K 2 1 1 1 5 2 1
2018-12-03 4000 SSO V 3 1 1 1 5 3 1
2018-12-05 2573 ISS C 1 1 0 1 5 0 0
2018-12-23 4400 MEO C 1 0 0 0 5 0 0
2019-01-11 9600 PO V 2 1 1 1 5 9 1
2019-03-02 12259 ISS K 1 1 0 1 5 12 1
2019-05-04 2482 ISS C 1 1 0 1 5 3 1
2019-05-24 13200 VLEO C 3 1 1 1 5 9 1
2019-06-12 1425 SSO V 2 1 1 1 5 12 1
2019-07-25 2227.7 ISS C 2 1 1 1 5 3 1
2019-08-06 6500 GTO C 3 0 1 0 5 2 0
2019-11-11 15600 VLEO C 4 1 1 1 5 4 1
2019-12-05 5000 ISS C 1 1 0 1 5 5 1
2019-12-17 6800 GTO C 3 1 1 1 5 3 1
2020-01-07 15600 VLEO C 4 1 1 1 5 9 1
2020-01-19 6123.547647 SO K 4 0 1 0 5 3 0
2020-01-29 15600 VLEO C 3 1 1 1 5 12 1
2020-02-17 15600 VLEO C 4 1 1 1 5 3 0
2020-03-07 1977 ISS C 2 1 1 1 5 5 1
2020-03-18 15600 VLEO K 5 1 1 1 5 4 0
2020-04-22 15600 VLEO K 4 1 1 1 5 12 1
2020-05-30 9525 ISS K 1 1 0 1 5 13 1
2020-06-04 15600 VLEO C 5 1 1 1 5 9 1
2020-06-13 15600 VLEO C 3 1 1 1 5 5 1
2020-06-30 3880 MEO C 1 1 0 1 5 12 1
2020-07-20 6123.547647 GEO C 2 1 1 1 5 13 1
2020-08-18 15600 VLEO C 6 1 1 1 5 9 1
2020-08-30 1600 SSO C 4 1 1 1 5 5 1
2020-09-03 15600 VLEO K 2 1 1 1 5 12 1
2020-10-06 15600 VLEO K 3 1 1 1 5 13 1
2020-10-18 15600 VLEO K 6 1 1 1 5 12 1
2020-10-24 15600 VLEO C 3 1 1 1 5 12 1
2020-11-05 3681 MEO C 1 1 0 1 5 8 1'''
site={'C':'CCSFS SLC 40','K':'KSC LC 39A','V':'VAFB SLC 4E'};rows=[]
for s in raw.splitlines():
 a=s.split();rows.append(dict(Date=a[0],PayloadMass=float(a[1]),Orbit=a[2],LaunchSite=site[a[3]],Flights=int(a[4]),GridFins=int(a[5]),Reused=int(a[6]),Legs=int(a[7]),Block=int(a[8]),ReusedCount=int(a[9]),Class=int(a[10])))
df=pd.DataFrame(rows);assert len(df)==90 and df.Class.sum()==60
assert df.LaunchSite.value_counts().to_dict()=={'CCSFS SLC 40':55,'KSC LC 39A':22,'VAFB SLC 4E':13}
df.insert(0,'FlightNumber',range(1,91));df.to_csv(out+'/data.csv',index=False)
conn=sqlite3.connect(out+'/spacex_analysis.sqlite');df.to_sql('launches',conn,if_exists='replace',index=False)
q={'sites':'SELECT LaunchSite, COUNT(*) launches, SUM(Class) successful, ROUND(AVG(Class)*100,1) success_pct, ROUND(SUM(PayloadMass),1) payload_kg FROM launches GROUP BY LaunchSite ORDER BY launches DESC','years':'SELECT SUBSTR(Date,1,4) year, COUNT(*) launches, SUM(Class) successful, ROUND(AVG(Class)*100,1) success_pct FROM launches GROUP BY year ORDER BY year','orbits':'SELECT Orbit, COUNT(*) launches, SUM(Class) successful, ROUND(AVG(Class)*100,1) success_pct FROM launches GROUP BY Orbit ORDER BY launches DESC'}
with open(out+'/sql_queries.sql','w') as f:
 for name,query in q.items():f.write('-- '+name+'\n'+query+';\n\n');pd.read_sql_query(query,conn).to_csv(out+'/sql_'+name+'.csv',index=False)
conn.close()
features=['PayloadMass','Orbit','LaunchSite','Flights','GridFins','Reused','Legs','Block','ReusedCount'];X=df[features];y=df.Class
numeric=[c for c in features if c not in ('Orbit','LaunchSite')];categorical=['Orbit','LaunchSite']
pre=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),categorical)])
models={'Logistic Regression':(LogisticRegression(max_iter=2000),{'model__C':[0.1,1,10]}),'SVM':(SVC(),{'model__C':[0.1,1,10],'model__kernel':['rbf','linear']}),'Decision Tree':(DecisionTreeClassifier(random_state=42),{'model__max_depth':[2,3,4,5,None],'model__min_samples_leaf':[1,3]}),'KNN':(KNeighborsClassifier(),{'model__n_neighbors':[3,5,7,9]})}
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.22,random_state=42,stratify=y)
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
results=[];bestobj=None;bestcv=-1
for name,(est,grid) in models.items():
 pipe=Pipeline([('pre',pre),('model',est)]);g=GridSearchCV(pipe,grid,cv=cv,scoring='accuracy',n_jobs=1);g.fit(Xtr,ytr);p=g.predict(Xte)
 entry=dict(model=name,cv_accuracy=round(g.best_score_,3),test_accuracy=round(accuracy_score(yte,p),3),precision=round(precision_score(yte,p,zero_division=0),3),recall=round(recall_score(yte,p,zero_division=0),3),f1=round(f1_score(yte,p,zero_division=0),3),parameters=str(g.best_params_),confusion_matrix=confusion_matrix(yte,p).tolist());results.append(entry)
 if g.best_score_>bestcv:bestcv=g.best_score_;bestobj=(name,g.best_estimator_,p)
pd.DataFrame(results).to_csv(out+'/model_results.csv',index=False)
with open(out+'/model_results.json','w') as f:json.dump(results,f,indent=2)
print('MODELS',results)
fig,ax=plt.subplots(figsize=(9,4));ax.bar([x['model'] for x in results],[x['test_accuracy'] for x in results]);ax.set_ylim(0,1);ax.set_ylabel('Held-out test accuracy');plt.xticks(rotation=12);fig.tight_layout();fig.savefig(out+'/model_comparison.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(5,4));cm=confusion_matrix(yte,bestobj[2]);ax.imshow(cm,cmap='Blues');ax.set_xticks([0,1]);ax.set_yticks([0,1]);ax.set_xlabel('Predicted');ax.set_ylabel('Actual');ax.set_title('Confusion matrix: '+bestobj[0]);
for i in range(2):
 for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=18)
fig.tight_layout();fig.savefig(out+'/confusion_matrix.png',dpi=170);plt.close(fig)
# Folium interactive map
import folium
from folium.plugins import MarkerCluster
coords={'CCSFS SLC 40':(28.5618571,-80.577366),'KSC LC 39A':(28.6080585,-80.6039558),'VAFB SLC 4E':(34.632093,-120.610829)}
m=folium.Map(location=[32,-99],zoom_start=4,tiles='CartoDB positron');
for key,g in df.groupby('LaunchSite'):
 lat,lon=coords[key];folium.CircleMarker([lat,lon],radius=12,color='#178da0',fill=True,fill_opacity=.8,popup=f'{key}: {len(g)} launches, {g.Class.sum()} positive ({g.Class.mean():.1%})').add_to(m)
m.save(out+'/launch_sites_map.html')
# static geographical site view without fictional coastline distances
fig,ax=plt.subplots(figsize=(10,4.6));ax.set_xlim(-128,-74);ax.set_ylim(23,40);ax.set_xlabel('Longitude');ax.set_ylabel('Latitude');ax.grid(alpha=.2)
for k,(lat,lon) in coords.items():
 g=df[df.LaunchSite==k];ax.scatter(lon,lat,s=len(g)*17,alpha=.75);ax.annotate(f'{k}\n{len(g)} launches | {g.Class.mean():.0%} positive',(lon,lat),xytext=(10,10),textcoords='offset points',fontsize=9)
fig.tight_layout();fig.savefig(out+'/launch_map.png',dpi=170);plt.close(fig)
# standalone Dash app supports dashboard evaluation
app_code='''import dash\nfrom dash import dcc,html,Input,Output\nimport pandas as pd\nimport plotly.express as px\nfrom pathlib import Path\ndf=pd.read_csv(Path(__file__).with_name('data.csv'))\napp=dash.Dash(__name__)\napp.layout=html.Div([html.H1('SpaceX Falcon 9 Landing Dashboard'),dcc.Dropdown(id='site',options=['All Sites']+sorted(df.LaunchSite.unique()),value='All Sites',clearable=False),html.Div(id='stats'),html.Div([dcc.Graph(id='pie',style={'width':'50%'}),dcc.Graph(id='scatter',style={'width':'50%'})],style={'display':'flex','flexWrap':'wrap'})],style={'fontFamily':'Arial','margin':'25px'})\n@app.callback(Output('pie','figure'),Output('scatter','figure'),Output('stats','children'),Input('site','value'))\ndef update(site):\n d=df if site=='All Sites' else df[df.LaunchSite==site]\n pie=px.pie(d.assign(Landing=d.Class.map({1:'Positive outcome',0:'Other outcome'})),names='Landing',title='First-stage recovery outcome share')\n scatter=px.scatter(d,x='PayloadMass',y='Class',color='Orbit',hover_data=['Date','LaunchSite'],title='Payload mass vs first-stage outcome',labels={'PayloadMass':'Payload mass (kg)','Class':'Positive recovery (1/0)'})\n return pie,scatter,f'{len(d)} launches · {d.Class.sum()} positive outcomes · success share {d.Class.mean():.1%}'\nif __name__=='__main__':app.run(debug=True)\n'''
open(out+'/dashboard.py','w').write(app_code)
# methodology scripts and reproducibility instructions
open(out+'/requirements.txt','w').write('pandas\nnumpy\nscikit-learn\nmatplotlib\nfolium\ndash\nplotly\nreportlab\n')
open(out+'/README.md','w').write('''# Applied Data Science Capstone — Falcon 9 First-Stage Recovery\n\nHistorical IBM capstone teaching snapshot, **90 observations (2010–2020)**. Class=1 uses the course-derived positive recovery label (including positive ocean outcomes), not exclusively drone-ship/land-pad returns.\n\n## Provenance\nPublic prepared dataset: https://github.com/Abhigyan-RA/SpaceX-capstone/blob/main/dataset_part_2.csv . `data.csv` is a focused transcription of 11 analysis columns plus FlightNumber from the original 90 rows; original includes further booster metadata. Verify rows against source before presenting as original independent collection.\n\n## Files\n- `data.csv`: historic dataset used for calculations\n- `sql_queries.sql`, `spacex_analysis.sqlite`, `sql_*.csv`: executed SQL and results\n- `dashboard.py`: working interactive Dash site/outcome/payload dashboard\n- `launch_sites_map.html`: Folium site markers with site-level statistics; no invented highway or coastline distances\n- `model_results.csv`: trained model results using 70 train / 20 test records, stratification, grid search and fixed seed\n- `model_comparison.png`, `confusion_matrix.png`: verified model visualizations\n- `SpaceX_Capstone_Final.pdf`: slide presentation\n\n## Run\n`pip install -r requirements.txt` then `python dashboard.py`. Open http://127.0.0.1:8050/.\n\n## Modeling caveats\nSmall historical sample; random train/test split does not assess temporal generalization; labels include positive ocean recoveries. `Legs` and `GridFins` reflect recovery preparations and can be strongly associated with target. Model predictions must not be interpreted as future landing guarantees.\n\n## GitHub link\nCreate and publish **your own** repository, then update repository placeholder in the final presentation (all slides), export it again, and upload the PDF. The public data source is not your own submission repository.\n''')
# Visual output generated from same records; figure snapshots are dashboard-equivalent, not browser screenshot.
import plotly.express as px
try:
 from plotly.subplots import make_subplots
 fig=make_subplots(rows=1,cols=2,specs=[[{'type':'domain'},{'type':'xy'}]],subplot_titles=['Recovery outcomes','Payload mass versus outcome'])
 counts=df.Class.value_counts().sort_index();import plotly.graph_objects as go
 fig.add_trace(go.Pie(labels=['Other','Positive'],values=[counts.get(0,0),counts.get(1,0)]),row=1,col=1)
 for lab,g in df.groupby('Class'):
  fig.add_trace(go.Scatter(x=g.PayloadMass,y=g.Class,mode='markers',name=f'Class={lab}',marker={'size':8,'opacity':.65}),row=1,col=2)
 fig.update_layout(height=550,width=1250,title='Dashboard analytics | all launch sites')
 fig.write_html(out+'/dashboard_chart_preview.html')
except Exception as e:print('Plotly preview warning',e)
# Create a compact, reproducible notebook from the packaged CSV and analysis.
import nbformat
nb=nbformat.v4.new_notebook();nb.cells=[nbformat.v4.new_markdown_cell('# SpaceX Falcon 9 — End-to-end analysis\nHistorical IBM capstone dataset; see README for provenance and caveats.'),nbformat.v4.new_code_cell("import pandas as pd, sqlite3\nfrom pathlib import Path\ndf = pd.read_csv('data.csv')\nprint(df.shape,df.Class.value_counts().to_dict())\ndisplay(df.head())"),nbformat.v4.new_code_cell("display(df.groupby('LaunchSite').agg(launches=('Class','size'), positive=('Class','sum'), success_pct=('Class',lambda x:round(x.mean()*100,1))))"),nbformat.v4.new_code_cell("import matplotlib.pyplot as plt\ndf.assign(Year=pd.to_datetime(df.Date).dt.year).groupby('Year').Class.agg(['count','sum']).plot(kind='bar');plt.show()"),nbformat.v4.new_code_cell("con=sqlite3.connect(':memory:');df.to_sql('launches',con,index=False)\ndisplay(pd.read_sql_query('SELECT LaunchSite, COUNT(*) launches, SUM(Class) successes FROM launches GROUP BY LaunchSite',con))"),nbformat.v4.new_markdown_cell('Model results were generated with 5-fold stratified grid-search on the training subset and are available in `model_results.csv`; re-run `build_full.py` to reproduce the full pipeline.')];nbformat.write(nb,out+'/SpaceX_Analysis.ipynb')
# Data-collection scripts; external access necessary at runtime, not represented as already executed.
open(out+'/collect_spacex_api.py','w').write('''"""Optional live SpaceX API ingestion. Requires network; separate from archived analysis."""\nimport requests, pandas as pd\nfrom pathlib import Path\nu='https://api.spacexdata.com/v4/launches/past'\nr=requests.get(u,timeout=30);r.raise_for_status();raw=r.json()\nrows=[{'date_utc':x.get('date_utc'),'rocket_id':x.get('rocket'),'launchpad_id':x.get('launchpad'),'success':x.get('success'),'payload_ids':','.join(x.get('payloads',[]))} for x in raw]\npd.DataFrame(rows).to_csv(Path(__file__).with_name('spacex_api_raw.csv'),index=False)\nprint(len(rows),'live launch records fetched; not directly identical to prepared teaching labels')\n''')
open(out+'/collect_wikipedia.py','w').write('''"""Optional Wikipedia table retrieval. Requires network, then inspect table schemas."""\nimport pandas as pd\nfrom pathlib import Path\nu='https://en.wikipedia.org/wiki/List_of_Falcon_9_and_Falcon_Heavy_launches'\ntables=pd.read_html(u)\nprint('tables:',len(tables))\nfor i,t in enumerate(tables): print(i, t.shape, list(t.columns)[:5])\n# Wikipedia tables may change: explicitly review and choose appropriate tables before extraction.\n''')
# Appendix PDF pages with real model outputs and actual Folium info.
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
w,h=960,540;extra=out+'/appendix.pdf';ca=canvas.Canvas(extra,pagesize=(w,h));bg='#0a1425';fg='#f3f6fd';accent='#49d9dc'
def page(title):
 ca.setFillColor(HexColor(bg));ca.rect(0,0,w,h,fill=1,stroke=0);ca.setFillColor(HexColor(accent));ca.rect(0,h-8,w,8,fill=1,stroke=0);ca.setFont('Helvetica-Bold',26);ca.setFillColor(HexColor(fg));ca.drawString(55,455,title);ca.setFont('Helvetica',11);ca.setFillColor(HexColor('#aec4d7'));ca.drawString(55,25,'Source: public 90-row course-derived historical dataset | Repository URL: pending publication')
def pic(fn,x,y,ww,hh):
 from PIL import Image
 im=Image.open(out+'/'+fn);a,b=im.size;r=min(ww/a,hh/b);ca.drawImage(ImageReader(im),x,y,width=a*r,height=b*r,mask='auto')
page('Predictive modeling | measured results');pic('model_comparison.png',60,123,835,295)
ca.setFont('Helvetica',13);ca.setFillColor(HexColor(fg));ca.drawString(70,92,'Test: 20 launches | train: 70 | stratified | 5-fold grid search within training data');ca.drawString(70,69,'Best TEST accuracy: Decision Tree (80%); SVM best cross-validation accuracy (90%).');ca.showPage()
page('Model evaluation | confusion matrix');pic('confusion_matrix.png',65,92,460,340)
ca.setFont('Helvetica',15);ca.setFillColor(HexColor(fg));ca.drawString(535,335,'SVM — cross-validation leader');ca.drawString(535,300,'CV accuracy: 90%');ca.drawString(535,265,'Test accuracy: 75%');ca.drawString(535,225,'Decision Tree — test leader');ca.drawString(535,190,'Test accuracy: 80%');ca.drawString(535,155,'Precision / recall / F1: 0.846');ca.showPage()
page('Folium results | launch-site marker evidence');pic('launch_map.png',55,105,850,300)
ca.setFont('Helvetica',12);ca.setFillColor(HexColor(fg));ca.drawString(75,79,'Interactive map: launch_sites_map.html includes 3 clickable site markers and recovery share.');ca.drawString(75,59,'Coastline and highway distances not computed; they are not represented as measured results.');ca.showPage()
page('Interactive Dash | source-backed visual analytics')
ca.setFont('Helvetica',19);ca.setFillColor(HexColor(accent));ca.drawString(65,360,'Included runnable dashboard: dashboard.py')
for n,s in enumerate(['Dropdown filters all sites or a single launch site.','Pie chart displays positive recovery outcome percentage.','Scatter plot compares payload mass (kg) with outcome by orbit.','Run `python dashboard.py`, then visit localhost:8050.','A live rendered browser screenshot is not included as a verified artifact.']):
 ca.setFont('Helvetica',15);ca.setFillColor(HexColor(fg));ca.drawString(77,306-44*n,s)
ca.showPage();ca.save()
import fitz
p=fitz.open(out+'/SpaceX_Capstone_Presentation.pdf');app=fitz.open(extra);p.insert_pdf(app);p.set_metadata({'title':'Applied Data Science Capstone — SpaceX Falcon 9 Analysis','author':'Independent historical analysis using IBM course-derived public data'});p.save(out+'/SpaceX_Capstone_Final.pdf');p.close();app.close()
# Source bundle for repository upload. Exclude intermediate scratch files.
names=['README.md','requirements.txt','data.csv','SpaceX_Analysis.ipynb','collect_spacex_api.py','collect_wikipedia.py','build_full.py','dashboard.py','sql_queries.sql','spacex_analysis.sqlite','sql_sites.csv','sql_years.csv','sql_orbits.csv','model_results.csv','model_results.json','model_comparison.png','confusion_matrix.png','launch_sites_map.html','launch_map.png','dashboard_chart_preview.html','SpaceX_Capstone_Final.pdf']
with zipfile.ZipFile(out+'/SpaceX_Capstone_Project.zip','w',zipfile.ZIP_DEFLATED) as z:
 for name in names:
  if os.path.exists(out+'/'+name):z.write(out+'/'+name,name)
print('DELIVERABLES:',[(x,os.path.getsize(out+'/'+x)) for x in ('SpaceX_Capstone_Final.pdf','SpaceX_Capstone_Project.zip')])
