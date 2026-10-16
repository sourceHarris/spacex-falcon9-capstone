"""Optional live SpaceX API ingestion. Requires network; separate from archived analysis."""
import requests, pandas as pd
from pathlib import Path
u='https://api.spacexdata.com/v4/launches/past'
r=requests.get(u,timeout=30);r.raise_for_status();raw=r.json()
rows=[{'date_utc':x.get('date_utc'),'rocket_id':x.get('rocket'),'launchpad_id':x.get('launchpad'),'success':x.get('success'),'payload_ids':','.join(x.get('payloads',[]))} for x in raw]
pd.DataFrame(rows).to_csv(Path(__file__).with_name('spacex_api_raw.csv'),index=False)
print(len(rows),'live launch records fetched; not directly identical to prepared teaching labels')
