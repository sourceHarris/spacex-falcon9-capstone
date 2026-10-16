"""Optional Wikipedia table retrieval. Requires network, then inspect table schemas."""
import pandas as pd
from pathlib import Path
u='https://en.wikipedia.org/wiki/List_of_Falcon_9_and_Falcon_Heavy_launches'
tables=pd.read_html(u)
print('tables:',len(tables))
for i,t in enumerate(tables): print(i, t.shape, list(t.columns)[:5])
# Wikipedia tables may change: explicitly review and choose appropriate tables before extraction.
