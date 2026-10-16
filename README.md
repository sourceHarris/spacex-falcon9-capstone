# Applied Data Science Capstone — Falcon 9 First-Stage Recovery

Historical IBM capstone teaching snapshot, **90 observations (2010–2020)**. Class=1 uses the course-derived positive recovery label (including positive ocean outcomes), not exclusively drone-ship/land-pad returns.

## Provenance
Public prepared dataset: https://github.com/Abhigyan-RA/SpaceX-capstone/blob/main/dataset_part_2.csv . `data.csv` is a focused transcription of 11 analysis columns plus FlightNumber from the original 90 rows; original includes further booster metadata. Verify rows against source before presenting as original independent collection.

## Files
- `data.csv`: historic dataset used for calculations
- `sql_queries.sql`, `spacex_analysis.sqlite`, `sql_*.csv`: executed SQL and results
- `dashboard.py`: working interactive Dash site/outcome/payload dashboard
- `launch_sites_map.html`: Folium site markers with site-level statistics; no invented highway or coastline distances
- `model_results.csv`: trained model results using 70 train / 20 test records, stratification, grid search and fixed seed
- `model_comparison.png`, `confusion_matrix.png`: verified model visualizations
- `SpaceX_Capstone_Final.pdf`: slide presentation

## Run
`pip install -r requirements.txt` then `python dashboard.py`. Open http://127.0.0.1:8050/.

## Modeling caveats
Small historical sample; random train/test split does not assess temporal generalization; labels include positive ocean recoveries. `Legs` and `GridFins` reflect recovery preparations and can be strongly associated with target. Model predictions must not be interpreted as future landing guarantees.

## GitHub link
Create and publish **your own** repository, then update repository placeholder in the final presentation (all slides), export it again, and upload the PDF. The public data source is not your own submission repository.
