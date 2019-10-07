import altair as alt
import pandas as pd
data = pd.DataFrame({'x': [1, 2, 3], 'y': [2, 1, 2]})

chart = alt.Chart(data).mark_line().encode(
     x='x',
     y='y'
)

from pprint import pprint
pprint(chart.to_dict())


url = 'data.json'
data.to_json(url, orient='records')

chart = alt.Chart(url).mark_line().encode(
    x='x:Q',
    y='y:Q'
)
pprint(chart.to_dict())