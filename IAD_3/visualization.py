import pandas as pd
import dash
from dash import dcc, html
import plotly.express as px

# Step 1: Read the Excel file
# Replace 'your_file.xlsx' with the path to your Excel
excel_path = "PhosAgro_ESG_databook-Rus.xlsx"
df = pd.read_excel(excel_path, sheet_name=None, engine="openpyxl")

# Step 2: Flatten and filter sheets (assume each sheet has similar format)
data = []
for sheet_name, sheet_df in df.items():
    sheet_df = sheet_df.dropna(how="all")  # Remove empty rows
    for i, row in sheet_df.iterrows():
        if isinstance(row[0], str) and row[0].strip() != '':
            try:
                item = {
                    "Metric": row[0],
                    "2019": row.get(3),
                    "2020": row.get(4),
                    "2021": row.get(5),
                    "2022": row.get(6),
                    "2023": row.get(7),
                    "Sheet": sheet_name
                }
                data.append(item)
            except Exception:
                continue

clean_df = pd.DataFrame(data)

# Step 3: Dash app setup
app = dash.Dash(__name__)
app.title = "Corporate Governance Dashboard"

app.layout = html.Div([
    html.H1("📊 Corporate Governance Dashboard", style={'textAlign': 'center'}),
    dcc.Dropdown(
        id="metric-selector",
        options=[{"label": row["Metric"], "value": row["Metric"]} for _, row in clean_df.iterrows()],
        value=clean_df["Metric"].iloc[0],
        style={"width": "80%", "margin": "0 auto"}
    ),
    dcc.Graph(id="line-chart")
])

@app.callback(
    dash.dependencies.Output("line-chart", "figure"),
    [dash.dependencies.Input("metric-selector", "value")]
)
def update_chart(selected_metric):
    row = clean_df[clean_df["Metric"] == selected_metric].iloc[0]
    years = ["2019", "2020", "2021", "2022", "2023"]
    values = [row[year] for year in years]
    fig = px.line(x=years, y=values, markers=True, title=f"{selected_metric} (2019–2023)")
    fig.update_layout(xaxis_title="Year", yaxis_title="Value", template="plotly_dark")
    return fig

if __name__ == "__main__":
    app.run(debug=True)
