import dash
from dash import html, dcc
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import numpy as np
import os
import gdown

# ==============================
# CONFIG
# ==============================
GRAPH_CONFIG = {
    'displayModeBar': True,
    'displaylogo': False,
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'graphique_fraude',
        'height': 1000,
        'width': 1400,
        'scale': 2
    }
}

pio.templates.default = "plotly_white"

# Colors
COLOR_TITLE = '#B22222'
COLOR_BEFORE = '#A52A2A'
COLOR_AFTER  = '#FFA500'
ACCENT_COLOR = '#004C99'
APP_BACKGROUND = '#F8F8F8'

# ==============================
# KPI DATA
# ==============================
total_before     = 255353
total_after      = 283726
montant_before   = 21457880.75
montant_after    = 25192001.68
fraudes_before   = 340
fraudes_after    = 473
taux_before      = 0.13
taux_after       = 0.17

delta_tx      = round(((total_after - total_before) / total_before) * 100, 1)
delta_montant = round(((montant_after - montant_before) / montant_before) * 100, 1)
delta_fraudes = round(((fraudes_after - fraudes_before) / fraudes_before) * 100, 1)
delta_taux    = round(taux_after - taux_before, 2)

# ==============================
# DATA SIMULATED FOR ML TABLES
# ==============================
ML_RESULTS_DATA = [
    {'Métrique': 'Accuracy', 'Valeur': '99.94%'},
    {'Métrique': 'Precision (Fraude)', 'Valeur': '92.5%'},
    {'Métrique': 'Recall (Fraude)', 'Valeur': '87.0%'},
    {'Métrique': 'F1-Score', 'Valeur': '89.7%'},
    {'Métrique': 'AUC-ROC', 'Valeur': '0.96'}
]

CONFUSION_MATRIX = np.array([[284310, 16], [50, 423]])

# ==============================
# DOWNLOAD CSVs IF MISSING
# ==============================
csv_files = {
    "creditcard_before.csv": "https://drive.google.com/uc?id=171LwZZmFCdANSgLWmcNqFbLOHs9YhZA2",
    "creditcard_after.csv": "https://drive.google.com/uc?id=1EYYcmIYSkO4qSw1EdhHCUt_vCeSwQtSP",
    "creditcard.csv": "https://drive.google.com/uc?id=1R401XOKLPvvAeNMMTzXpDv5KmuVG7Njx"
}

for name, url in csv_files.items():
    if not os.path.exists(name):
        gdown.download(url, name, quiet=True)

df_before = pd.read_csv("creditcard_before.csv")
df_after  = pd.read_csv("creditcard_after.csv")
df_full   = pd.read_csv("creditcard.csv")

# ==============================
# KPI CARD FUNCTION
# ==============================
def format_number(x):
    if isinstance(x, (int, np.integer)):
        return f"{x:,}".replace(",", " ")
    else:
        return f"{x:,.0f}".replace(",", " ")

def kpi_card(title, value, unit="", delta=None, good_is_up=True):
    value_str = format_number(value) + unit
    if delta is not None:
        positive = (delta > 0 and good_is_up) or (delta < 0 and not good_is_up)
        color = '#28A745' if positive else '#DC3545'
        icon = "▲" if delta > 0 else "▼"
        delta_text = f"{icon} {abs(delta)}%"
    else:
        color = '#6c757d'
        delta_text = "–"

    return html.Div(style={
        'backgroundColor':'#fff','padding':'32px','borderRadius':'18px',
        'boxShadow':'0 4px 12px rgba(0,0,0,0.05)','textAlign':'center','border':'1px solid #e9ecef'
    }, children=[
        html.P(title, style={'margin':'0 0 12px 0','color':'#495057','fontSize':'18px','fontWeight':'600'}),
        html.H3(value_str, style={'margin':'8px 0','color':ACCENT_COLOR,'fontSize':'42px','fontWeight':'bold'}),
        html.P(delta_text, style={'margin':'0','color':color,'fontWeight':'bold','fontSize':'24px'})
    ])

# ==============================
# PLOTS
# ==============================
def fixed(fig, h=460):
    fig.update_layout(height=h, margin=dict(t=90, b=70, l=70, r=50), 
                      title_font_family='Georgia, "Times New Roman", Times, serif', 
                      title_font_size=18, title_font_color='#495057')
    return fig

def plot_fraud_count():
    fig = go.Figure()
    fig.add_bar(x=['Avant Big Data', 'Après Big Data'], y=[fraudes_before, fraudes_after],
                marker_color=[COLOR_BEFORE, COLOR_AFTER],
                text=[fraudes_before, fraudes_after], textposition='outside')
    fig.update_layout(title="Nombre de transactions frauduleuses détectées", showlegend=False)
    return fixed(fig)

def plot_fraud_rate():
    fig = go.Figure()
    fig.add_bar(x=['Avant Big Data', 'Après Big Data'], y=[taux_before, taux_after],
                marker_color=[COLOR_BEFORE, COLOR_AFTER],
                text=[f"{taux_before:.2f}%", f"{taux_after:.2f}%"], textposition='outside')
    fig.update_layout(title="Taux de fraude détecté dans les données", showlegend=False)
    return fixed(fig)

def plot_histogram():
    fig = go.Figure()
    fig.add_histogram(x=df_before['Amount'].dropna(), name='Avant', marker_color=COLOR_BEFORE, opacity=0.75, nbinsx=70)
    fig.add_histogram(x=df_after['Amount'], name='Après', marker_color=COLOR_AFTER, opacity=0.75, nbinsx=70)
    fig.update_layout(barmode='overlay', title="Distribution des montants des transactions")
    return fixed(fig)

def plot_box_before():
    fig = px.box(df_before, x='Class', y='Amount', color='Class', color_discrete_map={0: '#6c757d', 1: COLOR_BEFORE},
                 title="Avant Big Data : les fraudes sont noyées dans le bruit")
    return fixed(fig)

def plot_box_after():
    fig = px.box(df_after, x='Class', y='Amount', color='Class', color_discrete_map={0: '#a0c4ff', 1: COLOR_AFTER},
                 title="Après Big Data : les fraudes deviennent clairement identifiables")
    return fixed(fig)

def plot_correlation():
    corr = df_after.corr(numeric_only=True)['Class'].drop('Class', errors='ignore').abs().sort_values(ascending=False).head(10)
    fig = go.Figure()
    fig.add_bar(y=corr.index, x=corr.values, orientation='h', marker_color=corr.values, marker_colorscale='Viridis',
                text=[f"{v:.3f}" for v in corr.values], textposition='outside')
    fig.update_layout(title="Top 10 des variables les plus discriminantes", yaxis=dict(autorange="reversed"))
    return fixed(fig, h=480)

# ==============================
# DASH APP
# ==============================
app = dash.Dash(__name__)
app.title = "Performance Anti-Fraude Post-Migration Big Data"

# Layout
app.layout = html.Div(style={'backgroundColor': APP_BACKGROUND, 'padding': '40px'}, children=[
    html.H1("Dashboard Anti-Fraude Post-Migration Big Data", style={'color': COLOR_TITLE, 'textAlign': 'center'}),
    html.Div(style={'display': 'flex', 'gap': '20px', 'marginTop': '40px', 'justifyContent': 'center'}, children=[
        kpi_card("Transactions Avant", total_before, delta=delta_tx),
        kpi_card("Transactions Après", total_after, delta=delta_tx),
        kpi_card("Montant Avant", montant_before, unit=" $", delta=delta_montant),
        kpi_card("Montant Après", montant_after, unit=" $", delta=delta_montant),
        kpi_card("Fraudes Avant", fraudes_before, delta=delta_fraudes, good_is_up=False),
        kpi_card("Fraudes Après", fraudes_after, delta=delta_fraudes),
        kpi_card("Taux Avant", taux_before, unit="%", delta=delta_taux, good_is_up=False),
        kpi_card("Taux Après", taux_after, unit="%", delta=delta_taux)
    ]),
    html.Div(style={'marginTop': '50px'}, children=[
        dcc.Graph(figure=plot_fraud_count(), config=GRAPH_CONFIG),
        dcc.Graph(figure=plot_fraud_rate(), config=GRAPH_CONFIG),
        dcc.Graph(figure=plot_histogram(), config=GRAPH_CONFIG),
        dcc.Graph(figure=plot_box_before(), config=GRAPH_CONFIG),
        dcc.Graph(figure=plot_box_after(), config=GRAPH_CONFIG),
        dcc.Graph(figure=plot_correlation(), config=GRAPH_CONFIG),
    ])
])

# ==============================
# RUN SERVER
# ==============================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port, debug=True)
