# -*- coding: utf-8 -*-
"""
Created on Sun May 25 19:34:53 2025

@author: natha
"""

import dash
from dash import html, dcc
import plotly.express as px

import dash
from dash import html, dcc
import plotly.express as px

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("4th Down Decision Tool Preview"),

    html.Div([
        html.Div([
            html.Label("Yards to go"),
            dcc.Input(id='yards-to-go', type='number', value=3, min=1, max=99, style={'width': '100%'}),
        ], style={'flex': '1', 'minWidth': '120px', 'marginRight': '10px'}),

        html.Div([
            html.Label("Yard line"),
            dcc.Input(id='yard-line', type='number', value=40, min=1, max=99, style={'width': '100%'}),
        ], style={'flex': '1', 'minWidth': '120px', 'marginRight': '10px'}),

        html.Div([
            html.Label("Time left (minutes)"),
            dcc.Input(id='time-left', type='number', value=8, min=0, max=60, style={'width': '100%'}),
        ], style={'flex': '1', 'minWidth': '120px', 'marginRight': '10px'}),

        html.Div([
            html.Label("Score differential"),
            dcc.Input(id='score-diff', type='number', value=-4, style={'width': '100%'}),
        ], style={'flex': '1', 'minWidth': '120px', 'marginRight': '10px'}),

        html.Div([
            html.Label("Kicker range (yards)"),
            dcc.Input(id='kicker-range', type='number', value=55, min=0, max=70, style={'width': '100%'}),
        ], style={'flex': '1', 'minWidth': '120px', 'marginRight': '10px'}),

        html.Div([
            html.Label("Punter range (yards)"),
            dcc.Input(id='punter-range', type='number', value=60, min=0, max=80, style={'width': '100%'}),
        ], style={'flex': '1', 'minWidth': '120px'}),
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '10px', 'marginBottom': '20px'}),

    html.P("Situation: 3 yards to go on the 40 yard line, 8 minutes left, down by 4 points."),
    html.P("Kicker range: 55 yards, Punter range: 60 yards"),

    html.Div(style={'display': 'flex', 'justifyContent': 'space-around'}, children=[

        html.Div(style={'border': '1px solid #ccc', 'padding': '15px', 'width': '30%'}, children=[
            html.H3("Go for it"),
            html.P("WPA: (EPAs * Success%) - (EPAf * Failure%)"),
            html.Ul([
                html.Li("Δ TD Probability: 18.7%"),
                html.Li("Δ FG Probability: 5.2%"),
                html.Li("Δ No Score Probability: 76.1%"),
                html.Li("Projected Δ Win Probability: 12.4%")
            ]),
            dcc.Graph(figure=px.pie(
                names=['TD', 'FG', 'No Score'],
                values=[0.35, 0.10, 0.55],
                title='Win Probability Delta'
            ))
        ]),

        html.Div(style={'border': '1px solid #ccc', 'padding': '15px', 'width': '30%'}, children=[
            html.H3("Field Goal"),
            html.P("WPA: Success * EPA_fg - Failure * EPA_miss"),
            html.Ul([
                html.Li("Δ TD Probability: 0%"),
                html.Li("Δ FG Probability: 30.0%"),
                html.Li("Δ No Score Probability: 70.0%"),
                html.Li("Projected Δ Win Probability: 7.8%")
            ]),
            dcc.Graph(figure=px.pie(
                names=['FG', 'Miss'],
                values=[0.75, 0.25],
                title='Win Probability Delta'
            ))
        ]),

        html.Div(style={'border': '1px solid #ccc', 'padding': '15px', 'width': '30%'}, children=[
            html.H3("Punt"),
            html.P("WPA: Based on field position change"),
            html.Ul([
                html.Li("Δ TD Probability: 0%"),
                html.Li("Δ FG Probability: 0%"),
                html.Li("Δ No Score Probability: 100%"),
                html.Li("Projected Δ Win Probability: 2.5%")
            ]),
            dcc.Graph(figure=px.pie(
                names=['No Score'],
                values=[1],
                title='Win Probability Delta'
            ))
        ]),

    ]),

    html.Div(style={'marginTop': '30px', 'borderTop': '2px solid #000', 'paddingTop': '20px'}, children=[
        html.H3("Recommendation"),
        html.Div(style={'fontSize': '24px', 'fontWeight': 'bold', 'color': 'green'}, children="Go for it"),
        html.P("WPA: 0.18"),
        html.P("Projected Δ Win Probability: 12.4%")
    ])
])


if __name__ == '__main__':
    app.run(debug=True)
