# -*- coding: utf-8 -*-
"""
app.py -- 4th Down Decision Tool: Dash UI.

Loads the trained artifacts through fourth_down.decide() and renders the three
options as broadcast-style cards (real field strip, big colored numbers, EPA/WPA
toggle). Serves on Render via gunicorn: `gunicorn app:server`.
"""

from dash import Dash, html, dcc, Input, Output
import fourth_down as fd

app = Dash(__name__, title="4th Down Decision Tool")
server = app.server

NAMES = {"go": "Go for it", "fg": "Field goal", "punt": "Punt"}

#------------------------------------------------------------------------------
# Page shell (fonts + theming)
#------------------------------------------------------------------------------
"""
Index template pulls in the condensed display font (Oswald) and a clean body
sans (Inter), then defines light and dark palettes as CSS variables. With no
data-theme attribute the page follows the OS via prefers-color-scheme; the
corner toggle sets data-theme to override. The field green is fixed in both
modes; only chrome, text and surfaces flip.
"""
app.index_string = """<!DOCTYPE html>
<html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{--bg:#eef0f3;--card:#fff;--text:#0f1720;--muted:#5b6672;--border:#e1e4ea;
--accent:#0e7c8b;--accent-ink:#fff;--green:#1a9750;--green-d:#0f6e3a;
--red:#d63b34;--red-d:#a52822;--gray:#c3c9d2;--field:#2f7d43;--field-ez:#245f34;}
@media (prefers-color-scheme:dark){:root:not([data-theme]){--bg:#0d1117;--card:#171d26;
--text:#e8edf3;--muted:#98a2b0;--border:#28303b;--accent:#25b3c3;--accent-ink:#04222a;
--green:#37c46e;--green-d:#1f9a52;--red:#f0554f;--red-d:#c93b36;--gray:#39424e;
--field:#276b39;--field-ez:#1c4e2b;}}
[data-theme=dark]{--bg:#0d1117;--card:#171d26;--text:#e8edf3;--muted:#98a2b0;
--border:#28303b;--accent:#25b3c3;--accent-ink:#04222a;--green:#37c46e;--green-d:#1f9a52;
--red:#f0554f;--red-d:#c93b36;--gray:#39424e;--field:#276b39;--field-ez:#1c4e2b;}
*{box-sizing:border-box;}
body{margin:0;background:var(--bg);color:var(--text);font-family:'Inter',system-ui,sans-serif;
-webkit-font-smoothing:antialiased;transition:background .2s,color .2s;}
.wrap{max-width:1060px;margin:0 auto;padding:20px 18px 60px;}
.disp{font-family:'Oswald','Inter',sans-serif;font-weight:600;letter-spacing:.02em;}
.topbar{display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;}
.title{font-family:'Oswald',sans-serif;font-weight:700;font-size:26px;letter-spacing:.06em;
text-transform:uppercase;margin:0;line-height:1;}
.sub{color:var(--muted);font-size:13px;margin-top:3px;}
.tbtn{background:var(--card);border:1px solid var(--border);color:var(--text);
width:42px;height:42px;border-radius:10px;font-size:18px;cursor:pointer;line-height:1;}
.panel{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:16px 18px;
display:grid;gap:14px 18px;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));margin-bottom:16px;}
.fld label{display:block;font-size:11px;text-transform:uppercase;letter-spacing:.05em;
color:var(--muted);margin-bottom:5px;font-weight:600;}
.fld input,.fld select{width:100%;height:42px;padding:0 10px;border-radius:9px;
border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:16px;
font-family:'Inter',sans-serif;}
.fld input:focus,.fld select:focus{outline:none;border-color:var(--accent);
box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 25%,transparent);}
.seg{display:flex;border:1px solid var(--border);border-radius:9px;overflow:hidden;}
.seg label{flex:1;text-align:center;padding:9px 6px;font-size:13px;cursor:pointer;
color:var(--muted);font-weight:600;margin:0;}
.seg input{display:none;}
.seg label:has(input:checked){background:var(--accent);color:var(--accent-ink);}
.clock{display:flex;gap:6px;}.clock .q{flex:1.4;}.clock .m,.clock .s{flex:1;}
.metatoggle{display:flex;align-items:center;gap:14px;margin-bottom:14px;flex-wrap:wrap;}
.metatoggle .seg{max-width:220px;}
.cap{color:var(--muted);font-size:12px;}
.banner{border-radius:14px;padding:14px 18px;margin-bottom:18px;background:var(--accent);
color:var(--accent-ink);display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;}
.banner .lab{font-size:12px;text-transform:uppercase;letter-spacing:.08em;opacity:.85;font-weight:600;}
.banner .pick{font-family:'Oswald',sans-serif;font-weight:700;font-size:26px;text-transform:uppercase;
letter-spacing:.03em;line-height:1;}
.banner .val{font-family:'Oswald',sans-serif;font-weight:600;font-size:22px;margin-left:auto;}
.cards{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));}
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:16px;
display:flex;flex-direction:column;gap:12px;}
.card.rec{border:2px solid var(--accent);}
.chd{display:flex;align-items:center;justify-content:space-between;gap:8px;}
.cname{font-family:'Oswald',sans-serif;font-weight:600;font-size:20px;text-transform:uppercase;
letter-spacing:.03em;}
.recbadge{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;
background:var(--accent);color:var(--accent-ink);padding:3px 8px;border-radius:20px;}
.cmetric{font-family:'Oswald',sans-serif;font-weight:700;font-size:30px;line-height:1;text-align:right;}
.cmetric small{display:block;font-size:10px;font-weight:600;letter-spacing:.06em;color:var(--muted);
text-transform:uppercase;}.cprob{font-size:12px;color:var(--muted);}
.field{position:relative;height:46px;border-radius:7px;background:var(--field);overflow:hidden;}
.ez{position:absolute;top:0;bottom:0;width:8%;background:var(--field-ez);}
.ez-l{left:0;}.ez-r{right:0;}
.yard{position:absolute;top:0;bottom:0;width:2px;background:rgba(255,255,255,.28);}
.ball{position:absolute;top:50%;transform:translate(-50%,-50%);width:15px;height:15px;
border-radius:50%;background:#fff;border:3px solid var(--green);box-shadow:0 1px 3px rgba(0,0,0,.4);}
.ball.opp{border-color:var(--red);}.ball.score{background:#f5b301;border-color:#f5b301;}
.postag{font-size:10px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;
text-align:center;color:var(--muted);}
.scores{display:flex;gap:14px;}.scores .col{flex:1;}
.big{font-family:'Oswald',sans-serif;font-weight:700;font-size:34px;line-height:1;}
.big.our{color:var(--green);}.big.opp{color:var(--red);}
.slab{font-size:12px;color:var(--muted);margin-left:4px;font-weight:600;}
.detail{font-size:12px;color:var(--muted);margin-top:2px;}
.dlt{font-size:12px;font-weight:600;margin-left:5px;}
.bar{display:flex;height:15px;border-radius:5px;overflow:hidden;gap:1.5px;background:var(--border);}
.foot{color:var(--muted);font-size:11px;margin-top:22px;line-height:1.6;}
</style></head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>"""


#------------------------------------------------------------------------------
# Input controls
#------------------------------------------------------------------------------
"""
The input panel: field position as a 1-50 yard number plus an Own/Opponent
segmented toggle, quarter + mm:ss, our and their score, yards to go, and the
kicker and punter ranges. Down is fixed at 4th, so there is no down input.
"""
def _num(id_, label, value, mn, mx, cls="fld"):
    return html.Div(className=cls, children=[
        html.Label(label),
        dcc.Input(id=id_, type="number", value=value, min=mn, max=mx, step=1)])


def input_panel():
    return html.Div(className="panel", children=[
        _num("yard_line", "Yard line", 40, 1, 50),
        html.Div(className="fld", children=[
            html.Label("Side of field"),
            dcc.RadioItems(id="side", value="opponent",
                           options=[{"label": "Own", "value": "own"},
                                    {"label": "Opp", "value": "opponent"}],
                           className="seg", inline=True)]),
        html.Div(className="fld", children=[
            html.Label("Quarter / clock"),
            html.Div(className="clock", children=[
                dcc.Dropdown(id="quarter", value=4, clearable=False, className="q",
                             options=[{"label": f"Q{i}", "value": i} for i in (1, 2, 3, 4)]
                             + [{"label": "OT", "value": 5}]),
                dcc.Input(id="minutes", type="number", value=8, min=0, max=15, className="m"),
                dcc.Input(id="seconds", type="number", value=0, min=0, max=59, className="s")])]),
        _num("ydstogo", "Yards to go", 3, 1, 99),
        _num("our_score", "Our score", 17, 0, 99),
        _num("their_score", "Their score", 21, 0, 99),
        _num("kicker_range", "Kicker range (FG yd)", 55, 20, 75),
        _num("punter_range", "Punter range (yd)", 60, 30, 80),
    ])


#------------------------------------------------------------------------------
# Card rendering
#------------------------------------------------------------------------------
"""
Each option renders a broadcast card: the headline metric (EPA in points or WPA
in win%), a real field strip with the ball at the resulting spot coloured by
possession, big green our-score and red opp-score with FG|TD splits and the Δ
against the current situation, and a 100% stacked next-score bar (green ours,
grey no-score, red theirs). The recommended card for the selected metric gets an
accent border.
"""
def _pct(x):
    return f"{round(x * 100)}%"


def _field(key, o):
    x = 8 + o["ball_pos"] * 0.84
    poss = o["possession"]
    yards = [html.Div(className="yard", style={"left": f"{8 + i * 8.4}%"}) for i in range(1, 10)]
    tag = {"us": "Your ball", "opp": "Opp ball",
           "score": "TD" if key == "go" else "FG good"}[poss]
    return html.Div([
        html.Div(className="field", children=[
            html.Div(className="ez ez-l"), html.Div(className="ez ez-r"), *yards,
            html.Div(className=f"ball {poss}", style={"left": f"{x}%"})]),
        html.Div(tag, className="postag", style={"marginTop": "5px"})])


def _bar(o):
    seg = [("our_td", "var(--green-d)"), ("our_fg", "var(--green)"),
           ("no_score", "var(--gray)"), ("opp_fg", "var(--red)"), ("opp_td", "var(--red-d)")]
    return html.Div(className="bar", children=[
        html.Div(style={"width": f"{o[k] * 100}%", "background": c}) for k, c in seg if o[k] > 0])


def _delta(d):
    up = d >= 0
    arrow = "\u25B2" if up else "\u25BC"
    return html.Span(f"{arrow} {abs(round(d * 100))}%",
                     className="dlt", style={"color": "var(--green)" if up else "var(--red)"})


def build_card(key, o, metric, recommended):
    if metric == "epa":
        val, unit = f"{o['epa']:+.2f}", "EPA"
    else:
        val, unit = f"{o['wpa'] * 100:+.1f}%", "WPA"
    prob = (f"{_pct(o['prob'])} {o['prob_label']}"
            + (f" \u00b7 {round(o['kick_distance'])} yd" if "kick_distance" in o else "")
            if o["prob"] is not None else "\u00a0")
    head = [html.Div([html.Span(NAMES[key], className="cname"),
                      html.Span("Pick", className="recbadge") if recommended else None],
                     style={"display": "flex", "alignItems": "center", "gap": "8px"}),
            html.Div([val, html.Small(unit)], className="cmetric")]
    return html.Div(className="card rec" if recommended else "card", children=[
        html.Div(head, className="chd"),
        html.Div(prob, className="cprob"),
        _field(key, o),
        html.Div(className="scores", children=[
            html.Div(className="col", children=[
                html.Div([html.Span(_pct(o["our_score"]), className="big our"),
                          html.Span("our score", className="slab"), _delta(o["our_score_delta"])]),
                html.Div(f"{_pct(o['our_fg'])} FG  |  {_pct(o['our_td'])} TD", className="detail")]),
            html.Div(className="col", children=[
                html.Div([html.Span(_pct(o["opp_score"]), className="big opp"),
                          html.Span("opp score", className="slab")]),
                html.Div(f"{_pct(o['opp_fg'])} FG  |  {_pct(o['opp_td'])} TD", className="detail")])]),
        _bar(o)])


#------------------------------------------------------------------------------
# Layout + callback
#------------------------------------------------------------------------------
"""
Assemble the page and wire the single results callback: any input or a flip of
the EPA/WPA toggle recomputes decide() and rebuilds the recommendation banner
and the three cards. A clientside callback handles the light/dark toggle.
"""
app.layout = html.Div(className="wrap", children=[
    html.Div(className="topbar", children=[
        html.Div([html.H1("4th Down Decision Tool", className="title"),
                  html.Div("Go / field goal / punt \u2014 expected points and win probability",
                           className="sub")]),
        html.Button("\u263e", id="theme-btn", className="tbtn", n_clicks=0,
                    title="Switch between light and dark mode")]),
    input_panel(),
    html.Div(className="metatoggle", children=[
        html.Div(className="fld", style={"marginBottom": "0"}, children=[
            html.Label("Rank options by"),
            dcc.RadioItems(id="metric", value="epa", className="seg", inline=True,
                           options=[{"label": "Expected points", "value": "epa"},
                                    {"label": "Win probability", "value": "wpa"}])]),
        html.Div("EPA is scoreboard points; WPA is change in win probability.", className="cap")]),
    html.Div(id="banner", className="banner"),
    html.Div(id="cards", className="cards"),
    html.Div(className="foot", children=[
        "Next-score figures are the chance the next score of the half is by each team, "
        "read at the resulting field position. Field-goal makes are valued at a literal "
        "+3 points on the EPA scale; go and punt use change-in-expected-points.",
    ]),
])


@app.callback(
    Output("banner", "children"), Output("cards", "children"),
    Input("yard_line", "value"), Input("side", "value"),
    Input("quarter", "value"), Input("minutes", "value"), Input("seconds", "value"),
    Input("our_score", "value"), Input("their_score", "value"),
    Input("ydstogo", "value"), Input("kicker_range", "value"),
    Input("punter_range", "value"), Input("metric", "value"))
def update(yard_line, side, quarter, minutes, seconds, our_score, their_score,
           ydstogo, kicker_range, punter_range, metric):
    def i(v, d):
        return d if v is None else v
    r = fd.decide(i(yard_line, 40), side or "opponent", i(quarter, 4),
                  i(minutes, 8), i(seconds, 0), i(our_score, 0), i(their_score, 0),
                  i(ydstogo, 3), i(kicker_range, 55), i(punter_range, 60))
    rec = r["rec_epa"] if metric == "epa" else r["rec_wpa"]
    o = r["options"][rec]
    val = f"{o['epa']:+.2f} EPA" if metric == "epa" else f"{o['wpa'] * 100:+.1f}% WPA"
    banner = [html.Span(f"Recommendation \u00b7 {'expected points' if metric=='epa' else 'win probability'}",
                        className="lab"),
              html.Span(NAMES[rec], className="pick"),
              html.Span(val, className="val disp")]
    cards = [build_card(k, r["options"][k], metric, k == rec) for k in ("go", "fg", "punt")]
    return banner, cards


app.clientside_callback(
    """function(n){ if(!n){return window.dash_clientside.no_update;}
      var el=document.documentElement, cur=el.getAttribute('data-theme');
      var sysDark=window.matchMedia('(prefers-color-scheme: dark)').matches;
      var next = cur ? (cur==='dark'?'light':'dark') : (sysDark?'light':'dark');
      el.setAttribute('data-theme', next);
      return next==='dark' ? '\\u2600' : '\\u263e'; }""",
    Output("theme-btn", "children"), Input("theme-btn", "n_clicks"))


if __name__ == "__main__":
    app.run(debug=True)
