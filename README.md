# 4th Down Decision Tool

A web app that helps you decide what to do on fourth down: go for it, kick a
field goal, or punt. You type in the situation and it shows the expected value
of each choice and which one comes out ahead.

The app runs in any browser, so you can open it on a phone, tablet, or laptop.
There is a light and dark mode button in the top corner.

## For coaches

### What you enter

* Yard line and side of the field (for example, the 40, opponent side).
* Quarter and time on the clock.
* Your score and their score.
* Yards to go.
* Kicker range: the longest field goal you would actually let your kicker try
  before it becomes a heave. Enter it as a field goal distance, so 55 means you
  trust him out to a 55 yard field goal.
* Punter range: how far your punter usually kicks it.

Down is always fourth, so there is no down to enter.

### What you get back

Three cards, one for each choice. Each card shows:

* A field with a marker for where the ball ends up and who has it. Green marker
  means you keep it, red means the other team gets it.
* Your chance to score next (green) and their chance to score next (red), with
  the field goal and touchdown split under each.
* A bar showing the full picture at a glance. More green is good for you, more
  red is good for them.
* The value of the choice on the metric you picked.

At the top is the recommended choice.

### The two views

You can rank the choices two ways using the toggle:

* Expected points: how many points the choice is worth on average.
* Win probability: how much the choice changes your chance of winning the game.

These can point to different choices, and that is on purpose. Late in a game
when you are behind, going for it can be the smart play for your win chances
even if a punt looks fine on points. Look at both and use your judgment.

### A note on the numbers

"Chance to score next" is the chance that the next score in the half is by that
team, measured from where the ball would end up. It is not the chance that this
one drive ends in a score.

On the expected points view, a made field goal counts as a flat plus 3 points.
The go and punt numbers on that view are change in expected points instead. So
the field goal number on the points view sits on a slightly different footing
than the other two. The win probability view does not have this quirk.

## Technical overview

The project has two halves. Training happens once in a while on your computer
and writes model files. The web app only reads those files, so it stays fast and
free to host.

### Files

* `train.py`: pulls the data, builds the models and tables, and saves them into
  `models/` and `data/`. Run this to refresh the numbers.
* `fourth_down.py`: the decision logic. It loads the saved files and turns a
  situation into the three options. It has no web code, so it is easy to test.
* `app.py`: the Dash web app. It calls `fourth_down.decide()` and draws the
  cards.
* `render.yaml` and `requirements.txt`: deploy settings for Render.
* `requirements-train.txt`: extra packages you need locally to run `train.py`.

### The data

Play by play from nflverse, pulled with `nfl_data_py`. Regular season only,
2018 to 2019 and 2021 through 2025. To add a season, put the year in the `SEASONS` list at the
top of `train.py` and run it again.

### How each choice is valued

Every choice is judged the same way: by the game state it leaves you in. That is
the standard method behind public fourth down charts, and it keeps go, field
goal, and punt on one honest footing.

For the points view, each choice is worth the expected points at the field
position it produces. Going for it is the chance of converting times the points
at the new spot, plus the chance of failing times the points you hand the
opponent at the line of scrimmage. A field goal is the make chance times 3
points, plus the miss chance times the points you hand the opponent at the spot
of the kick. A punt is the points you hand the opponent where the ball is downed,
which is why a stronger punter, pinning them deeper, improves it.

For the win view, a win probability model reads your win chance at that same
resulting state, with the score and clock folded in.

The conversion chance comes from an XGBoost model. It is used over a plain
logistic model because the effect of yards to go is far from a straight line: a
fourth and 1 is much easier than a fourth and 3, and the tree model matches the
real conversion rates by distance, where a logistic model flattens that edge. On
a convert the ball is spotted at the line to gain plus one yard, or a touchdown
if it reaches the end zone.

### Updating each year

1. Install the training packages: `pip install -r requirements-train.txt`.
2. Add the new season to `SEASONS` in `train.py`.
3. Run `python train.py`.
4. Commit the new `models/` and `data/` files.

Render redeploys and serves the new numbers. Install the packages from
`requirements.txt` before you train so the saved models match what the server
loads.

### Run it locally

```
pip install -r requirements-train.txt
python train.py
python app.py
```

Then open the address it prints.

### Deploy

Render runs `gunicorn app:server` on the free plan. The app only reads the saved
files, so it does not pull data or train at run time.

### Things to keep in mind

* Use the expected points view as your default. It is the reliable one and it
  matches public fourth down charts.
* The win probability view is most useful late in a close game. Earlier on, win
  probability barely moves across the field, so it is nearly flat and tends to
  favor the safe choice. It also under-rates going for it when you are trailing,
  because a simple win-probability lookup cannot see that punting there wastes
  one of your few remaining possessions. Getting that fully right needs a much
  heavier model than a free lightweight tool can carry.
* Rare situations have less data behind them, so treat the edges with more
  caution.
* On the points view a made field goal counts as a flat plus 3 points, while go
  and punt are measured as a change in expected points, so the field goal number
  sits on a slightly different footing than the other two. The win view does not
  have this quirk.
