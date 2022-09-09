<h1 align="center" style="padding-bottom: 30px">
    The NBA Problem
</h1 >
  <p align="center" style="font-size: 1.2rem">Finally making my algorithms professor happy by solving a problem that doesn't really need to be solved.</p>

<hr />

## Background on the Problem
DraftKings runs a fantasy style game where you draft players on a limited budget. You are allowed to select from all players who have games for that given day. For instance, you could have Lakers vs. Bucks and Heat vs. Nets on the lineup and will only be ablet to select from those players. 

Now, this game is challenging for a few reasons. The first is that each player has a given cost. For instance, your LeBron's and Durant's of the world will be very expensive (and sometimes over-valued @westbrook). The second is that you have to fill out a roster of positions - C, PF, SG, etc. The third, is that an NBA players performance for a given night has a high amount of variance and is difficult to predict. 

Putting the third issue aside, let's look at issues one and two. 

## Defining the Problem
When I first started playing the game on DraftKings, I thought to myself, "This seems like an optimization problem". Then I thought even harder - "This seems like the knapsack problem". It wan't the knapsack problem. But, it definitely had constraints and definitely had a goal to optimize over. So could it be solved using one of those algorithms I learned about back in the day? Maybe. Probably. Yes? Let's define some parameters.

### Contraints
So this problem has a few constraints, more specifically:

1. Each player has a given price and you cannot go over your given budget. 
2. Each player can play a certain position. 
3. You have to draft 1 player for each of the following positions: PG, SG, SF, PF, C, G, F, UTIL. 

With those constraints in mind, we can move on to the optimization. 

### Optimization
Your goal is to draft a lineup that will result in the most points on your fixed budget. Points for each player are determined based on their stat line for the night and is a functioin of actual points, assists, rebounds, steals, blocks, etc. Therefore, your goal is to maximize the sum of points scored by each player in your lineup. 

### Python at Work for the ILP
Putting all of this together, we can take use the `cvxpy` Python library to quickly construct our solver to determine the optimal lineup for a given set of player stats *after* the games have been played. 

```python
# Constraints
constraints = [
    weights@cp.sum(selections, axis=1) <= W,
    buckets[0, :]@selections[:, 0] == 1, # PG position constraint
    buckets[1, :]@selections[:, 1] == 1, # SG position constraint
    buckets[2, :]@selections[:, 2] == 1, # SF position constraint
    buckets[3, :]@selections[:, 3] == 1, # PF position constraint
    buckets[4, :]@selections[:, 4] == 1, # C position constraint
    buckets[5, :]@selections[:, 5] == 1, # G position constraint
    buckets[6, :]@selections[:, 6] == 1, # F position constraint
    buckets[7, :]@selections[:, 7] == 1, # UTIL position constraint
    cp.sum(selections) == len(positions),
    cp.sum(selections, axis=1) <=1
]

# Optimization
optimization = objective = cp.Maximize(values@cp.sum(selections, axis=1))
```

Then all we have to do is run solve! 

```python
# Create the problem
knapsack_problem = cp.Problem(objective, constraints)

# And solve it
result = knapsack_problem.solve()
```

## What's the Catch
So this sounds great and all, but there's one major issue to running this on a daily basis and winning all games - the optimization assumes that you know how many points a player *will* score for a given night. However, predicting how many points a player will score is **much** more difficult. 

Sill, I wanted to see how well the model would perform using the most basic prediction model - the player's running average for points. And the model worked... okay. But improving point predictions is definitely a challenge for a different day. 
