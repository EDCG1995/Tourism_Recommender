from LocationLens import LocationLens
from surprise import SVD, SVDpp
from surprise import NormalPredictor
from Evaluator import Evaluator
from surprise.model_selection import GridSearchCV
import random
import numpy as np

def LoadLocationLensData():
    ll = LocationLens()
    print("Loading location ratings...")
    data = ll.loadLocationLens()
    print("\nComputing location popularity ranks so we can measure novelty later...")
    rankings = ll.getPopularityRanks()
    return (ll, data, rankings)

np.random.seed(0)
random.seed(0)

# Load up common data set for the recommender algorithms
(ll, evaluationData, rankings) = LoadLocationLensData()

print("Searching the best parameters for the SVD algorithm...")

param_grid = {'n_epochs': [40], 'lr_all': [0.0025], 'n_factors': [10]}

gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'])

gs.fit(evaluationData)

#best RMSE
print("Best RMSE", gs.best_score['rmse'])
#best parameters
print(gs.best_params['rmse'])


# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluationData, rankings)

params = gs.best_params['rmse']

SVDtuned = SVD(n_epochs = params['n_epochs'], lr_all = params['lr_all'], n_factors = params['n_factors'])
evaluator.AddAlgorithm(SVDtuned, "SVD - Tuned")

SVDUntuned = SVD()
evaluator.AddAlgorithm(SVDUntuned, "SVD - Untuned")


# Just make random recommendations
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

# Fight!
evaluator.Evaluate(True)

evaluator.SampleTopNRecs(ll, testSubject=25654)
