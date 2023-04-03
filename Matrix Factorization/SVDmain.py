from collections import defaultdict
import surprise.model_selection
from EvaluationData import EvaluationData
from LocationLens import LocationLens
from surprise import SVD, accuracy, dump

import random
import numpy as np

def LoadLocationLensData():
    ll = LocationLens()
    print("Loading location ratings...")
    data = ll.loadLocationLens()
    print("\nComputing location popularity ranks so we can measure novelty later...")
    rankings = ll.getPopularityRanks()
    return (ll, data, rankings)

def user_pred_top_n(model,ed, uid = 16624, iid = 10, n = -1):
    if n > -1:
        print(f'generate top {n} predictions for user {uid} ')
        testset = set(ed.GetAntiTestSetForUser(uid))
        print(len(testset))
        raw_predictions = []
        for rec in testset:
            raw_predictions.append(model.predict(rec[0], rec[1], r_ui=4, verbose=True))
        raw_predictions.sort(key=lambda x: x.est, reverse=True)
        raw_predictions = raw_predictions[:n]
        predictions = []
        for rec in raw_predictions:
            predictions.append( ( ll.getLocationName(int(rec[1])) , rec[3]) )
        print(predictions)


    else:
        print(f'generate prediction for user {{uid}} and location {{iid}}')
        uid = str(uid)
        iid = str(iid)
        pred = model.predict(uid, iid, r_ui=4, verbose=True)
#
# def get_top_n(predictions,user,  n=10):
#
#
#     testAntiSet = ed.GetAntiTestSetForUser(user)
#     #computing recommendations
#     recommendations = []
#     print('We recommend:')
#     for userID, locationID, actualRating, estimatedRating, _ in predictions:
#         intLocationID = int(locationID)
#         if (intLocationID, estimatedRating) not in recommendations:
#             recommendations.append((intLocationID, estimatedRating))
#
#     recommendations.sort(key=lambda x: x[1], reverse=True)
#
#     for ratings in recommendations[:n]:
#         print(ll.getLocationName(ratings[0]), ratings[1])
#     return recommendations[:n]
model = []
try:
    model = dump.load('MODEL_FILE2')
    model= model[1]
except:
    pass

if model:
    (ll, evaluationData, rankings) = LoadLocationLensData()
    ed = EvaluationData(evaluationData, rankings)
    user_pred_top_n(model, n=10, ed=ed)

    # generate top 10 recommendations for user 9067

else:

    np.random.seed(0)
    random.seed(0)

    # Load up common data set for the recommender algorithms
    (ll, evaluationData, rankings) = LoadLocationLensData()
    ed = EvaluationData(evaluationData, rankings)
    recommender = SVD( n_epochs=40, lr_all=0.0025, n_factors=10)
    trainset, testset = surprise.model_selection.train_test_split(evaluationData, test_size=.10, random_state=1)
    predictions = recommender.fit(trainset).test(testset)
    #Get RMSE, MAE, MSE
    accuracy.rmse(predictions)
    accuracy.mae(predictions)
    accuracy.mse(predictions)

    file_name = 'MODEL_FILE2'
    dump.dump(file_name, algo=recommender)
    #generate prediction for user 9067 and location 125

