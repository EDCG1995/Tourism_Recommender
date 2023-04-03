import os
import csv
import sys
import re
import codecs
from surprise import Dataset
from surprise import Reader

from collections import defaultdict
import numpy as np

class LocationLens:

    locationID_to_name = {}
    name_to_locationID = {}
    ratingsPath = '..\\useritemDBs.csv'
    locationsPath = '..\\LocationsDBs.csv'
    
    def loadLocationLens(self):

        # Look for files relative to the directory we are running from
        os.chdir(os.path.dirname(sys.argv[0]))

        ratingsDataset = 0
        self.locationID_to_name = {}
        self.name_to_locationID = {}

        reader = Reader(line_format='user item rating', sep=',', skip_lines=1)

        ratingsDataset = Dataset.load_from_file(self.ratingsPath, reader=reader)

        types_of_encoding = ["utf8", "cp1252"]
        with open(self.locationsPath, newline='', encoding='ISO-8859-1') as csvfile:
                locationReader = csv.reader(csvfile)
                next(locationReader)  #Skip header line
                for row in locationReader:
                    locationID = int(row[0])
                    locationName = row[1]
                    self.locationID_to_name[locationID] = locationName
                    self.name_to_locationID[locationName] = locationID


        return ratingsDataset

    def getUserRatings(self, user):
        userRatings = []
        hitUser = False
        with open(self.ratingsPath, newline='') as csvfile:
            ratingReader = csv.reader(csvfile)
            next(ratingReader)
            for row in ratingReader:
                userID = int(row[1])
                if (user == userID):
                    locationID = int(row[0])
                    rating = float(row[2])
                    userRatings.append((locationID, rating))
                    hitUser = True
                if (hitUser and (user != userID)):
                    break

        return userRatings

    def getPopularityRanks(self):
        ratings = defaultdict(int)
        rankings = defaultdict(int)
        with open(self.ratingsPath, newline='') as csvfile:
            ratingReader = csv.reader(csvfile)
            next(ratingReader)
            for row in ratingReader:
                locationID = int(row[1])#might have to edit this number
                ratings[locationID] += 1
        rank = 1
        for locationID, ratingCount in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
            rankings[locationID] = rank
            rank += 1
        return rankings
    
    def getCategories(self):
        categories = defaultdict(list)
        categoryIDs = {}
        maxcategoryID = 0
        with open(self.locationsPath, newline='', encoding='ISO-8859-1') as csvfile:
            locationReader = csv.reader(csvfile)
            next(locationReader)  #Skip header line
            for row in locationReader:
                locationID = int(row[0])
                genreList = row[4].split('•')
                genreIDList = []
                for genre in genreList:
                    if genre in categoryIDs:
                        genreID = categoryIDs[genre]
                    else:
                        genreID = maxcategoryID
                        categoryIDs[genre] = genreID
                        maxcategoryID += 1
                    genreIDList.append(genreID)
                categories[locationID] = genreIDList
        # Convert integer-encoded genre lists to bitfields that we can treat as vectors
        for (locationID, genreIDList) in categories.items():
            bitfield = [0] * maxcategoryID
            for genreID in genreIDList:
                bitfield[genreID] = 1
            categories[locationID] = bitfield
        
        return categories
    
    # def getYears(self):
    #     p = re.compile(r"(?:\((\d{4})\))?\s*$")
    #     years = defaultdict(int)
    #     with open(self.locationsPath, newline='', encoding='ISO-8859-1') as csvfile:
    #         movieReader = csv.reader(csvfile)
    #         next(movieReader)
    #         for row in movieReader:
    #             movieID = int(row[0])
    #             title = row[1]
    #             m = p.search(title)
    #             year = m.group(1)
    #             if year:
    #                 years[movieID] = int(year)
    #     return years
    #
    # def getMiseEnScene(self):
    #     mes = defaultdict(list)
    #     with open("LLVisualFeatures13K_Log.csv", newline='') as csvfile:
    #         mesReader = csv.reader(csvfile)
    #         next(mesReader)
    #         for row in mesReader:
    #             movieID = int(row[0])
    #             avgShotLength = float(row[1])
    #             meanColorVariance = float(row[2])
    #             stddevColorVariance = float(row[3])
    #             meanMotion = float(row[4])
    #             stddevMotion = float(row[5])
    #             meanLightingKey = float(row[6])
    #             numShots = float(row[7])
    #             mes[movieID] = [avgShotLength, meanColorVariance, stddevColorVariance,
    #                meanMotion, stddevMotion, meanLightingKey, numShots]
    #     return mes
    #
    def getLocationName(self, locationID):
        if locationID in self.locationID_to_name:
            return self.locationID_to_name[locationID]
        else:
            return ""
        
    def getLocationID(self, locationName):
        if locationName in self.name_to_locationID:
            return self.name_to_locationID[locationName]
        else:
            return 0