import os
import pandas as pd
import numpy as np
import requests as req
import json as JSON

URL = 'http://ergast.com/api/f1'

def fetch_and_save_jsons():
    years = [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

    for year in years:
        for round in range(1, 25):
            try:
                race = req.get(f'{URL}/{year}/{round}/results.json') # fetch race

                # make new json file and populate it with fetched race json
                with open(f'data/jsons/{year}_round{round}_race.json', 'w') as f:
                    JSON.dump(race.json(), f, indent=4)
            except Exception as e:
                print(f'An error occurred: {e}')
                continue

def clean_jsons():
    for json in os.listdir('data/jsons'):
        json_path = os.path.join('data/jsons', json) # build path of each json

        if os.path.isfile(json_path): 
            with open(json_path, 'r') as f: # load json into a variable for modification
                race = JSON.load(f)
            
            race = race['MRData']['RaceTable']['Races'][0] # remove extra info surrounding the needed data
            with open(json_path, 'w') as f: # overwite the original json with the modified one
                JSON.dump(race, f, indent=4)

def build_raw_dataframe():
    instances = [] # initialize collection of rows

    for race in os.listdir('data/jsons'):
        race_path = os.path.join('data/jsons', race) # build path of race

        if os.path.isfile(race_path):
            with open(race_path, 'r') as f: # load race json in a variable for extraction
                race_data = JSON.load(f)
        else:
            print(f'This file does not exist: {race_path}')
        
        # create a dictionary for this specific instance
        for result in race_data['Results']:
            instance = {}
            instance['finishing_pos'] = int(result['position'])
            instance['circuit_id'] = race_data['Circuit']['circuitId']
            instance['driver_id'] = result['Driver']['driverId']
            instance['constructor_id'] = result['Constructor']['constructorId']
            instance['grid_pos'] = int(result['grid'])
            instance['status'] = result['status'] if result['status'][0] != '+' else 'Finished'

            instance['fastest_lap_missing'] = 0
            try:
                instance['fastest_lap'] = result['FastestLap']['Time']['time']
                instance['fastest_lap_avg_speed'] = result['FastestLap']['AverageSpeed']['speed']
            except Exception:
                instance['fastest_lap'] = 300
                instance['fastest_lap_avg_speed'] = 0
                instance['fastest_lap_missing'] = 1
            
            instances.append(instance)
        raw_df = pd.DataFrame(instances)
        raw_df.to_csv('data/dataframes/raw.csv', index=False)