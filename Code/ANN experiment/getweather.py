import configparser
import requests
import sys
import time
import datetime
import pandas as pd

def get_api_key():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config['openweathermap']['api']

def get_weather(api_key, location,time):
    url = "https://api.openweathermap.org/data/2.5/weather?q={}&units=metric&appid={}&type=hour&start={}".format(location, api_key, time)
    r = requests.get(url)
    return r.json()

def get_time(data):
    T = []
    unixtime = []
    Y = pd.to_datetime(data['Date'],format='%d/%m/%Y').dt.year
    M = pd.to_datetime(data['Date'],format='%d/%m/%Y').dt.month
    D = pd.to_datetime(data['Date'],format='%d/%m/%Y').dt.day
    for i in range(len(data.index)):
        T.append(datetime.date(Y[i],M[i],D[i]))
        unixtime.append(time.mktime(T[i].timetuple()))
    data['UT'] = unixtime
    return data

def get_location(data):
    HtoL = pd.read_csv('HomeTeamLocation.csv')
    data['Location']=data.apply(lambda df: HtoL[HtoL['HomeTeam']==df['HomeTeam']]['Location'].values[0] , axis=1)
    return data

def main():
    training_data=pd.read_csv('epl-training.csv')

    location = 'London'
    training_data = get_time(training_data)
    training_data = get_location(training_data)
    print(training_data)
    api_key = get_api_key()
    location = training_data['Location'][4178]
    time = training_data['UT'][4178]
    print(location)
    weather = get_weather(api_key, location, time)

    #print(weather['main']['temp'])
    print(weather)


if __name__ == '__main__':
    main()
