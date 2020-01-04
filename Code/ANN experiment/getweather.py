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

def get_weather(api_key, location, time):
    url = "http://history.openweathermap.org/data/2.5/history/city?q={},{}&type=hour&start={}&units=metric&appid={}".format(location,'UK',time, api_key)
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
    return unixtime

def get_location(data):
    HtoL = pd.read_csv('HomeTeamLocation.csv')
    H = data['HomeTeam']
    L =[]
    print(set(data['HomeTeam'].unique())-set(HtoL['HomeTeam'].unique()))
    print(HtoL)

def main():
    training_data=pd.read_csv('epl-training.csv')

    location = 'London'
    training_data['UT'] = get_time(training_data)
    print(training_data)
    print(training_data['HomeTeam'].nunique())
    get_location(training_data)
    """api_key = get_api_key()
    weather = get_weather(api_key, location, get_time(1))

    #print(weather['main']['temp'])
    print(weather)"""


if __name__ == '__main__':
    main()
