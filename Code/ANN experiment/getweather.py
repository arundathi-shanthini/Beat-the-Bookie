#Importing libriaries
import configparser
import requests
import sys
import time
import datetime
import pandas as pd

#get the api key for the weather api form a seperate file
def get_api_key():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config['openweathermap']['api']
#obtain the weather from the api
def get_weather(api_key, location,time):
    url = "https://api.openweathermap.org/data/2.5/weather?q={}&units=metric&appid={}&type=hour&start={}".format(location, api_key, time)
    r = requests.get(url)
    return r.json()
#convert dates in the dataset to unix time for the api to use
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
#obtain the locaton of the home team based on a seperate csv file
def get_location(data):
    HtoL = pd.read_csv('HomeTeamLocation.csv')
    data['Location']=data.apply(lambda df: HtoL[HtoL['HomeTeam']==df['HomeTeam']]['Location'].values[0] , axis=1)
    return data
#main code
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
#as we have not paid for the api we have failed to obtail the historical data thus we will not be using weather as a feture in the training or the prediction
