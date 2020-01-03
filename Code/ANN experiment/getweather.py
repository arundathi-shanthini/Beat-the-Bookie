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

def get_time(date):
    d = datetime.date(2015,1,5)
    unixtime = time.mktime(d.timetuple())
    return unixtime

def main():
    training_data=pd.read_csv('epl-training.csv')

    location = 'London'

    api_key = get_api_key()
    weather = get_weather(api_key, location, get_time(1))
    print(get_time(1))
    #print(weather['main']['temp'])
    print(weather)


if __name__ == '__main__':
    main()
