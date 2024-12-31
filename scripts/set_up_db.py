"""
This script will be used to create the SQLite3 Database and set uo the tables.
Once all the tables have been set up and created, the api client can scrape articles ,
and store them.
"""
import sqlite3
import json

DB_PATH = 'scripts/articles.db'
TABLE_CONFIG_PATH = 'scripts/configs/table_config.json'

if __name__  == '__main__':
    config_handle = open(TABLE_CONFIG_PATH, 'r')
    config_handle = json.loads(config_handle.read())
    #Create DB if does not exists
    connection = sqlite3.connect(DB_PATH)
    for table_name, query in config_handle.items():
        cursor = connection.cursor()
        cursor.execute(config_handle[table_name])