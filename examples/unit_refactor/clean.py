from pymongo import MongoClient
import yaml

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def delete_database(client, db_name):
    if client[db_name] != None:
        client.drop_database(db_name)
    print(f"Deleted database: {db_name}")

def main():
    # Load the configuration from config.yaml
    config = load_config('config.yaml')
    
    # Connect to the MongoDB server
    client = MongoClient(f"mongodb://{config['MONGOHOST']}:27017/")
    
    # Delete the specified database
    delete_database(client, config['DBNAME'])
    
    # Close the MongoDB connection
    client.close()

if __name__ == "__main__":
    main()
