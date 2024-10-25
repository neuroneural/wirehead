from wirehead import MultiHeadDataset as WireheadDataset 

dataset = WireheadDataset(config_path = "config.yaml")

for i in range(100):
    idx = [i%10]
    data = dataset[idx]
    #a,b = data[0][0], data[0][1]
    a,b,c,d = data[0][0], data[0][1] , data[0][2], data[0][3]
    print("Loader, index: ", i)

print("Unit example passed successfully")
