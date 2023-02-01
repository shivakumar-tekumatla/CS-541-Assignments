import pickle 

loaded_model = pickle.load(open("model.sav", 'rb'))

print(loaded_model[-1,:])


