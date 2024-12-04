import pickle



# To load it later
with open('recommender_model.pkl', 'rb') as f:
    recommender_loaded = pickle.load(f)

output = recommender_loaded("tricolour salad")

