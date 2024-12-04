from fastapi import FastAPI, status, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import os
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_folder = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_folder), name="static")

csv1path = os.path.join(static_folder, "1662574418893344.csv")
model_path = os.path.join(static_folder, "recommender_model.pkl")
rating_csv = os.path.join(static_folder, "ratings.csv")

# Load data
try:
    df = pd.read_csv(csv1path)
    rating = pd.read_csv(rating_csv)
    print("Files loaded successfully")
except Exception as e:
    print(f"Error loading files: {e}")

rating_matrix = rating.pivot_table(index='Food_ID', columns='User_ID', values='Rating').fillna(0)

def Get_Recommendations(title):
    user = df[df['Name'] == title]
    
    if user.empty:
        raise HTTPException(status_code=404, detail="Food item not found")

    user_index = np.where(rating_matrix.index == int(user['Food_ID'].values[0]))[0]
    
    if user_index.size == 0:
        raise HTTPException(status_code=404, detail="User index not found in rating matrix")
    
    user_ratings = rating_matrix.iloc[user_index[0]]

    reshaped = user_ratings.values.reshape(1, -1)
    csr_rating_matrix =  csr_matrix(rating_matrix.values)
    recommender = NearestNeighbors(metric='cosine')
    recommender.fit(csr_rating_matrix)
    distances, indices = recommender.kneighbors(reshaped, n_neighbors=16)

    nearest_neighbors_indices = rating_matrix.index[indices[0][1:]]
    nearest_neighbors = pd.DataFrame({'Food_ID': nearest_neighbors_indices})

    result = pd.merge(nearest_neighbors, df, on='Food_ID', how='left')

    return result.head().to_dict(orient='records')


class FormSchema(BaseModel):
    title:str

@app.post("/api/result")
def recommended(form: FormSchema):
    try:
        return Get_Recommendations(form.title)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))