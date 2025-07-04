import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import random

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id="a108245395774fd8b4a5d30b1f228465",
                                               client_secret="f32dcdd4f1034baebb96843b19d8f2b1",
                                               redirect_uri="https://localhost:8000",
                                               scope="user-read-playback-state streaming ugc-image-upload playlist-modify-public"))

df1 = pd.read_csv(r'data_moods.csv')
fp=open(r'mood.txt','r')
mood = fp.read()
fp.close()

df2 = df1.loc[df1['mood'] == mood]
df2 = df2.astype({'id':'string'})
list_of_songs=[]
for row in df2.iterrows():
    list_of_songs.append("spotify:track:"+str(row[1]['id']))
print(len(list_of_songs))
if len(list_of_songs) == 0:
    quit()
if len(list_of_songs) >= 15:
    list_of_songs=random.sample(list_of_songs,15)
playlist_name = mood +' Songs'
playlist_description = mood +' Songs'
user_id = sp.me()['id']
sp.user_playlist_create(user=user_id,name=playlist_name,public=True,description=playlist_description)
prePlaylists = sp.user_playlists(user=user_id)
playlist = prePlaylists['items'][0]['id']
print(playlist)
sp.user_playlist_add_tracks(user=user_id, playlist_id=playlist, tracks=list(list_of_songs))
print("Created "+mood+" playlist")
fp=open(r'id.txt','w')
fp.write(playlist)
fp.close()
import os
os.system(r'python test2.py')