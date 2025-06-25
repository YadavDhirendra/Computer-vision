fp = open(r"new.html","w")
f2 = open(r"id.txt","r")
p=f2.read()
f2.close()
fp.write('''
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
  
    <title>HTML 5 Boilerplate</title>
  </head>
  <body background="spotifybg.jpeg">
  
	<script src="index.js"></script>
    <iframe style="border-radius:12px" src="https://open.spotify.com/embed/playlist/'''+p+'''?utm_source=generator" width="100%" height="800" frameBorder="0" allow="autoplay; clipboard-write; 
    encrypted-media; picture-in-picture" loading="lazy"></iframe>

  </body>
</html>''')

fp.close()

import os
os.system(r"open new.html")
