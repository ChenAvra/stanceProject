# SYSTANCE - Stance Detection Project 
### By: Iris Dreizenshtok, Chen Avraham, Adi Avinun and Gal Tadir

We built a framework for comparing stance detection algorithms with 6 different datasets.

You can select up to three algorithms for comparison and one dataset each time.

The system will display the statistics of the algorithms and the dataset with the measures for each algorithm.

To run the backend please follow the instructions step by step (order is important): 

**Preparations:**

download FNC.csv file from
https://drive.google.com/file/d/1pfz4YHjWsglMCctd6RBAFo9kU_y-kMO6/view?usp=sharing and place it in Backend/DB folder

download glove.6b.5d.txt from https://www.kaggle.com/watts2/glove6b50dtxt and place it in Backend/LIU/data folder

download glove.6b.5d.txt from https://www.kaggle.com/watts2/glove6b50dtxt and place it in Backend/SEN folder and Backend/TRANSFORMER/embeddings

download glove.6b.300d.txt from https://www.kaggle.com/thanakomsn/glove6b300dtxt and place it in Backend/SEN folder and Backend/TAN

download glove.6b.zip from https://www.kaggle.com/anindya2906/glove6b and place it in Backend/SEN folder

download snli_1.0.zip from https://nlp.stanford.edu/projects/snli/snli_1.0.zip and place it in the Backend/SEN folder. Extract snli_1.0_dev.txt and place it in Backend/SEN folder

download GoogleNews-vectors-negative300.bin from https://www.kaggle.com/leadbest/googlenewsvectorsnegative300 and place it in Backend/Allada_Nandakumar

**Fill tables in DB:**
1. Go to Backend/DB/DBManager.py and remove comments from commands at the buttom of the page (right after functions declaration).
2. Run this file.
3. Return the comments (if you won't do it the tables will be duplocated every run).

**Run the code:**

For backend running, run service.py file

**Frontend part:**

To run the website you should also clone to frontend repository: https://github.com/irisDreizen/systance-frontend
Follow the README there.

**Important note:**

After your website is up please follow the instructions:
1. Go to main page.
2. click 'Compare Algorithms' button.
3. In datasets - choose semEval2016, In algorithms - choose TAN. Fill 65 for train percent and click on 'Run' botton.
4. Wait for the end of the running (that may take a few minutes...).

This part will allow you to run TAN algorithm in 'Check My Stance' part. If you are not willing to use it, you can ignore this note.


