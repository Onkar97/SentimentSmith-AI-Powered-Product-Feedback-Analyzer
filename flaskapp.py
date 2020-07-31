from flask import Flask, render_template, request, redirect, url_for, session, Markup, make_response
import pickle
import csv
from flask_mysqldb import MySQL
import MySQLdb.cursors
import pymysql
import os
import re
import numpy as np
from werkzeug.utils import secure_filename
import pandas as pd
import joblib
import json
from textblob import TextBlob
from collections import Counter

loaded_model=joblib.load("./pkl_objects/model.pkl")
loaded_vec=joblib.load("./pkl_objects/vectorizer.pkl")

app = Flask(__name__)

labels = [
    "Rating 1","Rating 2","Rating 3","Rating 4","Rating 5"]

colors = [
    "Red", "Green", "Yellow", "Blue",
    "Orange"]
    
labelbar = [ "Not Helpful", "Helpful"]

# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = 'your secret key'

# Enter your database connection details below
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'onkar'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'pythonlogin'

# Intialize MySQL
mysql = MySQL(app)

#Getting the rating of the product
def helpful(rating):
    if (rating > 1):
        return '1'
    return '0'
    
# Removing stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize, pos_tag_sents
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
lemmatizer = WordNetLemmatizer() 
stop = stopwords.words('english')
ps=PorterStemmer()

def getStemmedReview(reviews_text):
    reviews_text=reviews_text.lower()
    reviews_text=reviews_text.replace("<br /><br />"," ")
    #Tokenize
    tokens=word_tokenize(reviews_text)
    new_tokens=[token for token in tokens if token not in stop]
    lemmatize_tokens = [lemmatizer.lemmatize(token) for token in new_tokens]
    clean_review=' '.join(lemmatize_tokens)

    return clean_review

# Apply Sentiments
def sentiments(rating):
    if (rating == 5) or (rating == 4):
        return '1'
    elif (rating == 3) or (rating == 2) or (rating == 1):
        return '0'

# Function returning only nouns from reviews
def getNoun(array):
    noun_array = []
    
    for j in array:
        blob = TextBlob(j)
        x=blob.noun_phrases
        noun_array.extend(x)
        
    return noun_array
    
#Function returning maximum count and words
def getMaxCount(document):
    counts = Counter(document)
    count = counts.most_common(5)
    maxword=[]
    maxcount=[]
    
    for letter, count in count:
        maxword.append(letter)
        maxcount.append(count)
    
    return maxword, maxcount

def classify(document):
    df = pd.read_csv(document)
    df["Sentiment"] = df["Rating"].apply(sentiments)
    df["ReviewText"] = df["ReviewText"].astype(str)
    df["ReviewText"]=df["ReviewText"].apply(getStemmedReview)
    
    
    N=np.array(df['ReviewText'].loc[df['Sentiment'] == '0'])
    P=np.array(df['ReviewText'].loc[df['Sentiment'] == '1'])
        
    Positive = getNoun(P)
    Negative = getNoun(N)
      
    maxpositive, maxpositivecount = getMaxCount(Positive)
    maxnegative, maxnegativecount = getMaxCount(Negative)

    
    strat_test = df.dropna(subset=["Rating"])
    strat_test = df.dropna(subset=["Vote"])
    strat_test["Rating"] = strat_test["Rating"].astype(int)
    strat_test["helpful"] = df["Vote"].apply(helpful)
    strat_test["helpful"] = strat_test["helpful"].astype(int)
    X_test = strat_test["ReviewText"]
    X_test = X_test.fillna(' ')
    rate1 = len(strat_test[strat_test["Rating"] == 1])
    rate2 = len(strat_test[strat_test["Rating"] == 2])
    rate3 = len(strat_test[strat_test["Rating"] == 3])
    rate4 = len(strat_test[strat_test["Rating"] == 4])
    rate5 = len(strat_test[strat_test["Rating"] == 5])
    
    numhelpful = len(strat_test[strat_test["Vote"] == 1])
    
    numnothelpful = len(strat_test[strat_test["Vote"] == 0])
    
    values = [
        rate1, rate2, rate3, rate4, rate5 ]
        
    barvalues = [ numnothelpful, numhelpful]
    
    X = loaded_vec.transform(X_test.astype('U'))
    y = loaded_model.predict(X)[0]
    proba = np.max(loaded_model.predict_proba(X))
    return proba, values, barvalues, maxpositive, maxpositivecount, maxnegative, maxnegativecount
    
@app.route('/login/', methods=['GET', 'POST'])
def login():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s AND password = %s', (username, password))
        # Fetch one record and return result
        account = cursor.fetchone()
        # If account exists in accounts table in out database
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            # Redirect to home page
            return redirect(url_for('home'))
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
    # Show the login form with message (if any)
    return render_template('login.html', msg=msg)
    
@app.route('/logout')
def logout():
   
   # Remove session data, this will log the user out
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   # Redirect to login page
   return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s', [username])
        account = cursor.fetchone()
        
        # If account exists show error and validation checks
        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not username or not password or not email:
            msg = 'Please fill out the form!'
        else:
           
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute('INSERT INTO accounts VALUES (NULL, %s, %s, %s)', (username, password, email))
            mysql.connection.commit()
            msg = 'You have successfully registered!'
            # Parent Directory path 
            parent_dir = "./Users"
   
            path = os.path.join(parent_dir, username) 
            os.mkdir(path)
 
        
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'
    # Show registration form with message (if any)
    return render_template('register.html', msg=msg)
   
@app.route('/home')
def home():
    # Check if user is loggedin
    if 'loggedin' in session:
        user = session['username']
        
        cur = mysql.connection.cursor()
        cur.execute('SELECT product_name, owner, uploaded_date, product_price FROM product WHERE user_name = %s', [user])
        data = cur.fetchall()
        # User is loggedin show them the home page
        return render_template('home.html', username = user, tables=data)
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))
    
@app.route('/home/<string:product_name>')
def drops(product_name):
    productname= product_name
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('DELETE FROM product WHERE product_name = %s', [productname])
    cursor.execute('DELETE FROM product_reviews WHERE product_name = %s', [productname])
    mysql.connection.commit()
        
    return redirect(url_for('home'))
        
@app.route('/productform')
def productform():
    if 'loggedin' in session:
        return render_template('productform.html')
    return redirect(url_for('login'))

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST' and 'product_name' in request.form:
        prod_name = request.form.get('product_name')
        
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM product WHERE product_name = %s', [prod_name])
        product_details = cursor.fetchone()
        
        return render_template('feedback.html', data=product_details)

@app.route('/productform', methods=['GET', 'POST'])
def homes():
    # Check if user is loggedin
    if 'loggedin' in session:
        
        user = session['username']
        # Output message if something goes wrong...
        msg = ''
        # Check if "username" POST requests exist (user submitted form)
        if request.method == 'POST' and 'productname' in request.form:
            # Create variables for easy access
            productname = request.form['productname']
            owner = request.form['manufacturer']
            productsize = request.form['productsize']
            productcolour = request.form['productcolour']
            productprice = request.form['productprice']
            productshape = request.form['productshape']
            addinformation = request.form['addinformation']
            productimage = request.files.get('productimage')
                    
            # Check if product exists using MySQL
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute('SELECT * FROM product WHERE product_name = %s', [productname])
            product = cursor.fetchone()
        
            # If account exists show error and validation checks
            if product:
                msg = 'Product already exists!'
        
            else:
                # Product doesnt exists and the form data is valid, now insert new product into product table
                
                UPLOAD_FOLDER = './static/images'
                app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
                filename = secure_filename(productimage.filename)
                filenames = productname + ".png"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filenames)
                productimage.save(filepath)
		
		# Saving product name as csv file in user folder.
                Users = './Users'
                path = os.path.join(Users, user)
                app.config['UPLOAD_FOLDER'] = path  
                
                csvfile = productname + ".csv"
                df = pd.DataFrame()
                df.to_csv(os.path.join(app.config['UPLOAD_FOLDER'],csvfile), header = False, index=False)
                
                cursor.execute('INSERT INTO product VALUES (NULL, %s, %s, %s, CURDATE(), %s, %s, %s, %s, %s, %s)', (user, productname, owner,  productsize, productcolour, productprice, productshape, addinformation, filepath))
                mysql.connection.commit()   
                msg = "You have successfully registered the product !"  
            
        # User is loggedin show them the home page
        return render_template('productform.html', username=session['username'], msg = msg)
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

@app.route('/profile')
def profile():
    # Check if user is loggedin
    if 'loggedin' in session:
    
        # We need all the account info for the user so we can display it on the profile page
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE id = %s', [session['id']])
        account = cursor.fetchone()
        
        return render_template('profile.html', account=account)
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

@app.route('/result')
def result():
    if 'loggedin' in session:
        user = session['username']
        
        cur = mysql.connection.cursor()
        cur.execute('SELECT product_name FROM product WHERE user_name = %s', [user])
        details = cur.fetchall()
        
        return render_template('result.html', list=details)
    return redirect(url_for('login'))

ALLOWED_EXTENSIONS = 'csv'
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           
@app.route('/view',methods=['POST'])
def view():    
    if 'loggedin' in session:   
        
        user = session['username']
        
        UPLOAD_FOLDER = './Users'
        path = os.path.join(UPLOAD_FOLDER, user)
        app.config['UPLOAD_FOLDER'] = path 
    
        #Fetching product name and save it to CSV file
        if request.method == 'POST':  
        
            productname = request.form['select_product']
            cursordata = mysql.connection.cursor()
            cursordata.execute('SELECT uploaded_date, owner FROM product WHERE product_name = %s', [productname])
            dataproduct = cursordata.fetchone()
            mysql.connection.commit()
            
            cur = mysql.connection.cursor()
            cur.execute('SELECT user_email, product_rating, product_text, vote FROM product_reviews WHERE product_name = %s', [productname])
            details = cur.fetchall()
            
            if details:
                df = pd.DataFrame(details) 
            
                filename = productname + ".csv"  
                # saving the dataframe 
                fullpath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                df.to_csv(fullpath, header = ['User','Rating','ReviewText','Vote'], index=False) 
            
                proba, values, valuesbar, maxpos, maxposcnt, maxneg, maxnegcnt = classify(fullpath)
                likes = str(round(proba*100, 2))
                msglike = likes + '% of people like your product.'
                dislikes = str(round(100-(float(likes)),2))
                msghate = dislikes + '% of people dislike your product.'
                
                return render_template('view.html', name = productname, productdata = dataproduct, likemsg=msglike, hatemsg=msghate, ratingtitle='Review Ratings', max = 100, linelabels=labels, linevalues = values, piedata=zip(values, labels, colors), pietable=zip(values, labels, colors), helpfultitle = "Product votes", barlabels = labelbar, barvalues = valuesbar, bartable = zip(labelbar, valuesbar, colors), gaugedata = zip(valuesbar, labelbar, colors), analyzeposdata = zip(maxpos, maxposcnt), analyzenegdata = zip(maxneg, maxnegcnt))
            else:
                msg = 'Your Product does not have any review yet!'
                return render_template('view.html', name = productname, productdata = dataproduct, msg = msg)
                

 
@app.route('/upload', methods = ['GET', 'POST'])
def upload():
    if 'loggedin' in session:   
        
        user = session['username']
        
        UPLOAD_FOLDER = './Users'
        path = os.path.join(UPLOAD_FOLDER, user)
        app.config['UPLOAD_FOLDER'] = path 
    
        if request.method == 'POST':
            name = request.form['pname']  
            if 'file' not in request.files:
                msg = 'No file part'
                return redirect(request.url)
            
            file = request.files['file']
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                msg = 'No selected file'
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                msg = 'File Uploaded Successfully.'
                proba, values, valuesbar, maxpos, maxposcnt, maxneg, maxnegcnt = classify(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                likes = str(round(proba*100, 2))
                msglike = likes + '% of people like your product.'
                dislikes = str(round(100-(float(likes)),2))
                msghate = dislikes + '% of people dislike your product.'
                
                return render_template('upload.html', name = name, likemsg=msglike, hatemsg=msghate, ratingtitle='Review Ratings', max = 10000, linelabels=labels, linevalues = values, piedata=zip(values, labels, colors), pietable=zip(values, labels, colors), helpfultitle = "Product votes", barlabels = labelbar, barvalues = valuesbar, bartable = zip(labelbar, valuesbar, colors), gaugedata = zip(valuesbar, labelbar, colors), analyzeposdata = zip(maxpos, maxposcnt), analyzenegdata = zip(maxneg, maxnegcnt))

    return redirect(url_for('result')) 
    
@app.route('/')
def mainpage(): 
    cur = mysql.connection.cursor()
    cur.execute('SELECT * FROM product')
    details = cur.fetchall()
   
    return render_template('mainpage.html', images = details)        
    
    
@app.route('/', methods=['POST'])
def storeReview():

    if request.method == 'POST':
        productname = request.form['product_name']
        email = request.form['useremail']
        rate = request.form['productRating']
        text = request.form['ReviewText']
        user_vote = request.form['vote']    

        #Save into Database Table : product_reviews
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('INSERT INTO product_reviews VALUES (NULL, %s, %s, %s, %s, %s)', (productname, email, rate, text, user_vote))
        mysql.connection.commit()
        
    return redirect(url_for('mainpage'))    
              
if __name__ == '__main__':
    app.run(debug=True)
