from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# load the model from disk
filename = 'data.pkl'
clf = pickle.load(open(filename, 'rb'))
# cv=pickle.load(open('tranform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('demo.html')

@app.route('/predict',methods=['POST'])
def predict():
#	df= pd.read_csv("spam.csv", encoding="latin-1")
#	df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
#	# Features and Labels
#	df['label'] = df['class'].map({'ham': 0, 'spam': 1})
#	X = df['message']
#	y = df['label']
#	
#	# Extract Feature With CountVectorizer
#	cv = CountVectorizer()
#	X = cv.fit_transform(X) # Fit the Data
#    
#    pickle.dump(cv, open('tranform.pkl', 'wb'))
#    
#    
#	from sklearn.model_selection import train_test_split
#	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#	#Naive Bayes Classifier
#	from sklearn.naive_bayes import MultinomialNB
#
#	clf = MultinomialNB()
#	clf.fit(X_train,y_train)
#	clf.score(X_test,y_test)
#    filename = 'nlp_model.pkl'
#    pickle.dump(clf, open(filename, 'wb'))
    
	#Alternative Usage of Saved Model
	# joblib.dump(clf, 'NB_spam_model.pkl')
	# NB_spam_model = open('NB_spam_model.pkl','rb')
	# clf = joblib.load(NB_spam_model)

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
        test_lines = CleanTokenize(data)
        test_sequences = tokenizer_obj.texts_to_sequences(test_lines)
        test_review_pad = pad_sequences(test_sequences, maxlen=max_length, padding='post')
		# vect = cv.transform(data).toarray()
		my_prediction = clf.predict(test_review_pad)
        my_prediction*=100
    return render_template('demo.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)



    # def predict_sarcasm(s):
    # x_final = pd.DataFrame({"headline":[s]})
    # test_lines = CleanTokenize(x_final)
    # test_sequences = tokenizer_obj.texts_to_sequences(test_lines)
    # test_review_pad = pad_sequences(test_sequences, maxlen=max_length, padding='post')
    # pred = model.predict(test_review_pad)
    # pred*=100
    # if pred[0][0]>=50: return render_template('demo.html', prediction_text='The text is' $ {"It's a sarcasm!"}) 
    # else: return render_template('demo.html', prediction_text='The text is' $ {"It's not a sarcasm!"})