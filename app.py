import pandas as pd
from flask import Flask, jsonify, request
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


# load model
model = pickle.load(open('fakemodel.pkl', 'rb'))

# app
app = Flask(__name__)

# routes
port_stem = PorterStemmer()
nltk.download('stopwords')
english_sw = stopwords.words('english')


def stemming(content):
    # accept only a-z charaters
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()  # convert them to lowercase
    stemmed_content = stemmed_content.split()  # convert to list
    stemmed_content = [port_stem.stem(
        word) for word in stemmed_content if not word in english_sw]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


@app.route('/', methods=['POST'])
def predict():
    # get data
    data = request.get_json(force=True)

    # convert data into dataframe
    data_df = pd.DataFrame(data,  index=[0])
    data_df['content'] = data_df['content'].apply(stemming)

    X = data_df['content'].values

    vectorizer = TfidfVectorizer()
    vectorizer.fit(X)
    X = vectorizer.transform(X)

    # predictions
    result = model.predict(X)

    # send back to browser
    output = {'results': int(result[0])}

    # return data
    return jsonify(results=output)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
