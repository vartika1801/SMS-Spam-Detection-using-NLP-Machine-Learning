# SMS Spam Detection using NLP and Machine Learning

This project focuses on building an SMS spam classifier using Natural Language Processing (NLP) techniques and multiple machine learning algorithms. The goal is to identify whether a given message is spam or not (ham).

---

## Dataset

The dataset used is the **SMSSpamCollection**, a set of SMS messages tagged as either **ham** (not spam) or **spam**.  
- **Format**: Tab-separated (`.txt`)
- **Columns**:
  - `label`: spam or ham
  - `message`: the text of the SMS message

You can download the dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) or from [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

---

## Project Workflow

### 1. **Data Preprocessing**

- **Text Cleaning**:
  - Removal of special characters and digits using regular expressions
  - Conversion of all text to lowercase

- **Tokenization**:
  - Splitting sentences into words using Python’s `split()` and `nltk.sent_tokenize`

- **Stopword Removal**:
  - Filtering out common words using NLTK's built-in stopword list

- **Stemming**:
  - Applied using `PorterStemmer` from NLTK to reduce words to their root form

- **Lemmatization**:
  - Implemented using `WordNetLemmatizer` from NLTK to extract base word forms

---

### 2. **Feature Extraction Techniques**

We explore three popular techniques to convert text into numerical format:

#### Bag of Words (BoW)
- Created using `CountVectorizer`
- Used bi-grams with `ngram_range=(2, 2)`
- Set `max_features=2500` to limit vocabulary size

#### TF-IDF (Term Frequency-Inverse Document Frequency)
- Created using `TfidfVectorizer`
- Used both uni-grams and bi-grams
- Limited to top 2500 features

#### Word2Vec Embeddings
- Trained a `Word2Vec` model using Gensim on the preprocessed corpus
- Averaged vector representations of words to get a fixed-length feature vector for each sentence

---

### 3. **Model Training and Evaluation**

| Algorithm                | Feature Used | Accuracy |
|--------------------------|--------------|----------|
| Multinomial Naive Bayes  | BoW          | 97.21%   |
| Multinomial Naive Bayes  | TF-IDF       | 98.11%   |
| Random Forest Classifier | TF-IDF       | 98.38%   |
| Random Forest Classifier | Word2Vec     | 84.63%   |

- Models were evaluated using:
  - `accuracy_score`
  - `classification_report` (Precision, Recall, F1-score)
  - `confusion_matrix` 

---

## Libraries Used

- **Pandas** – Data handling
- **NumPy** – Numerical computations
- **Scikit-learn** – ML models and vectorization
- **NLTK** – NLP preprocessing (stopwords, stemming, lemmatization)
- **Gensim** – Word2Vec embeddings
- **TQDM** – Progress visualization

Install dependencies using:

```bash
pip install pandas numpy scikit-learn nltk gensim tqdm
