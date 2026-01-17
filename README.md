# Email/SMS Spam Classifier

A machine learning project that classifies emails and SMS messages as spam or legitimate using natural language processing and the Naive Bayes algorithm.

## 📋 Project Overview

This project builds an intelligent spam detection system that uses TF-IDF vectorization combined with a Bernoulli Naive Bayes classifier to accurately identify spam messages. The model is trained on a labeled dataset and deployed as an interactive web application using Streamlit.

## 🎯 Features

- **Text Preprocessing**: Tokenization, stopword removal, stemming, and punctuation handling
- **Feature Extraction**: TF-IDF vectorization with n-grams (1-2) up to 3000 features
- **Multiple ML Models**: Trained and compared GaussianNB, MultinomialNB, and BernoulliNB classifiers
- **Interactive Web Interface**: User-friendly Streamlit application for real-time predictions
- **Exploratory Data Analysis**: Statistical analysis and visualization of spam vs. ham messages

## 📁 Project Structure

```
email-spam-finder/
├── app.py                  # Streamlit web application
├── spam-classifier.ipynb   # Jupyter notebook with model development and training
├── spam.csv               # Dataset containing labeled spam/ham messages
├── vectorizer.pkl         # Serialized TF-IDF vectorizer
├── model.pkl              # Serialized trained Bernoulli Naive Bayes model
└── README.md              # Project documentation
```

## 🛠️ Technologies Used

- **Python 3.x**
- **Libraries**:
  - `pandas` - Data manipulation and analysis
  - `numpy` - Numerical computing
  - `scikit-learn` - Machine learning algorithms
  - `nltk` - Natural language processing
  - `streamlit` - Web application framework
  - `matplotlib` & `seaborn` - Data visualization
  - `wordcloud` - Text visualization

## 📊 Dataset

The project uses the `spam.csv` dataset containing:

- **Target**: Binary classification (Ham: 0, Spam: 1)
- **Text**: Message content to be classified
- Features include character count, word count, and sentence count

## 🔧 Installation & Setup

1. **Clone or navigate to the project directory**:

   ```bash
   cd email-spam-finder
   ```

2. **Install required dependencies**:

   ```bash
   pip install pandas numpy scikit-learn nltk streamlit matplotlib seaborn wordcloud
   ```

3. **Download NLTK resources** (automatically done by the app on first run):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

## 🚀 Usage

### Training the Model

Run the Jupyter notebook `spam-classifier.ipynb` to:

- Load and explore the dataset
- Perform data cleaning and preprocessing
- Conduct exploratory data analysis
- Train and evaluate multiple classifiers
- Save the vectorizer and best model (`vectorizer.pkl` and `model.pkl`)

### Running the Web Application

Launch the Streamlit app:

```bash
streamlit run app.py
```

Then:

1. Enter an email or SMS message in the text area
2. Click the "Predict" button
3. The app will display whether the message is **Spam** or **Not Spam**

## 🧠 Model Details

### Text Preprocessing Pipeline

1. **Lowercase conversion**: Standardize text case
2. **Tokenization**: Split text into individual words
3. **Alphanumeric filtering**: Remove special characters
4. **Stopword removal**: Eliminate common English words
5. **Stemming**: Reduce words to their root form using Porter Stemmer

### Feature Extraction

- **TF-IDF Vectorizer** with:
  - Maximum of 3000 features
  - Unigrams and bigrams (1-2 grams)

### Classification Model

- **Bernoulli Naive Bayes** - Best performing classifier among tested models
- Efficiently handles sparse binary features from TF-IDF vectorization

## 📈 Model Performance

The notebook evaluates three Naive Bayes variants:

- **GaussianNB**: Assumes continuous feature distributions
- **MultinomialNB**: Designed for discrete count data
- **BernoulliNB**: Optimized for binary feature vectors ✓ _Selected_

Metrics tracked:

- Accuracy Score
- Confusion Matrix
- Precision Score

## 📝 Data Processing

- **Data Cleaning**: Removes unnecessary columns and duplicates
- **Label Encoding**: Converts 'ham'/'spam' to 0/1
- **Missing Values**: Checked and handled appropriately
- **Duplicate Removal**: Eliminates duplicate messages from dataset

## 🎨 Visualizations Included

- Pie chart showing ham vs. spam distribution
- Word cloud visualization of spam messages
- Bar plot of most common spam keywords
- Statistical summaries of text features

**Happy Spam Filtering!** 🚀
