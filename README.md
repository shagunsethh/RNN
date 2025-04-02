# Sentiment Analysis Using RNN

# Group 27
#### Shagun Seth(055042)
#### Sweta Behera(055051)

## 📌 Problem Statement
Customer reviews on e-commerce platforms like Amazon contain valuable insights about user experiences. However, manually analyzing thousands of reviews is impractical. This project automates sentiment analysis using *recurrent neural networks (RNNs)* to classify reviews as positive or negative.

## 📖 Abstract
This study implements a *LSTM-based RNN* for sentiment analysis. It preprocesses Amazon reviews by *removing stopwords, tokenizing text, and applying word embeddings (GloVe)*. The model is trained on a balanced dataset and evaluated using multiple performance metrics.

*Final Model Accuracy: 96.37%* 🎯

---

## 📊 Dataset Description
•⁠  ⁠*Source:* Amazon Reviews Dataset  
•⁠  ⁠*Features:*  
  - ⁠ Review Text ⁠: Customer feedback  
  - ⁠ Sentiment Label ⁠: 0 = Negative, 1 = Positive  
•⁠  ⁠*Preprocessing Applied:*  
  - Removed stopwords while retaining key sentiment words (e.g., "not")  
  - Converted text to numerical sequences using *GloVe embeddings*  

---

## 🚀 Steps Performed

### 🔹 Data Preprocessing
✔ Cleaned and tokenized text  
✔ Applied padding for uniform input lengths  

### 🔹 Exploratory Data Analysis (EDA)
✔ *Visualized sentiment distribution*  
✔ Created *word clouds* for frequent words in positive & negative reviews  

### 🔹 Handling Class Imbalance
✔ Applied *SMOTE* to oversample minority class  

### 🔹 Feature Engineering
✔ Used *GloVe word embeddings* instead of TF-IDF  

### 🔹 Model Training & Evaluation
✔ Implemented a *Bidirectional LSTM-based RNN*  
✔ Used *dropout and batch normalization*  
✔ *Fine-tuned hyperparameters* for best results  
✔ Evaluated performance using *accuracy, confusion matrix, and F1-score*  

---

## 📈 Model Performance

### ✅ *Final Accuracy: 96.37%*
•⁠  ⁠*Training Loss:* 0.1198  
•⁠  ⁠*Validation Loss:* 0.1765  

### ✅ *Confusion Matrix*
| Sentiment | Precision | Recall | F1-score | Support |
|-----------|-----------|---------|-----------|---------|
| Negative  | 0.94      | 0.99    | 0.96      | 579     |
| Positive  | 0.99      | 0.93    | 0.96      | 579     |
| *Overall* | *0.97* | *0.96* | *0.96* | *1158* |

📊 *Misclassification Analysis:*  
•⁠  ⁠Some reviews had *mixed sentiment*, leading to incorrect predictions.  
•⁠  ⁠Short reviews were harder to classify due to lack of context.  

---

## 🔍 Key Insights
✔ *GloVe embeddings enhanced performance* over TF-IDF  
✔ *Class balancing (SMOTE) improved model fairness*  
✔ *Bidirectional LSTM captured long-range dependencies effectively*  
✔ *Dropout layers reduced overfitting*  

---

## 🎯 Managerial Implications
📌 *Customer Experience Insights:* Identify areas for product improvement  
📌 *Brand Reputation Management:* Real-time monitoring of customer sentiment  
📌 *Competitive Intelligence:* Compare sentiment trends across competitors  
📌 *Automated Review Moderation:* Filter spam or inappropriate content  

---

## 🔮 Future Enhancements
🚀 Implement *BERT/GPT models* for better context understanding  
🚀 Expand to *multi-language sentiment analysis*  
🚀 Introduce *sentiment scoring* instead of binary classification  
🚀 Deploy as an *API for real-time sentiment tracking*  

---

## 🛠 Technical Stack
•⁠  ⁠*Deep Learning:* TensorFlow, Keras, PyTorch  
•⁠  ⁠*NLP Libraries:* NLTK, Spacy, Gensim  
•⁠  ⁠*Visualization:* Matplotlib, Seaborn  
•⁠  ⁠*Data Processing:* Pandas, NumPy, Scikit-learn  

---

## 📚 References
•⁠  ⁠Amazon Reviews Dataset  
•⁠  ⁠Research Papers on LSTM & Sentiment Analysis  

---

## ⭐ Final Thoughts
This project demonstrates the power of *deep learning for sentiment analysis. With **96.37% accuracy, it provides a robust tool for extracting insights from customer feedback. Future improvements, such as **BERT models and real-time sentiment tracking*, will further enhance its applications in business intelligence.
