# IMDb Sentiment Classifier

A Natural Language Processing project that classifies IMDb movie reviews as **positive** or **negative** sentiment.

This project compares a simple baseline with two classical machine-learning models and includes exploratory data analysis, evaluation, error analysis, and interpretation of the learned text features.

## Project Structure

```text
IMDb-Sentiment-Classifier/
├── IMDb_Sentiment_Classifier.ipynb
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

## Objective

Build a binary text-classification model that predicts the sentiment of a movie review:

- `0` — Negative
- `1` — Positive

The goal is not only to achieve a strong score, but also to compare models fairly and examine where the best model makes mistakes.

## Dataset

The project uses the [IMDb movie review dataset](https://keras.io/api/datasets/imdb/) provided by Keras.

- 25,000 training reviews
- 25,000 test reviews
- Binary sentiment labels: positive or negative
- Vocabulary limited to the 10,000 most frequent words

The dataset is initially stored as sequences of word indexes. The notebook decodes these sequences into readable text before analysis and modelling.
## Pipeline

| Step | Description |
| --- | --- |
| Data Loading | Load the IMDb dataset through `tensorflow.keras.datasets.imdb` |
| Text Decoding | Convert encoded word sequences into readable reviews |
| EDA | Analyse class balance and review-length distribution |
| Validation Split | Create a stratified validation set from the training data |
| Baseline | Use a majority-class classifier as a minimum reference |
| Feature Extraction | Convert reviews into TF-IDF features with unigrams and bigrams |
| Models | Compare Multinomial Naive Bayes and Logistic Regression |
| Evaluation | Measure accuracy, precision, recall, F1-score, and confusion matrix |
| Error Analysis | Inspect incorrectly classified reviews |
| Interpretation | Identify words and phrases most associated with each sentiment |
| Final Evaluation | Train the selected model on all training data and evaluate once on the official test set |

## Models

### Dummy Baseline

A classifier that always predicts the most frequent class. This gives a minimum performance level that the real models must beat.

### Multinomial Naive Bayes

A fast probabilistic model commonly used as a baseline for text classification.

### Logistic Regression

A linear classification model trained on TF-IDF text features. It can estimate probabilities and makes the learned word weights inspectable.
## Results

All models were evaluated on the same stratified validation split.

| Model | Validation Accuracy | Validation F1-score |
| --- | ---: | ---: |
| Dummy Baseline | 50.00% | 0.00% |
| Multinomial Naive Bayes | 87.94% | 87.89% |
| Logistic Regression | 88.76% | 88.97% |

Logistic Regression achieved the best validation performance and was selected as the final model.

### Final Test Performance

The selected model was retrained on the complete 25,000-review training set and evaluated once on the untouched official test set.

| Metric | Score |
| --- | ---: |
| Accuracy | 89.49% |
| Precision | 88.89% |
| Recall | 90.26% |
| F1-score | 89.57% |

## Error Analysis

The model performs well on direct positive and negative opinions, but some cases remain difficult:

- Reviews with mixed opinions, such as positive acting but a weak plot
- Sarcasm and irony
- Reviews where sentiment changes near the end
- Very short reviews with little context
- Rare words replaced by the out-of-vocabulary token

The notebook displays misclassified reviews to examine these failures instead of relying only on aggregate metrics.

## Model Interpretation

For Logistic Regression, the learned coefficients show which words or phrases are most associated with each class.

This does **not** mean those words cause a sentiment. It only shows which textual patterns the model relied on in this specific dataset.
## Quick Start

```bash
git clone https://github.com/Leonardo-M-AI/IMDb-Sentiment-Classifier.git
cd IMDb-Sentiment-Classifier
pip install -r requirements.txt
jupyter notebook IMDb_Sentiment_Classifier.ipynb
```

## Requirements

```text
tensorflow
scikit-learn
pandas
numpy
matplotlib
seaborn
jupyter
```

## Tech Stack

`Python` · `TensorFlow/Keras` · `scikit-learn` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn`
## Limitations

- The dataset contains English-language movie reviews only.
- The vocabulary is limited to the most frequent 10,000 words.
- The model can misinterpret irony, sarcasm, and nuanced opinions.
- Good performance on IMDb reviews does not guarantee good performance on product reviews, social-media posts, or other domains.
- This is a sentiment-classification project, not a system for judging the objective quality of a film.

## What I Would Do Next

- Compare the classical TF-IDF approach with an LSTM or transformer-based model.
- Test the model on reviews outside the IMDb dataset to measure generalisation.
- Add a confidence threshold so uncertain predictions are labelled as `uncertain`.
- Build a small Streamlit interface where users can enter a review and view the prediction.
- Experiment with hyperparameter tuning for TF-IDF and Logistic Regression.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
