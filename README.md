# Natural Language Processing Challenge (Ironhack)

**Project Description:** A Natural Language Processing (NLP) project aimed at classifying news headlines as either real or fake news using text vectorization and machine learning classifiers.

## Dataset Description
The dataset was provided by Ironhack for this specific challenge.
* **Files:** `training_data.csv`
* **Dimensions:** 34,152 rows and 2 columns.
* **Target Variable:** 0 (Fake News) and 1 (Real News).
* **Class Balance:** The data is relatively balanced with 17,572 fake news records and 16,580 real news records.
* **Data Quirks:** The dataset contains no missing values, though 1,946 duplicated rows were identified during exploration.

## Research Goal
The primary objective of this project was to build a classifier capable of distinguishing between real and fake news headlines. A secondary goal was to practice and compare different text vectorization techniques (TF-IDF, Bag of Words, and Embeddings) across various classification models. *Note: My specific focus was on TF-IDF vectorization with various preprocessing methods, while my collaborative partner explored Bag of Words and Embeddings.*

## Steps Taken
* **Data Exploration:** Loaded, explored, and split the data into training and testing sets.
* **Text Preprocessing:** Cleaned the text, applied tokenization, removed stop words, performed Part-of-Speech (POS) tagging, and utilized lemmatization.
* **Vectorization:** Applied TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical features.
* **Modeling:** Trained and evaluated multiple classifiers, including Logistic Regression, XGBoost, Decision Tree, Random Forest, and Support Vector Classifier (SVC).
* **Tuning:** Performed hyperparameter tuning specifically on the TF-IDF and SVC combination, comparing results between raw text and preprocessed text.

## Main Findings
* **Evaluation Metrics:** Accuracy on test data, overfitting checks, and ROC-AUC scores were the most helpful indicators for deciding the best-performing models.
* **Top Performers:** Logistic Regression and Support Vector Classifier (SVC) yielded the best results for this specific classification task.
* **Preprocessing Impact:** Interestingly, the combination of TF-IDF vectorization and SVC performed better when using the raw text data rather than the heavily preprocessed text.

## How to Reproduce the Project
* **Prerequisites:** Python 3, Jupyter Notebook, and standard data science libraries (`pandas`, `scikit-learn`, `nltk` or `spacy`).
* Clone this repository to your local machine.
* Ensure the datasets are placed inside the `Data` folder.
* Run the Jupyter Notebook located in the `Notebooks` folder to see the step-by-step analysis and model training.

## Next Steps / Ideas for Improvement
* Tune the Logistic Regression model further and benchmark it extensively against the other models.
* Experiment with different word embedding techniques (like Word2Vec or GloVe) to see if semantic relationships improve accuracy.
* Explore Deep Learning approaches, such as building a Convolutional Neural Network (CNN) or an RNN, instead of relying solely on traditional classifiers.

## Repo Structure
| Folder/File | Description |
| :--- | :--- |
| **`Notebooks/`** | Contains the main Jupyter Notebook with the EDA, preprocessing, and modeling, alongside a `utils.py` file used to generate model metrics. |
| **`Data/`** | Stores the dataset files: `training_data.csv`, `testing_data.csv`, and the final output `testing_predicted_data.csv`. |
