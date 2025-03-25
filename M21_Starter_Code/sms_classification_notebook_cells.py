# Cell 1: Import libraries
# Import pandas
import pandas as pd
# Import the required dependencies from sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

# Set the column width to view the text message data.
pd.set_option('max_colwidth', 200)

# Import Gradio
import gradio as gr

# Cell 2: Define the SMS classification function
def sms_classification(sms_text_df):
    """
    Perform SMS classification using a pipeline with TF-IDF vectorization and Linear Support Vector Classification.

    Parameters:
    - sms_text_df (pd.DataFrame): DataFrame containing 'text_message' and 'label' columns for SMS classification.

    Returns:
    - text_clf (Pipeline): Fitted pipeline model for SMS classification.

    This function takes a DataFrame with 'text_message' and 'label' columns, splits the data into
    training and testing sets, builds a pipeline with TF-IDF vectorization and Linear Support Vector
    Classification, and fits the model to the training data. 
    The fitted pipeline is returned to make future predictions.
    """
    # Set the features variable to the text message column.
    X = sms_text_df['text_message']
    
    # Set the target variable to the "label" column.
    y = sms_text_df['label']

    # Split data into training and testing and set the test_size = 33%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Build a pipeline to transform the test set to compare to the training set.
    text_clf = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('classifier', LinearSVC())
    ])

    # Fit the model to the transformed training data and return model.
    text_clf.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = text_clf.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print(f"Model accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return text_clf

# Cell 3: Load the dataset
# Load the dataset into a DataFrame
sms_df = pd.read_csv("Resources/SMSSpamCollection.csv")
sms_df.head()

# Cell 4: Train the model
# Call the sms_classification function with the DataFrame and set the result to the "text_clf" variable
text_clf = sms_classification(sms_df)

# Cell 5: Define the prediction function
# Create a function called `sms_prediction` that takes in the SMS text and predicts the whether the text is "not spam" or "spam". 
# The function should return the SMS message, and say whether the text is "not spam" or "spam".
def sms_prediction(text):
    """
    Predict the spam/ham classification of a given text message using a pre-trained model.

    Parameters:
    - text (str): The text message to be classified.

    Returns:
    - str: A message indicating whether the text message is classified as spam or not.

    This function takes a text message and a pre-trained pipeline model, then predicts the
    spam/ham classification of the text. The result is a message stating whether the text is
    classified as spam or not.
    """
    # Create a variable that will hold the prediction of a new text.
    prediction = text_clf.predict([text])[0]
    
    # Using a conditional if the prediction is "ham" return the message:
    # f'The text message: "{text}", is not spam.' Else, return f'The text message: "{text}", is spam.'
    if prediction == "ham":
        return f'The text message: "{text}", is not spam.'
    else:
        return f'The text message: "{text}", is spam.'

# Cell 6: Create and launch the Gradio interface
# Create a Gradio interface for the SMS spam classifier
iface = gr.Interface(
    fn=sms_prediction,
    inputs=gr.Textbox(lines=2, placeholder="Enter an SMS message here..."),
    outputs="text",
    title="SMS Spam Classifier",
    description="Enter an SMS message to check if it's spam or not."
)

# Launch the interface
iface.launch()
