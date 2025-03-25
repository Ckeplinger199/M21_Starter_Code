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
pd.set_option('display.width', 1000)  # Wider display for better readability

def sms_classification(sms_text_df):
    """
    Perform SMS classification using a pipeline with TF-IDF vectorization and Linear Support Vector Classification.

    Parameters:
    - sms_text_df (pd.DataFrame): DataFrame containing 'text_message' and 'label' columns for SMS classification.

    Returns:
    - text_clf (Pipeline): Fitted pipeline model for SMS classification.
    - X_test: Test features
    - y_test: Test labels
    """
    # Set the features variable to the text message column.
    X = sms_text_df['text_message']
    
    # Set the target variable to the "label" column.
    y = sms_text_df['label']

    # Split data into training and testing and set the test_size = 33%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    print(f"Training set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")

    # Build a pipeline to transform the test set to compare to the training set.
    text_clf = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('classifier', LinearSVC())
    ])

    # Fit the model to the transformed training data and return model.
    print("Fitting the model...")
    text_clf.fit(X_train, y_train)
    print("Model training complete!")
    
    return text_clf, X_test, y_test

# Load the dataset into a DataFrame
def load_data():
    print("Reading CSV file...")
    sms_df = pd.read_csv("Resources/SMSSpamCollection.csv")
    print(f"Dataset shape: {sms_df.shape}")
    print(f"Class distribution:\n{sms_df['label'].value_counts()}")
    return sms_df

# Create a function called `sms_prediction` that takes in the SMS text and predicts whether the text is "not spam" or "spam". 
def sms_prediction(text, model):
    """
    Predict the spam/ham classification of a given text message using a pre-trained model.

    Parameters:
    - text (str): The text message to be classified.
    - model: The trained classification model.

    Returns:
    - str: A message indicating whether the text message is classified as spam or not.
    """
    # Create a variable that will hold the prediction of a new text.
    prediction = model.predict([text])[0]
    
    # Using a conditional if the prediction is "ham" return the message:
    if prediction == "ham":
        return f'The text message: "{text}", is not spam.'
    else:
        return f'The text message: "{text}", is spam.'

# Main execution
if __name__ == "__main__":
    print("=" * 80)
    print("SMS SPAM CLASSIFICATION MODEL")
    print("=" * 80)
    
    print("\nStep 1: Loading SMS dataset...")
    # Load the dataset
    sms_df = load_data()
    
    # Display the first few rows
    print("\nFirst few rows of the dataset:")
    print(sms_df.head())
    
    print("\nStep 2: Training the model...")
    # Train the model
    text_clf, X_test, y_test = sms_classification(sms_df)
    
    print("\nStep 3: Evaluating the model...")
    # Evaluate the model
    y_pred = text_clf.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print(f"Model accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Test with some examples
    print("\nStep 4: Testing with example messages:")
    test_messages = [
        "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18",
        "I'll be home by 6, can we have dinner together?",
        "FREE ENTRY into our £250 weekly comp just send the word WIN to 80086 NOW. 18 T&C www.txttowin.co.uk",
        "Hey, what's up? Want to meet for coffee later today?"
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\nExample {i}:")
        print(f"Message: {message}")
        result = sms_prediction(message, text_clf)
        print(f"Prediction: {result}")
    
    print("\n" + "=" * 80)
    print("MODEL TRAINING COMPLETE")
    print("=" * 80)
