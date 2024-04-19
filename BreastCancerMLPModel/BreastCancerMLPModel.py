from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

class BreastCancerMLPModel:
    def __init__(self):
        """
        Initialize the BreastCancerMLPModel.

        This method initializes the MLPClassifier model with specified hyperparameters
        for hidden layers, maximum number of iterations, alpha (regularization parameter),
        solver, verbosity level, random state, and tolerance. It also initializes a
        StandardScaler object to scale the input features and a dictionary to map class
        labels to their corresponding diagnosis (Benigno or Maligno).

        Returns:
        -------
        None
            This method does not return any value. It sets up the MLPClassifier model,
            StandardScaler, and class labels dictionary.
        """
        # Initialize MLPClassifier model with specified hyperparameters
        self.model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, alpha=0.0001,
                                solver='adam', verbose=10, random_state=42, tol=0.0001)
        
        # Initialize StandardScaler for feature scaling
        self.scaler = StandardScaler()
        
        # Dictionary to map class labels to their corresponding diagnosis
        self.class_labels = {1: "Benigno", 0: "Maligno"}

    def fit(self):
        """
        Fit the MLP model to the breast cancer dataset.

        This method loads the breast cancer dataset, splits it into training and test sets,
        scales the features using a StandardScaler to ensure they are on the same scale,
        trains the MLP model, and evaluates its performance using both the training and
        test datasets.

        Scaling the features is important because many machine learning algorithms perform
        optimally when features are on similar scales. The StandardScaler calculates the
        mean and standard deviation of each feature in the training set, and then it
        standardizes each feature by subtracting the mean and dividing by the standard
        deviation.

        After scaling the features, the model is trained on the scaled training data. The
        performance of the model is evaluated using both the scaled training and test
        datasets to assess its accuracy on both sets.

        Returns:
        -------
        None
            This method does not return any value. It prints the train and test accuracies
            to the console.
        """
        # Load the dataset
        breast_cancer = load_breast_cancer()
        X = breast_cancer.data
        y = breast_cancer.target

        # Split the dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train the model
        self.model.fit(X_train_scaled, y_train)

        # Evaluate the model
        train_accuracy = accuracy_score(y_train, self.model.predict(X_train_scaled))
        test_accuracy = accuracy_score(y_test, self.model.predict(X_test_scaled))
        
        # Print results
        print("Model trained successfully.")
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        # Calculate and print confusion matrix
        train_conf_matrix = confusion_matrix(y_train, self.model.predict(X_train_scaled))
        test_conf_matrix = confusion_matrix(y_test, self.model.predict(X_test_scaled))
        print("Train Confusion Matrix:\n", train_conf_matrix)
        print("Test Confusion Matrix:\n", test_conf_matrix)

    def predict(self, data):
        """
        Make predictions for the input data.

        This method takes labeled input data, parses it to extract features, scales the
        features using the StandardScaler object, and then uses the trained MLPClassifier
        model to make predictions. It prints the size and values of the feature matrix
        before and after scaling for debugging purposes. It returns the predicted diagnosis
        label (Benigno or Maligno) and the probability of the predicted label.

        Parameters:
        -----------
        data : str
            Labeled input data in the format "mean_radius: ..., mean_texture: ..., etc."

        Returns:
        --------
        tuple
            A tuple containing the predicted diagnosis label (str) and the probability
            (float) of the predicted label.

        Raises:
        -------
        ValueError
            If the number of features in the input data does not match the expected number
            of features.
        """
        # Parse labeled input data and make prediction
        feature_matrix, _ = self._parse_input_data(data)
        
        # Print feature matrix size and values before scaling
        print("Feature matrix size before scaling:", feature_matrix.shape)
        print("Feature matrix before scaling:", feature_matrix)
        
        # Scale features using StandardScaler
        feature_matrix_scaled = self.scaler.transform(feature_matrix)
        
        # Print feature matrix size and values after scaling
        print("Feature matrix size after scaling:", feature_matrix_scaled.shape)
        print("Feature matrix after scaling:", feature_matrix_scaled)
        
        # Make prediction using the trained MLPClassifier model
        prediction = self.model.predict(feature_matrix_scaled)
        
        # Map predicted label to its corresponding diagnosis (Benigno or Maligno)
        predicted_label = self.class_labels[prediction[0]]
        
        # Calculate the probability of the predicted label
        probability = self.model.predict_proba(feature_matrix_scaled)[0][prediction[0]]
        
        return predicted_label, probability

    def _parse_input_data(self, data):
        """
        Private method to parse input data and extract feature values.

        This method takes input data in a formatted string and extracts the feature
        values by splitting the string based on commas and then colon separators.
        It initializes a matrix to store the data and assigns the extracted values
        to the feature matrix. Returns the feature matrix and None.

        Parameters:
        -----------
        data : str
            Labeled input data in the format "mean_radius: ..., mean_texture: ..., etc."

        Returns:
        --------
        tuple
            A tuple containing the feature matrix (numpy array) and None.

        Raises:
        -------
        ValueError
            If the input data format is incorrect or cannot be parsed.
        """
        # Split input data into pairs and extract values
        pairs = data.strip().split(',')
        values = [pair.split(":")[1] for pair in pairs]

        # Initialize a matrix to store the data
        feature_matrix = np.zeros((1, len(values)))

        # Assign the values to the feature matrix
        for j, value in enumerate(values):
            feature_matrix[0, j] = float(value)

        return feature_matrix, None