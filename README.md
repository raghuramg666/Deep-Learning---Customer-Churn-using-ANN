### Deep Learning - Customer Churn using ANN

This repository contains the implementation of an artificial neural network (ANN) designed to predict customer churn based on various input features. The model is built using TensorFlow and Keras, demonstrating the power of neural networks in understanding customer behavior and retention strategies.

#### Prerequisites
To run this project, you will need the following:
- Python 3.8 or later
- pip (Python package installer)

#### Installation
To install the required packages, run the following command:
```
pip install -r requirements.txt
```

#### Tech Stack
- **Programming Language**: Python
- **Libraries and Frameworks**:
  - TensorFlow and Keras for building and training the neural network.
  - Flask for deploying the model as a web service.
  - Jupyter for running notebooks that document experiments and predictions.
  - Pandas and NumPy for data manipulation and numerical computations.
  - Scikit-learn for additional machine learning functionality.
  - Matplotlib and Seaborn for data visualization.
- **Data Storage**: Local CSV files
- **Development Tools**: Jupyter Notebook, Visual Studio Code

#### Project Structure
- `Churn_Modelling.csv`: Dataset used for training the model.
- `app.py`: Flask application for deploying the model as a web service.
- `experiments.ipynb`: Jupyter notebook containing the experiments and analysis.
- `model.h5`: Saved model after training.
- `prediction.ipynb`: Jupyter notebook for loading the model and making predictions.
- `requirements.txt`: List of packages required to run the project.

#### Running the Application
To run the Flask application, execute:
```
python app.py
```
This will start the server on `localhost` and can be accessed via a web browser at `http://127.0.0.1:5000`.

#### Model Training
To train the model from scratch, you can follow the steps provided in `experiments.ipynb`. This notebook includes detailed instructions on data preprocessing, model building, training, and evaluation.

#### Making Predictions
Use the `prediction.ipynb` notebook to load the pre-trained model and make predictions using new data.
