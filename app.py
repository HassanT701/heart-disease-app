import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load the dataset
DATA_URL = 'https://www.kaggle.com/datasets/shivarajmishra/heart-data?resource=download'
data = pd.read_csv('heartDisease.csv')

# Display first 5 rows of the data
st.title('Heart Disease Prediction')
st.text('By: Hassan Talha')
st.header('A small sample of the dataset used:')
st.write(data.head())

# Link to the dataset
st.write(f"[Link to full dataset]({DATA_URL})   * NOTE: Target value 1 has been changed to 0; Target value 2 has been changed to 1 *")
st.write("")
st.write("Age: Age in years")
st.write("Cp: Chest pain type (0 = typical angina; 1 = atypical angina; 2 = non-anginal pain; 3: asymptomatic)")
st.write("Trestbps: Resting blood pressure in mm Hg")
st.write("Chol: Serum cholesterol in mg/dl")
st.write("fbs: Fasting blood sugar > 120 mg/dl (1=true; 0=false)")
st.write("Restecg: Resting electrocardiographic results (0 = normal; 1 = having ST-T wave abnormality; 2 = probable or definite left ventricular hypertrophy)")
st.write("Thalach: Maximum heart rate achieved")
st.write("Exang: Exercise-induced angina (Chest pain) (1 = yes; 0 = no)")
st.write("Oldpeak: ST depression during exersize")
st.write("Slope: Slope of ST depression during exercise (0 = upsloping; 1 = flat; 2 = downsloping)")
st.write("Ca: Number of major vessels (0 - 3) colored by fluoroscopy (fluid used by doctors to examine major arteries)")
st.write("Thal: Thalassemia (Blood disorder type if applicaple) (3 = normal; 6 = fixed defect; 7 = reversable defect)")
st.write("Target: Heart disease (0 = no, 1 = yes)")

# Ensure target values are binary (0 and 1)
data['target'] = np.where(data['target'] > 1, 1, data['target'])

# Define features and target
X = data.drop('target', axis=1)
y = data['target']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build and compile the model with the best hyperparameters
def build_model():
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    optimizer = Adam(learning_rate=0.01)  # Best learning rate
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_model()

# Early stopping and learning rate reduction callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

# Train the model with the best batch size
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, callbacks=[early_stopping, reduce_lr], verbose = 0)

# Create sliders for user input
st.sidebar.header('Enter your information here:')
user_input = {}
for column in X.columns:
    if column == 'age':
        user_input[column] = st.sidebar.number_input('Age', min_value=1, max_value=120, value=int(X[column].mean()))
    elif column == 'sex':
        user_input[column] = st.sidebar.selectbox('Sex', ('Male', 'Female'))
        user_input[column] = 1 if user_input[column] == 'Male' else 0
    elif column == 'cp':
        user_input[column] = st.sidebar.selectbox('Chest Pain Type', ('Typical Angina','Atypical Angina','Non-Anginal Pain','No Pain'))
        user_input[column] = 1 if user_input[column] == 'Typical Angina' else 2 if user_input[column] == 'Atypical Angina' else 3 if user_input[column] == 'Non-Anginal Pain' else 4
    elif column == 'trestbps':
        user_input[column] = st.sidebar.number_input('Resting Blood Pressure in mm Hg (75-200)', min_value=75, max_value=200, value=int(X[column].mean()))
    elif column == 'chol':
        user_input[column] = st.sidebar.number_input('Serum Cholesterol in mg/dl (100-600)', min_value=100, max_value=600, value=int(X[column].mean()))
    elif column == 'fbs':
        user_input[column] = st.sidebar.selectbox('Fasting Blood Sugar', ('Above 120','Below 120'))
        user_input[column] = 1 if user_input[column] == 'Above 120' else 0
    elif column == 'restecg':
        user_input[column] = st.sidebar.selectbox('Resting ECG results', ('Normal','ST-T Wave Abnormality','Probable or definite Ventricular Hypertrophy'))
        user_input[column] = 0 if user_input[column] == 'Normal' else 1 if user_input[column] == 'ST-T Wave Abnormality' else 2
    elif column == 'thalach':
        user_input[column] = st.sidebar.number_input('Maximum Heart Rate (50-300)', min_value=50, max_value=300, value=int(X[column].mean()))
    elif column == 'exang':
        user_input[column] = st.sidebar.selectbox('Excersize Induced Chest Pain', ('Yes', 'No'))
        user_input[column] = 1 if user_input[column] == 'Yes' else 0
    elif column == 'oldpeak':
        user_input[column] = st.sidebar.number_input('ST depression during exersize (0.0-10.0)', min_value=0.0, max_value=10.0, value=float(X[column].mean()))
    elif column == 'slope':
         user_input[column] = st.sidebar.selectbox('Slope of ST depression during exercise', ('Upsloping','Flat','Downsloping'))
         user_input[column] = 0 if user_input[column] == 'Upsloping' else 1 if user_input[column] == 'Flat' else 2
    elif column == 'ca':
         user_input[column] = st.sidebar.selectbox('Number of major vessels colored by Fluorosopy', ('0','1','2','3'))
         user_input[column] = 0 if user_input[column] == '0' else 1 if user_input[column] == '1' else 2 if user_input[column] == '2' else 3
    elif column == 'thal':
         user_input[column] = st.sidebar.selectbox('Thalassemia', ('Normal','Fixed Defect','Reversable Defect'))
         user_input[column] = 3 if user_input[column] == 'Normal' else 6 if user_input[column] == 'Fixed Defect' else 7

user_input_df = pd.DataFrame(user_input, index=[0])
user_input_scaled = scaler.transform(user_input_df)

if st.sidebar.button('Predict'):
    prediction = model.predict(user_input_scaled)
    prediction = (prediction > 0.5).astype(int)
    st.header(f' Prediction: {":red[Heart Disease is very likely]" if prediction[0][0] == 1 else ":green[Heart Disease is unlikely but it is still a good idea to get screened by a doctor]"}')
   
    # Add user's input to the data for graphing
    user_data = user_input_df.copy()
    user_data['target'] = 0 if prediction[0][0] == 0 else 1

    # Plot the existing dataset
    st.write('Your input compared with the dataset:')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=data, x='age', y='chol', hue='target', palette={0: 'blue', 1: 'red'}, ax=ax)
    
    # Overlay the user's choice on the plot
    plt.scatter(user_data['age'], user_data['chol'], color='black', s=100, edgecolor='k')
    
    st.pyplot(fig)
    # Inject JavaScript code to scroll to the bottom
    scroll_script = """
    <script>
    window.scrollTo(0, document.body.scrollHeight);
    </script>
    """
    st.components.v1.html(scroll_script, height=0)
