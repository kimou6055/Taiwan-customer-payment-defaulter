
import gradio as gr
import xgboost as xgb
import numpy as np

# Load the XGBoost model weights
model = xgb.Booster()
model.load_model('model_weights.xgb')

# Define a function to take in the input features and return the predicted target
def predict_defaulter(EDUCATION, SEX, MARRIAGE, LIMIT_BAL, AGE, PAY_1, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6, BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6, PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6):

    # Map the education level to numeric values
    education_level = get_education_level(EDUCATION)

    SEX_level = get_SEX_level(SEX)
    MARRIAGE_level = get_MARIAGE_level(MARRIAGE)

    # Create a numpy array from the input features
    features = np.array([education_level, SEX_level, MARRIAGE_level, LIMIT_BAL, AGE, PAY_1, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6, BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6, PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6])

    # Reshape the array to match the expected shape of the model input
    features = features.reshape(1, -1)

    # Use the loaded model to predict the target
    prediction = model.predict(xgb.DMatrix(features))

    # Return the predicted target
    if prediction[0] == 0:
        return "Not a Defaulter"
    else:
        return "Defaulter"

# Define a function to map education to numeric values
def get_education_level(education):
    if education == "graduate school":
        return 1
    elif education == "university":
        return 2
    elif education == "high school":
        return 3
    elif education == "others":
        return 4
    else:
        return 0

def get_SEX_level(SEX):
    if SEX == "Masculin":
        return 1
    else:
        return 2

def get_MARIAGE_level(MARIAGE):
    if MARIAGE == "marié":
        return 1
    elif MARIAGE == "célibataire":
        return 2
    else:
        return 3



# Define the input and output interfaces for the Gradio app
input_interface = [
    gr.inputs.Radio(choices=['graduate school', 'university', 'high school', 'others'], label='EDUCATION'),
    gr.inputs.Radio(choices=['Masculin', 'Féminin'], label='SEX'),
    gr.inputs.Radio(choices=['marié', 'célibataire','others'], label='MARIAGE'),
    gr.inputs.Slider(0, 10000, label='LIMIT_BAL'),
    gr.inputs.Slider(18, 100, label='AGE'),
gr.inputs.Slider(minimum=-1, maximum=49, step=1, label='PAY_1'),
gr.inputs.Slider(minimum=-1, maximum=49, step=1, label='PAY_2'),
gr.inputs.Slider(minimum=-1, maximum=49, step=1, label='PAY_3'),
gr.inputs.Slider(minimum=-1, maximum=49, step=1, label='PAY_4'),
gr.inputs.Slider(minimum=-1, maximum=49, step=1, label='PAY_5'),
gr.inputs.Slider(minimum=-1, maximum=49, step=1, label='PAY_6'),
    gr.inputs.Slider(0, 10000, label='BILL_AMT1'),
    gr.inputs.Slider(0, 10000, label='BILL_AMT2'),
    gr.inputs.Slider(0, 10000, label='BILL_AMT3'),
    gr.inputs.Slider(0, 10000, label='BILL_AMT4'),
    gr.inputs.Slider(0, 10000, label='BILL_AMT5'),
    gr.inputs.Slider(0, 10000, label='BILL_AMT6'),
    gr.inputs.Slider(0, 10000, label='PAY_AMT1'),
    gr.inputs.Slider(0, 10000, label='PAY_AMT2'),
    gr.inputs.Slider(0, 10000, label='PAY_AMT3'),
    gr.inputs.Slider(0, 10000, label='PAY_AMT4'),
    gr.inputs.Slider(0, 10000, label='PAY_AMT5'),
    gr.inputs.Slider(0, 10000, label='PAY_AMT6')

    ]

output_interface = gr.outputs.Textbox(label='Predicted Defaulter')

gradio_app = gr.Interface(predict_defaulter, inputs=input_interface, outputs=output_interface, title='Credit Card Defaulter Predictor')
gradio_app.launch()
