import gradio as gr
import xgboost as xgb
import numpy as np

# Load the XGBoost model weights
model = xgb.Booster()
model.load_model('model_weights.xgb')

# Define a function to take in the input features and return the predicted target
def predict_defaulter(EDUCATION_1, EDUCATION_2, EDUCATION_3, EDUCATION_4, SEX_1, SEX_2, MARRIAGE_1, MARRIAGE_2, MARRIAGE_3, LIMIT_BAL, AGE, PAY_1, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6, BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6, PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6):
    # Create a numpy array from the input features
    features = np.array([EDUCATION_1, EDUCATION_2, EDUCATION_3, EDUCATION_4, SEX_1, SEX_2, MARRIAGE_1, MARRIAGE_2, MARRIAGE_3, LIMIT_BAL, AGE, PAY_1, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6, BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6, PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6])
    
    # Reshape the array to match the expected shape of the model input
    features = features.reshape(1, -1)
    
    # Use the loaded model to predict the target
    prediction = model.predict(xgb.DMatrix(features))
    
    # Return the predicted target
    if prediction[0] == 0:
        return "Not a Defaulter"
    else:
        return "Defaulter"

# Define the input and output interfaces for the Gradio app
input_interface = [
    gr.inputs.Slider(0, 1, label='EDUCATION_1'),
    gr.inputs.Slider(0, 1, label='EDUCATION_2'),
    gr.inputs.Slider(0, 1, label='EDUCATION_3'),
    gr.inputs.Slider(0, 1, label='EDUCATION_4'),
    gr.inputs.Slider(0, 1, label='SEX_1'),
    gr.inputs.Slider(0, 1, label='SEX_2'),
    gr.inputs.Slider(0, 1, label='MARRIAGE_1'),
    gr.inputs.Slider(0, 1, label='MARRIAGE_2'),
    gr.inputs.Slider(0, 1, label='MARRIAGE_3'),
    gr.inputs.Slider(0, 1000000, label='LIMIT_BAL'),
    gr.inputs.Slider(18, 100, label='AGE'),
    gr.inputs.Slider(-2, 8, label='PAY_1'),
    gr.inputs.Slider(-2, 8, label='PAY_2'),
    gr.inputs.Slider(-2, 8, label='PAY_3'),
    gr.inputs.Slider(-2, 8, label='PAY_4'),
    gr.inputs.Slider(-2, 8, label='PAY_5'),
    gr.inputs.Slider(-2, 8, label='PAY_6'),
    gr.inputs.Slider(-100000, 1000000, label='BILL_AMT1'),
    gr.inputs.Slider(-100000, 1000000, label='BILL_AMT2'),
    gr.inputs.Slider(-100000, 1000000, label='BILL_AMT3'),
    gr.inputs.Slider(-100000, 1000000, label='BILL_AMT4'),
    gr.inputs.Slider(-100000, 1000000, label='BILL_AMT5'),
    gr.inputs.Slider(-100000, 1000000, label='BILL_AMT6'),
    gr.inputs.Slider(0, 1000000, label='PAY_AMT1'),
    gr.inputs.Slider(0, 1000000, label='PAY_AMT2'),
    gr.inputs.Slider(0, 1000000, label='PAY_AMT3'),
    gr.inputs.Slider(0, 1000000, label='PAY_AMT4'),
    gr.inputs.Slider(0, 1000000, label='PAY_AMT5'),
    gr.inputs.Slider(0, 1000000, label='PAY_AMT6')

    ]

output_interface = gr.outputs.Textbox(label='Predicted Defaulter')

gradio_app = gr.Interface(predict_defaulter, inputs=input_interface, outputs=output_interface, title='Credit Card Defaulter Predictor')
gradio_app.launch()
