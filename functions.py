import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import WeibullFitter, LogNormalFitter, LogLogisticFitter, ExponentialFitter


def fit_models_summary(models, survival_data):

    ''' Function for initiating models from lifelines, 
           fitting to the data and printing respective summaries'''
    
    for name, model in models.items():
        model.fit(durations = survival_data["tenure"], event_observed = survival_data["churn"])
        print(f'{name} Model Summary:')
        print(model.summary)
        print(' ')


def visualize_survival_curves(models):
    
    ''' Function for plotting survival curves for the models from lifelines'''

    plt.figure(figsize = (20, 10))
    colors = ['navy', 'indigo', 'maroon', 'sienna']

    for (name, model), color in zip(models.items(), colors):
        if hasattr(model, "plot_survival_function"):
            model.plot_survival_function(color = color, label = name)

    plt.title('Survival Curves for Parametric Models')
    plt.xlabel('Tenure')
    plt.ylabel('Survival Probability')

    plt.legend()
    plt.show()


def compare_models_aic(models):

    '''Function for compairing AIC scores of the models from lifelines and
        choosing the model with the lowest score.'''

    aic_values = {name: model.AIC_ for name, model in models.items()}
    best_model = min(aic_values, key=aic_values.get)
    print(f'Best Model based on AIC: {best_model}')


def calculate_and_add_clv(model, survival_data):

    '''Function for calculating CLV per customer by predicting survival function 
        and adding a new CLV column to the data'''

    # Predict the survival function for each customer
    pred_survival_function = model.predict_survival_function(survival_data)

    # Calculate CLV for each customer
    clv_per_customer = pred_survival_function.apply(lambda row: row.sum())

    # Add CLV as a new column in the dataset
    survival_data['CLV'] = clv_per_customer


def plot_clv_distribution_by_segments(survival_data, segments):

    ''' Function for plotting distribution of CLV scores across different segments'''

    for segment in segments:
        segment_values = survival_data[segment].unique()

        plt.figure(figsize=(20, 5))

        for value in segment_values:
            subset = survival_data[survival_data[segment] == value]
            plt.hist(subset['CLV'], bins=30, alpha=0.5, label=f'{segment}={value}')

        plt.title(f'CLV Distribution within Different Segments')
        plt.xlabel('Customer Lifetime Value (CLV)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()


def calculate_annual_retention_budget(model, survival_data, retention_scale="medium"):
    
    '''Function for calculating annual retention budget based on the relative cost per customer and the identified number of at-risk customers'''

    pred_survival_function = model.predict_survival_function(survival_data)

    # Assign relative cost based on scale
    scale_to_cost = {"small": 50, "medium": 125, "large": 200}
    cost_per_retention = scale_to_cost.get(retention_scale, 100)  # Default to 100 if scale is not recognized

    # Identify the number of subscribers at risk within a year
    at_risk_indices = pred_survival_function.index[pred_survival_function.iloc[:, 0] > 0.5] # Considering 50% as a threshold
    subscribers_at_risk = survival_data.loc[at_risk_indices]

    # Calculate the Annual Retention Budget
    annual_retention_budget = len(subscribers_at_risk) * cost_per_retention

    print(f'Number of Subscribers at Risk: {len(subscribers_at_risk)}')
    print(f'Predicted Annual Retention Budget: ${annual_retention_budget}')
