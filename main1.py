import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA, KernelPCA
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import pickle


EXAMPLE_NO = 1


def streamlit_menu(example=1):
    if example == 1:
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",  # required
                options=["Home", "Predict"],  # required
                icons=["house", "book"],  # optional
                menu_icon="cast",  # optional
                default_index=0,  # optional
                
            )
        return selected

    

selected = streamlit_menu(example=EXAMPLE_NO)

if selected == "Home":
    st.title(f"Welcome to Prediction of Droughts using Weather & Soil Data")
    components.html(
                    """
    
                        <div>
                      
                        <img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2055480%2Ff5ad8544ab11d043972fb9209a874dd3%2Flevels.PNG?generation=1611148560535086&alt=media" width="500" height="600">
                        </div>
                      <div>
                      <table style="  border: 1px solid; background-color:white;>
                      <tr style="background-color:black;color:white;border: 1px solid;"> 
                        <th>Indicator</th>
                        <th>Description</th>
                      </tr>
                      <tr>
                        <td>WS10M_MIN => </td>
                        <td>Minimum Wind Speed at 10 Meters (m/s)</td>
                      </tr>
                      <tr>
                        <td>QV2M => </td>
                        <td>Specific Humidity at 2 Meters (g/kg)</td>
                      </tr>
                      <tr>
                        <td>T2M_RANGE => </td>
                        <td>Temperature Range at 2 Meters (C)</td>
                      </tr>
                      <tr>
                        <td>WS10M	 => </td>
                        <td>Wind Speed at 10 Meters (m/s)</td>
                      </tr>
                      <tr>
                        <td>T2M	 => </td>
                        <td>Temperature at 2 Meters (C)</td>
                      </tr>
                      <tr>
                        <td>WS50M_MIN => </td>
                        <td>Minimum Wind Speed at 50 Meters (m/s)</td>
                      </tr>
                      <tr>
                        <td>T2M_MAX	=> </td>
                        <td>Maximum Temperature at 2 Meters (C)</td>
                      </tr>
                      <tr>
                        <td>WS50M	=> </td>
                        <td>Wind Speed at 50 Meters (m/s)</td>
                      </tr>
                      <tr>
                        <td>TS	 => </td>
                        <td>Earth Skin Temperature (C)</td>
                      </tr>
                      <tr>
                        <td>WS50M_RANGE	 => </td>
                        <td>Wind Speed Range at 50 Meters (m/s)</td>
                      </tr>
                      <tr>
                        <td>WS50M_MAX => </td>
                        <td>Maximum Wind Speed at 50 Meters (m/s)</td>
                      </tr>
                      <tr>
                        <td>WS10M_MAX => </td>
                        <td>Maximum Wind Speed at 10 Meters (m/s)</td>
                      </tr>
                      <tr>
                        <td>WS10M_RANGE => </td>
                        <td>Wind Speed Range at 10 Meters (m/s)</td>
                      </tr>
                      <tr>
                        <td>PS => </td>
                        <td>Surface Pressure (kPa)</td>
                      </tr>
                      <tr>
                        <td>T2MDEW => </td>
                        <td>Dew/Frost Point at 2 Meters (C)</td>
                      </tr>
                      <tr>
                        <td>T2M_MIN	=> </td>
                        <td>Minimum Temperature at 2 Meters (C)</td>
                      </tr>
                      <tr>
                        <td>T2MWET => </td>
                        <td>Wet Bulb Temperature at 2 Meters (C)</td>
                      </tr>
                      <tr>
                        <td>PRECTOT => </td>
                        <td>Precipitation (mm day-1)</td>
                      </tr>
                    </table>
                    </div>
                        """,
                        height=1050,
                      width=600,
                      )
if selected == "Predict":
    st.title(f"Enter the details to predict drought")
    with st.form(key = "form1"):
        WS10M_MIN = st.text_input(label = "Enter the Minimum Wind Speed at 10 Meters(WS10M_MIN) (m/s) ")
        QV2M = st.text_input(label = "Enter the Specific Humidity at 2 Meters(QV2M) (g/kg) ")
        T2M_RANGE = st.text_input(label = "Enter the Temperature Range at 2 Meters(T2M_RANGE) (C) ")
        WS10M = st.text_input(label = "Enter the Wind Speed at 10 Meters(WS10M) (m/s) ")
        T2M= st.text_input(label = "Enter the Temperature at 2 Meters (C)(T2M) ")
        WS50M_MIN=st.text_input(label = "Enter the Minimum Wind Speed at 50 Meters(WS50M_MIN) (m/s) ")
        T2M_MAX	= st.text_input(label = "Enter the Maximum Temperature at 2 Meters (T2M-MAX)(C)")
        WS50M= st.text_input(label = "Enter the Wind Speed at 50 Meters(WS50M) (m/s) ")
        TS= st.text_input(label = "Enter the Earth Skin Temperature(TS) (C) ")
        WS50M_RANGE= st.text_input(label = "Enter the Wind Speed Range at 50 Meters(WS50M_RANGE) (m/s) ")
        WS50M_MAX= st.text_input(label = "Enter the Maximum Wind Speed at 50 Meters(WS50M_MAX) (m/s)")
        WS10M_MAX= st.text_input(label = "Enter the Wind Speed Range at 10 Meters(WS10M_MAX)(m/s) ")
        WS10M_RANGE= st.text_input(label = "Enter the Wind Speed Range at 10 Meters(WS10M_RANGE)(m/s)")
        PS= st.text_input(label = "Enter the Surface Pressure(PS) (kPa)")
        T2MDEW= st.text_input(label = "Enter the Dew/Frost Point at 2 Meters(T2MDEW)(C)")
        T2M_MIN= st.text_input(label = "Enter the Minimum Temperature at 2 Meters(T2M_MIN) (C)")
        T2MWET= st.text_input(label = "Enter the Wet Bulb Temperature at 2 Meters(T2MWET) (C)")
        PRECTOT= st.text_input(label = "Enter the Precipitation(PRECTOT) (mm day-1)")
        submit = st.form_submit_button(label = "Submit")
        if submit:
          WS10M_MIN = float(WS10M_MIN)
          QV2M = float(QV2M)
          T2M_RANGE = float(T2M_RANGE)
          WS10M = float(WS10M)
          T2M= float(T2M)
          WS50M_MIN= float(WS50M_MIN)
          T2M_MAX	= float(T2M_MAX)
          WS50M= float(WS50M)
          TS= float(TS)
          WS50M_RANGE= float(WS50M_RANGE)
          WS50M_MAX= float(WS50M_MAX)
          WS10M_MAX= float(WS10M_MAX)
          WS10M_RANGE= float(WS10M_RANGE)
          PS= float(PS)
          T2MDEW= float(T2MDEW)
          T2M_MIN= float(T2M_MIN)
          T2MWET= float(T2MWET)
          PRECTOT= float(PRECTOT)
          # st.write("WS10M_MIN: ",WS10M_MIN)
          # st.write(type(WS10M_MIN))
          drought_df = pd.read_csv('C:/Users/BHARGAV/Downloads/us-drought-meteorological-data/validation_timeseries/validation_timeseries.csv')
          # st.write(drought_df.head())
          drought_df = drought_df.dropna()
          drought_df['year'] = pd.DatetimeIndex(drought_df['date']).year
          drought_df['month'] = pd.DatetimeIndex(drought_df['date']).month
          drought_df['day'] = pd.DatetimeIndex(drought_df['date']).day
          drought_df['score'] = drought_df['score'].round().astype(int)
          drought_df = drought_df[(drought_df['PRECTOT'] <= drought_df['PRECTOT'].mean() + 3*drought_df['PRECTOT'].std()) &
          (drought_df['PRECTOT'] >= drought_df['PRECTOT'].mean() - 3*drought_df['PRECTOT'].std())]

          drought_df = drought_df[(drought_df['PS'] <= drought_df['PS'].mean() + 3*drought_df['PS'].std()) &
          (drought_df['PS'] >= drought_df['PS'].mean() - 3*drought_df['PS'].std())]

          drought_df = drought_df[(drought_df['QV2M'] <= drought_df['QV2M'].mean() + 3*drought_df['QV2M'].std()) &
          (drought_df['QV2M'] >= drought_df['QV2M'].mean() - 3*drought_df['QV2M'].std())]

          drought_df = drought_df[(drought_df['T2M'] <= drought_df['T2M'].mean() + 3*drought_df['T2M'].std()) &
          (drought_df['T2M'] >= drought_df['T2M'].mean() - 3*drought_df['T2M'].std())]

          drought_df = drought_df[(drought_df['T2MDEW'] <= drought_df['T2MDEW'].mean() + 3*drought_df['T2MDEW'].std()) &
          (drought_df['T2MDEW'] >= drought_df['T2MDEW'].mean() - 3*drought_df['T2MDEW'].std())]

          drought_df = drought_df[(drought_df['T2MWET'] <= drought_df['T2MWET'].mean() + 3*drought_df['T2MWET'].std()) &
          (drought_df['T2MWET'] >= drought_df['T2MWET'].mean() - 3*drought_df['T2MWET'].std())]

          drought_df = drought_df[(drought_df['T2M_MAX'] <= drought_df['T2M_MAX'].mean() + 3*drought_df['T2M_MAX'].std()) &
          (drought_df['T2M_MAX'] >= drought_df['T2M_MAX'].mean() - 3*drought_df['T2M_MAX'].std())]

          drought_df = drought_df[(drought_df['T2M_MIN'] <= drought_df['T2M_MIN'].mean() + 3*drought_df['T2M_MIN'].std()) &
          (drought_df['T2M_MIN'] >= drought_df['T2M_MIN'].mean() - 3*drought_df['T2M_MIN'].std())]

          drought_df = drought_df[(drought_df['T2M_RANGE'] <= drought_df['T2M_RANGE'].mean() + 3*drought_df['T2M_RANGE'].std()) &
          (drought_df['T2M_RANGE'] >= drought_df['T2M_RANGE'].mean() - 3*drought_df['T2M_RANGE'].std())]

          drought_df = drought_df[(drought_df['TS'] <= drought_df['TS'].mean() + 3*drought_df['TS'].std()) &
          (drought_df['TS'] >= drought_df['TS'].mean() - 3*drought_df['TS'].std())]

          drought_df = drought_df[(drought_df['WS10M'] <= drought_df['WS10M'].mean() + 3*drought_df['WS10M'].std()) &
          (drought_df['WS10M'] >= drought_df['WS10M'].mean() - 3*drought_df['WS10M'].std())]

          drought_df = drought_df[(drought_df['WS10M_MAX'] <= drought_df['WS10M_MAX'].mean() + 3*drought_df['WS10M_MAX'].std()) &
          (drought_df['WS10M_MAX'] >= drought_df['WS10M_MAX'].mean() - 3*drought_df['WS10M_MAX'].std())]

          drought_df = drought_df[(drought_df['WS10M_MIN'] <= drought_df['WS10M_MIN'].mean() + 3*drought_df['WS10M_MIN'].std()) &
          (drought_df['WS10M_MIN'] >= drought_df['WS10M_MIN'].mean() - 3*drought_df['WS10M_MIN'].std())]

          drought_df = drought_df[(drought_df['WS10M_RANGE'] <= drought_df['WS10M_RANGE'].mean() + 3*drought_df['WS10M_RANGE'].std()) &
          (drought_df['WS10M_RANGE'] >= drought_df['WS10M_RANGE'].mean() - 3*drought_df['WS10M_RANGE'].std())]

          drought_df = drought_df[(drought_df['WS50M'] <= drought_df['WS50M'].mean() + 3*drought_df['WS50M'].std()) &
          (drought_df['WS50M'] >= drought_df['WS50M'].mean() - 3*drought_df['WS50M'].std())]

          drought_df = drought_df[(drought_df['WS50M_MAX'] <= drought_df['WS50M_MAX'].mean() + 3*drought_df['WS50M_MAX'].std()) &
          (drought_df['WS50M_MAX'] >= drought_df['WS50M_MAX'].mean() - 3*drought_df['WS50M_MAX'].std())]

          drought_df = drought_df[(drought_df['WS50M_MIN'] <= drought_df['WS50M_MIN'].mean() + 3*drought_df['WS50M_MIN'].std()) &
          (drought_df['WS50M_MIN'] >= drought_df['WS50M_MIN'].mean() - 3*drought_df['WS50M_MIN'].std())]

          drought_df = drought_df[(drought_df['WS50M_RANGE'] <= drought_df['WS50M_RANGE'].mean() + 3*drought_df['WS50M_RANGE'].std()) &
          (drought_df['WS50M_RANGE'] >= drought_df['WS50M_RANGE'].mean() - 3*drought_df['WS50M_RANGE'].std())]
          independent_variables = drought_df.drop('score', 1)
          independent_variables = independent_variables.drop('fips', 1)
          independent_variables = independent_variables.drop('date', 1)
          # st.write(independent_variables.head())
          target = drought_df['score']
          X_train, X_test, y_train, y_test = train_test_split(independent_variables, target, test_size=0.2, random_state=0)
          # st.write(X_test)
          # sc = StandardScaler()
          # X_train = sc.fit_transform(X_train)
          # X_test = sc.transform(X_test)
          # st.write(X_test)
          independent_variables = independent_variables.drop('PRECTOT', 1)
          independent_variables = independent_variables.drop('T2MWET', 1)
          independent_variables = independent_variables.drop('WS10M_MAX', 1)
          independent_variables = independent_variables.drop('WS10M_MIN', 1)
          independent_variables = independent_variables.drop('WS50M_MIN', 1)
          independent_variables = independent_variables.drop('month', 1)
          independent_variables = independent_variables.drop('year', 1)
          independent_variables = independent_variables.drop('day', 1)
          # st.write(independent_variables.head())
          X_train, X_test, y_train, y_test = train_test_split(independent_variables, target, test_size=0.2, random_state=0)
          # sc = StandardScaler()
          # X_train = sc.fit_transform(X_train)
          # X_test = sc.transform(X_test)
          # st.write(type(X_test))
          dff = [PS,
          QV2M,
          T2M,
          T2MDEW,
          T2M_MAX,
          T2M_MIN,
          T2M_RANGE,
          TS,
          WS10M,
          WS10M_RANGE,
          WS50M,
          WS50M_MAX,
          WS50M_RANGE
          ]
          arr = np.array(dff)
          # st.write(type(arr))
          arr = arr.reshape(1,-1)
          # st.write("Arr:\n",arr)
          # st.write("X_test:\n",X_test)
          # sc = StandardScaler()
          # X_train = sc.fit_transform(X_train)
          # arr = sc.transform(arr)
          st.write(arr)
          # st.write(X_train)
          # st.write(y_train)
          RF_classifier = RandomForestClassifier(n_estimators = 20, max_depth=70, random_state=0)
          RF_classifier.fit(independent_variables,target)
          y_pred_RF = RF_classifier.predict(arr)
          st.write("RF:",y_pred_RF)
          # knn_classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
          # knn_classifier.fit(independent_variables, target)
          # y_pred_knn = knn_classifier.predict(arr)
          # st.write("KNN:",y_pred_knn)
