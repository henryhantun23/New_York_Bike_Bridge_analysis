import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def main():
    dp = pd.read_csv("NYC_Bicycle_Counts_2016_Corrected.csv")  ##read in the files

    ##Data Cleanup
    dp = dp.replace("T", 0)
    dp = dp.replace(' \(S\)', '', regex=True)
    dp = dp.replace(',', '', regex=True)

    ##Part1 Bridges
    X_BMW = dp[['Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge']].astype(float)
    X_BMQ = dp[['Brooklyn Bridge', 'Manhattan Bridge', 'Queensboro Bridge']].astype(float)
    X_WMQ = dp[['Williamsburg Bridge', 'Manhattan Bridge', 'Queensboro Bridge']].astype(float)
    X_BWQ = dp[['Brooklyn Bridge', 'Williamsburg Bridge', 'Queensboro Bridge']].astype(float)
    Y = dp['Total'].astype(float)

    reg_1 = linear_model.LinearRegression()
    reg_1.fit(X_BMW, Y)
    reg_2 = linear_model.LinearRegression()
    reg_2.fit(X_BMQ, Y)
    reg_3 = linear_model.LinearRegression()
    reg_3.fit(X_WMQ, Y)
    reg_4 = linear_model.LinearRegression()
    reg_4.fit(X_BWQ, Y)

    print('Coefficients for Brooklyn Bridge, Manhattan Bridge, Williamsburg Bridge:')
    print(reg_1.coef_)
    print('Coefficients for Brooklyn Bridge, Manhattan Bridge, Queensboro Bridge:')
    print(reg_2.coef_)
    print('Coefficients for Williamsburg Bridge, Manhattan Bridge, Queensboro Bridge:')
    print(reg_3.coef_)
    print('Coefficients for Brooklyn Bridge, Williamsburg Bridge, Queensboro Bridge:')
    print(reg_4.coef_)
    

    print()
    ##Part2 Weather
    weather = dp[['High Temp (°F)', 'Low Temp (°F)', 'Precipitation', 'Total']].astype(float)
    weather = normalize(weather, axis=0, norm='max')
    X_weather = weather[:, :-1]
    Y_total = weather[:, -1]

    [X_weather_train, X_weather_test, y_total_train, y_total_test] = train_test_split(X_weather, Y_total, test_size=0.25, random_state=101)

    reg_weather = linear_model.LinearRegression()
    reg_weather.fit(X_weather_train, y_total_train)
    y_pred = reg_weather.predict(X_weather_test)

    print(reg_weather.coef_)
    print('Coefficient of Determination: %.2f'
          % r2_score(y_total_test, y_pred))
    
    t = np.arange(0, 1, 0.01875)

    plt.plot(X_weather, Y_total, 'k.')
    plt.plot(t, y_pred, 'g-')
    plt.title('Normalized Weather vs. Total Bicyclists')
    plt.xlabel('Normalized Weather')
    plt.ylabel('Normalized Total Bicyclists')
    plt.show()

    print()
    ##Part3 Precipitation
    X_total = dp[['Total']].astype(float)
    Y_rain = np.array([1 if i > 0 else 0 for i in dp['Precipitation'].astype(float)])

    logistic_model = LogisticRegressionCV(Cs=10, cv=10, class_weight="balanced").fit(X_total, Y_rain)

    print('Logistic Model Score: %.2f' 
            % logistic_model.score(X_total, Y_rain))


if __name__ == '__main__':
    main()

