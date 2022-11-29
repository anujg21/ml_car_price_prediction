# ml_car_price_prediction # 
The model to predict the used car price is based on the various feature selection techniques. 

We have used car data consisting of 18 features and 426k rows. This project intends to identify the most contributing factors influencing the price of used cars. The result will help the dealer of used cars keep up with the inventory and cost for better customer satisfaction. 

It is challenging for humans to determine results on such a massive dataset. Hence, we need a solution that can take an extensive dataset and conclude the analysis faster. What could be a better alternative than computers with decent CPU and memory? We will use the supervised machine learning algorithms for the conducive analysis.

As any human would do, the Machine learning(ML) algorithm needs to understand the data beforehand to make smarter decisions. The Jupyter notebook (used_car_price_model.ipynb) is used to instruct the computer to perform the ML analysis. The subsequent read guides how using the CRISP-DM framework, we came up with the model to analyze used cars.

* **Data Understanding**
To understand the data, we plotted various variables (columns) combinations to see the data distribution. Looking closely at the plots indicates,

    * The data is unequally distributed with varying mean and standard deviation. 
    * Some outliers like in the feature "year". (The boxplot tells you about any noticeable outliers). 
    * The price of a diesel car, fair condition, and mid-size were more costly than other combinations. 
    * Mercedes-Benz and Volvo luxury cars were costlier than economy cars like GMC, Chevrolet, Toyota, and Ford. 
    * Surprisingly, the Green color of the cars was costlier than any other. 
    * Manual transmission of cars was costlier than automatic. However, automatic sells more than manual or others(Hybrid). 
    * Most cars sold were of type offroad, sedan, and SUV. 
    * The data consists of categorical and numeric columns; thus, the correlation did not reveal noticeable information. Except for the fact that there was a moderate negative relationship between the year and the odometer. 
    * It was essential to identify null/empty rows because having the missing value will not help with the analysis. We used the msno library to plot the missing value in the data, and to our surprise, there were many missing values. 

* **Data Preparation**
As highlighted in the data understanding, the data consists of both numeric and categorical data. Humans are good with non-numeric data. However, computers are not. The computers understand the numeric input. Thus, we will have to convert the categorical columns/features into a computer-readable format. 
Another factor to consider in the data preparation is fixing the data variation. The minimum, maximum, mean, and std vary vastly for the numeric columns. Thus we need to fix this variation for the computer or ML model to weigh them appropriately. 

    * We will add the "age" feature in the data by substrating the current year with the year column. 
    * The features are separated into categories: categorical, numeric, and non-essential. The non-essential are dropped from the data as they will not add any value to the model. Non-essential columns - ['VIN', 'id', 'state']
    * Numeric columns/features 
        * Sklearn comes with the library SimpleImpute to address the missing value in the data. SimpleImpute takes the defined strategy while instantiating the class and filling in the missing values. We will use the "median" strategy to fill in the empty rows of the numeric features. 
    * Categorical features 
        * Filling in the missing values in the categorical data will be tricky as we will need a transformer that can treat the non-numeric columns like the one we used in the case of numeric features. So, to address the missing values, we will first transform the column to the numeric (maybe using OneHot encoding, LabelEncoder, OrdinalEncoder), then apply Imputing on the features, i.e., Transform -> Impute. 
        The encoder fails with the missing value, so what do we do? The approach is to keep the null/empty rows out of the transformation logic, apply the desired transformation, and put the null rows back. i.e., Keep empty rows aside -> Transform the data -> place the null rows back -> Impute. We will use LabelEncoder and IterativeImputer to transform the categorical features holistically. 
    * The last step is to normalize the data as it consists of colossal variation, but we will do it while building the model. 
Now that the data is in the computer-readable format, we can prepare the model to perform the analysis. 

* **Modeling** preparing the machine learning model that will attain the goal of this project. 
We will use Linear, Ridge, and Lasso Regressions. Remember, we still need to scale or normalize our data, which contains the variation. We will do both scaling and modeling subsequently using the Sklearn pipeline. We are to run many combinations to build an effective model. In the Jupyter notebook review, the various pipeline constructed and the Loss Function for each combination. We have split the data into train and test sets of 7:3 ratio. Used training data to construct the model and test data to calculate the Loss function. 

* **Evaluation** 
After the model, we are to analyze the coefficient of the features considered by the model. Think of the coefficient as the way the model tells which features/columns are to be given importance during prediction. The features coefficient will help the dealer to understand what factors are essential for the used cars. Loss functions help understand the accuracy of the model and how far off is the prediction of the model given the test data. Looking at the MSE, MAE, and RMSE of the scaled models, the loss functions seem related. 
***Based on the plot, the odometer, manufacturer, transmission, age, and size are the most critical features that constitute the price of the used cars.*** 

* **Next steps and consideration** 
    * Reading a couple of online articles about other models, the KNN & Random Forest model can help better predict such a scenario than any other model. So, it will be worth looking at the KNN model's loss function. 
    * The dependent/target column price is not scaled. After scaling the price column, we should reconstruct the model and calculate the Loss functions. 
    * We should consider plotting the distribution plot of the raw data using the distplot library. 
    * In the current process, we ran the combination of models separately. Using the Gridsearch & Makepipe Sklearn libraries, we should reduce the number of lines in the code. 
    * While plotting the predicted value and test data, I noticed some discrepancies, so it will be worth understanding the cause and reconstructing the model.
    * Removing outliers from the data and reconstructing the models. 
    * Remvoving duplicates from the data. 
    
