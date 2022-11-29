# ml_car_price_prediction # 
The model to predict the used car price based on the various feature selection techniques. 

We are provided with the used car data consists of 18 features and 426k rows. The intent of this project is to identify the most contributing factors influencing the price of used cars. The result will help the dealer of used cars keep up with the inventory and cost for the better customer satisfaction. 

It is practically challening for human to analyze such a huge dataset for making a decision hence, we need the solution that can take big dataset and conclude the analysis faster. What could be a better alternative than computers with decent CPU and memory? We will plan to use the supervised machine learning algorithms for the condusive analysis.

As with any human would do, the Machine learning(ML) algorithm need to understand the data beforehand to make any smarter decision. The jupyter notebook (used_car_price_model.ipynb) is used to instruct the computer to perform the ML analysis. In the subsequent read you will get how using CRISP-DM framework we came up with the model to make the analysis of used car easy.

* **Data Understanding**
To understand the data, we plotted various combination of the variables (columns) to see how the data is distributed. Looking closely at the plots, you will notice 

    * The data is unequally distributed with varying mean and standard deviation. 
    * Some outliers like in the feature "year". (The boxplot tells you about any noticable outliers). 
    * The price of diesel car, fair condition, and mid-size were costly than other combination. 
    * The luxury cars Mercedes-benz and Volvo were costlier than economy cars like Gmc, Chevrolet, Toyota, and Ford. 
    * Surprisingly, Green color of the cars were costlier than any other. 
    * Manual transmission of cars were costlier than automatic. However, automatic are sold more than the manual or others(Hybrid). 
    * The most cars sold were of type offroad, sedan and suv. 
    * The data consists of both categorical and numeric columns thus the correlation didn't reveal noticeable information. Except for the fact that there was moderate negative relationship between the year and odometer. 
    * It was important to identify null/empty rows because having the missig value will not help with the analysis. We used msno library to plot the missing value in the data and to the surprise, there are many missing values. 

* **Data Preparation**
As highlighted in the data understanding, the data consists of both numeric and categorical data. Humans are good with the non-numeric data however, the computers aren't. The computers understand the numeric input thus we will have to convert the categorical columns/features into the computer readable format. 
Another factor to consider in the data preparation is about fixing the variation in the data. The minimum, maximum, mean and std are varying vastly for the numeric columns. Thus we need to fix this variation for the computer or ML model to weigh the appropriately. 

    * We will add the "age" feature in the data by substrating the current year with the year column in the data. 
    * The features are separated into three subsets, categorical, numeric, and non-essential. The non-essential are dropped from the data as they won't add any value to the model. Non-essential columns - ['VIN', 'id', 'state']
    * Numeric columns/features 
        * Sklearn comes with the library SimpleImpute to address the missing value in the data. SimpleImpute takes the defined strategy while instantiating the class and fill the missing values with it. We will use "median" strategy to fill in the empty rows of the numeric features. 
    * Categorical features 
        * Filling in the missing values in the categorical data will be tricky as we will need a transformer that can treat the non-numeric columns like the one we used in the case of numeric features However, I don't think such library exists. So, to address the missing values, we will first transform the column to the numeric (maybe using OneHot encoding, LabelEncoder, OrdinalEncoder), then apply Imputing on the features i.e. Transform -> Impute. 
        The encoder fails if you have the missing value so now what we do? The approach now is to keep the null/empty rows out of the transformation logic, apply the desired transformation, and put the null rows back. i.e. Keep empty rows aside -> Transform the data -> place the null rows back -> Impute. We will use LabelEncoder and IterativeImputer to transform the categorical features holistically. 
    * The last step is to normalize the data as it consists of huge variation but we will do it while building the model. 
Now that the data is in the computer readable format, we can prepare the model to perform the analysis. 

* **Modeling** preparing the maching learning model that will attain the goal of this project. 
We will use Linear, Ridge, and Lasso Regressions. Remember we haven't scaled or normalized our data yet and it still contains the variation. We will do both scaling and modeling subsequently using the Sklearn pipeline. We are to run many combinations to build the effective model. In the Jupyter noetbook review the various pipeline construted and for each combination the Loss Function. We have splitted the data into train and test sets of 7:3 ratio. Used training data to construct the model and test data to calculate the Loss function. 

* **Evaluation** 
After the model is built, we are to analyze the coefficent of the features considered by the model. Think of the coefficent as the way of the model to tell which features/columns are to be given importance during prediction. The features coefficent will help the dealer to understand what factors are important for the used cars. 
Based on the plot, it seems odometer, transmission, age, and size are the most important features that consitutes to the price of the used cars. 




  


