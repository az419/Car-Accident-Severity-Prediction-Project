# Car Accident Severity Prediction Project

This is the final project from Coursera IBM Data Science Capstone Moduel. This project is focusing on building a machine learning model to predict the severity code of a car accident based on the big data (raw dataset can be found here: https://s3.us.cloud-object-storage.appdomain.cloud/cf-courses-data/CognitiveClass/DP0701EN/version-2/Data-Collisions.csv).



## Business Understanding:

With a rising number of car accidents happened in our community, an predictor must be built in order to give people a guidance of the potential of how severe a car accident might happen given some prerequisites like: road conditions, weather conditions, location, etc. When the predictor gives a severe code prediction, it could alert the driver to pay attention when driving in order to avoid the car collision happening.



## Data understanding:

This dataset consists of 37 attributes and those are known as independent variables and the 'SEVERITYCODE' is known as dependent variable. The aim is building a machine learning model to predict the 'SEVERITYCODE' based on those independent variables.

By looking at this dataset, there are some problems need to be fixed:

- It has too many attributes (37)

- It is an unbalanced dataset (SEVERITYCODE column)

- Some of the data is NaN

  <img src="/Users/mars/Library/Application Support/typora-user-images/Screen Shot 2020-09-22 at 5.42.35 PM.png" alt="Screen Shot 2020-09-22 at 5.42.35 PM" style="zoom:200%;" />



*Severity Code as shown as follows as a reference:*

*0: Little to no Probability (Clear Conditions)*

*1: Very Low Probability — Chance or Property Damage*

*2: Low Probability — Chance of Injury*

*3: Mild Probability — Chance of Serious Injury*

*4: High Probability — Chance of Fatality*



🔧The first problem can be fixed by removing the unwanted attributes and only keep the useful variables:

![Image for post](https://miro.medium.com/max/4028/1*GuKOxYXLpEnfMDgzCiNvzg.png)

Another problem can be observed is that the type of those data is object which can be difficult to implement further model training and testing. By coding those conditions with a corresponding number can fix this problem, codes are shown below:![Image for post](https://miro.medium.com/max/3676/1*u3ulQELElQLCBWqAJHg02g.png)



🔧 The second problem can be fixed by balancing out this dataset, what we want is to have a equal number of SEVERITYCODE 1 and SEVERITYCODE 2, codes are shown below:![Image for post](https://miro.medium.com/max/4080/1*cUMWIaZfD66SDSzy0JkZuA.png)



🔧 The third goal is to deal with the NaN value and in this case, I decided to drop the NaN value:

![](https://miro.medium.com/max/4048/1*6KbOJxTB907HVoUjngssgA.png)



## Data Preparation:

After finishing dealing with unprepared dataset, we are ready to move on to the data initialisation section.



🔧We use the 'asarray' method from numpy library to create an array for both independent and dependent variables for the sake of convenient model implementation.

![Image for post](https://miro.medium.com/max/4468/1*PggeVa8YatsjkfxIUE8JQg.png)



🔧Next, do the data normalisation for the independent variables:

![Image for post](https://miro.medium.com/max/4496/1*R3tIVdSwXiHcaoDHpkyFQg.png)



🔧Then, this dataset needs to be split into training and testing set:

![Image for post](https://miro.medium.com/max/4488/1*jPhFJ42KN99qrTVmVUIneQ.png)



## Modeling:

In this case, I come up with four methodologies which are:

- K Nearest Neighbours(KNN)
- Decision Tree
- Support Vector Machine(SVM)
- Logistic Regression

#### **K Nearest Neighbours(KNN):**

<img src="/Users/mars/Library/Application Support/typora-user-images/Screen Shot 2020-09-22 at 5.43.19 PM.png" alt="Screen Shot 2020-09-22 at 5.43.19 PM" style="zoom:200%;" />

<img src="/Users/mars/Library/Application Support/typora-user-images/Screen Shot 2020-09-22 at 5.43.33 PM.png" alt="Screen Shot 2020-09-22 at 5.43.33 PM" style="zoom:200%;" />

<img src="/Users/mars/Library/Application Support/typora-user-images/Screen Shot 2020-09-22 at 5.43.45 PM.png" alt="Screen Shot 2020-09-22 at 5.43.45 PM" style="zoom:200%;" />

#### Decision Tree

![Image for post](https://miro.medium.com/max/4480/1*1knCv-E9JLAJc4Ow4ZOqJA.png)



#### **Support Vector Machine(SVM)**

![Image for post](https://miro.medium.com/max/4484/1*4XufBM_MnMMZd9AI6N-4mg.png)



#### **Logistic Regression**

![Image for post](https://miro.medium.com/max/4480/1*Zw-7BvGqdv_mdWcViueEoA.png)



## Evaluation:

For each of the models we will calculate the **Jaccard index and F1-Score**:

- *The Jaccard Index, also known as the Jaccard similarity coefficient, is a statistic used in understanding the similarities between sample sets. The measurement emphasizes similarity between finite sample sets, and is formally defined as the size of the intersection divided by the size of the union of the sample sets.*

- *It is calculated from the precision and recall of the test, where the* ***precision is the number of correctly identified positive results divided by the number of all positive results, including those not identified correctly,\****and the* ***recall is the number of correctly identified positive results divided by the number of all samples that should have been identified as positive.\*** *The highest possible value of F1 is 1, indicating perfect precision and recall, and the lowest possible value is 0, if either the precision or the recall is zero.*

  

![Image for post](https://miro.medium.com/max/4476/1*wGZn-Wp2incFBpcd63Iwhw.png)



![Image for post](https://miro.medium.com/max/4448/1*l-E_JhK67tfWNbhWGrww2Q.png)



![Image for post](https://miro.medium.com/max/4472/1*ZwLdzLnuUvJ1pdzDg5diWg.png)



![Image for post](https://miro.medium.com/max/4456/1*MPsTNKqjZSHW3F4BddIgYg.png)



Results can be clearly shown in a table:

![Image for post](https://miro.medium.com/max/2876/1*YJhtcHb-TjGJw6RvS7zdVw.png)



<p>Copyright &copy; Alyson Zhang