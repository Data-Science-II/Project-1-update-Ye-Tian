# Project-1-update-Ye-Tian

In this project, I used five models: multiple linear regression, quadractic regression, quadX regression, cubic regression and cubicX regression. With them, I tried forward selection, backward elimination and stepwise regression feature selections. The datasets I tested on are:
1. auto-mpg.csv
2. concrete.csv
3. Winequality-red.csv
4. Forestfires.csv
5. AirQualityUCI.csv
6. PRSA_data_2010.1.1-2014.12.31.csv
To run the file, place the documents PredictorMat.scala, Regression.scala, QuadRegression.scala,QuadXRegression.scala, CubicRegression.scala, CubicXRegression.scala under scalation_1.6/scalation_modeling/src/main/scala/scalation/analytics, and put in the following codes in command prompt windows:
cd scalation_modeling  sbt  runMain.scalation.analytics.object
where the object should be replaced by, e.g., Reg_forSel_AutoMPG_Test, which is the regression model with forward selection on AutoMPG datasets. To change the datasets, put the datasets under scalation_1.6\data\analytics, and change the dataset input in the corresponding test object in related scala files.
