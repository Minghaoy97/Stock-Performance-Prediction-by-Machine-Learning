# <div align='center' >EECS545 Machine Learning Final Project </div><div align='center' >Stock Performance Prediction by Machine Learning Methods </div>

## Project Information
This project predicts stock performance, either stock price or trend, based on three machine learning algorithms: 
Support Vector Regression (SVR), Long short-term memory (LSTM) and Random Forest (RF). We construct our own dataset by
combining calculated 12 technical factors and fundamental factors of 7 stocks, and 
pre-process the data before applying them to specific models by normalizing and dimension reduction techniques. We test the performance of SVR and LSTM by
root mean squared error (RMSE) while RF by accuracy. After the experiments,
SVR model performs better than LSTM in predicting stock price. RF model shows
relatively high accuracy in predicting the price trend and the performance closely
relates to choice of class and decision path.

## Group Members
Chengshuo Zhang cszhang@umich.edu &emsp; Shouren Wang shourenw@umich.edu

Yanlin Xiao yanlinx@umich.edu &emsp;&emsp;&emsp;&emsp;&ensp;&thinsp; Minghao Yang minghaoy@umich.edu
