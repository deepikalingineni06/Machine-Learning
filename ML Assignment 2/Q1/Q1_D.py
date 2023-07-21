import numpy as np
import matplotlib.pyplot as plt
import math

def main():
    print('START Q1_D\n')
	
    # read data 1
    Portfolio_1 = open("datasets/Q1_B_train.txt","r")
    Portfolio_1_Information = Portfolio_1.readlines()
    for Token in range(len(Portfolio_1_Information)):
        Temporary_Information = Portfolio_1_Information[Token]
        Temporary_Information = Dataa_Filtering(Temporary_Information).split(",")
        Portfolio_1_Information[Token] = [float(Temporary_Information[0]), float(Temporary_Information[1])]

    # read data 2
    Portfolio_2 = open("datasets/Q1_C_test.txt","r")
    Portfolio_2_Information = Portfolio_2.readlines()
    for Token in range(len(Portfolio_2_Information)):
        Temporary_Information = Portfolio_2_Information[Token]
        Temporary_Information = Dataa_Filtering(Temporary_Information).split(",")
        Portfolio_2_Information[Token] =  [float(Temporary_Information[0]), float(Temporary_Information[1])]
       
    # read data 3
    Portfolio_3 = open("datasets/Q3_data.txt","r")
    Portfolio_3_Information = Portfolio_3.readlines()
    for Token in range(len(Portfolio_3_Information)):
        Temporary_Information = Portfolio_3_Information[Token]
        Temporary_Information = Dataa_Filtering(Temporary_Information).split(",")
        Portfolio_3_Information[Token] = [float(Temporary_Information[0]), float(Temporary_Information[1]), int(Temporary_Information[2]), str(Temporary_Information[3])]

    # load data 
    Last_Data_Info = Portfolio_1_Information[:20] + Portfolio_2_Information

    # loop to iterate k
    for idxP in range(1,11):
        print("Value of K is --", idxP)
        
        # loop for all values of d
        for idxd in range(0,7):
            print("Value of D --", idxd)
            
            # prediction and printing results
            print("MSE --", (( np.array([Regression_Method(idxd, idxP, Temp_Value[0]) for Temp_Value in  Last_Data_Info]) - np.array([Temp_Value[1] for Temp_Value in  Last_Data_Info]))**2).mean())
        print()

    # loop to iterate k
    for idxP in range(1,11):
        Prediction_List = []
        
        # loop to iterate d
        for idxd in range(0,7):
            List_Prediction = []
            List_Actual_Value = []
            for Temp_Value in  Last_Data_Info:
                Temp_X_Value = Temp_Value[0]
                List_Actual_Value.append(Temp_Value[1])
                
                # modle run
                
                Temp_Y_Value = Regression_Method(idxd, idxP, Temp_X_Value)
                List_Prediction.append(Temp_Y_Value)
            List_Prediction = np.array(List_Prediction)
            Prediction_List.append(List_Prediction)
        
        # loading things in variables
        Temp_X_Data = [Temp_Value[0] for Temp_Value in  Last_Data_Info]
        Pred_Y_Data_1 = Prediction_List[0]
        Pred_Y_Data_2 = Prediction_List[1]
        Pred_Y_Data_3 = Prediction_List[2]
        Pred_Y_Data_4 = Prediction_List[3]
        Pred_Y_Data_5 = Prediction_List[4]
        Pred_Y_Data_6 = Prediction_List[5]
        Pred_Y_Data_7 = Prediction_List[6]
        
        # plotting results
        plt.scatter(Temp_X_Data, Pred_Y_Data_1, c='g', label = " d=0")
        plt.scatter(Temp_X_Data, Pred_Y_Data_2, c='b', label = " d=1")
        plt.scatter(Temp_X_Data, Pred_Y_Data_3, c='c', label = " d=2")
        plt.scatter(Temp_X_Data, Pred_Y_Data_4, c='m', label = " d=3")
        plt.scatter(Temp_X_Data, Pred_Y_Data_5, c='r', label = " d=4")
        plt.scatter(Temp_X_Data, Pred_Y_Data_6, c='y', label = " d=5")
        plt.scatter(Temp_X_Data, Pred_Y_Data_7, c='k', label = " d=6")
        plt.title("K value "+str(idxP))
        plt.legend()
        plt.show()  

# data cleaning function
def Dataa_Filtering(My_Data):
    My_Data = My_Data.replace(" ","")
    My_Data = My_Data.replace("\n","")
    My_Data = My_Data.replace("(","")
    My_Data = My_Data.replace(")","")
    return My_Data

# simple trigonometric regression
def Regression_Method(idxd, idxP, Temp_X_Value):
    Temp_Y_Value = 1  + sum([ math.sin(idxd*idxP*Temp_X_Value)*math.sin(idxd*idxP*Temp_X_Value) for idxd in range(1, idxd+1)])
    return Temp_Y_Value

# class for locally weight regression
class Temporary_Local_Weighted_Regression:
    
    # constructor
    def __init__(self):
        self.Tau_Value = 0.204
        
    # training method
    def Train_Model(self, X_0_Value, X_Value, Y_Value):
        X_0_Value = np.r_[1, X_0_Value]
        Length_X_Value = np.ones(len(X_Value))
        X_Value = np.c_[Length_X_Value, X_Value]
        Weight_X = X_Value.T * self.Weight_Calci(X_0_Value, X_Value)
        Cal_Ins_One = np.linalg.pinv(Weight_X @ X_Value)
        Value_Theta = Cal_Ins_One @ Weight_X @ Y_Value
        Final_results = X_0_Value @ Value_Theta
        return Final_results
    
    # weight calculate
    def Weight_Calci(self,X_0_Value, X_Value):
        Tau_2_Power = (-2 * (self.Tau_Value **2) )
        Final_results = np.exp(np.sum((X_Value - X_0_Value) ** 2, axis=1) / Tau_2_Power)
        return Final_results

# class for logistic regression
class LogidticRegression:
    
    # constrctor
    def __init__(self,Alpha_Value=0.001,Itr_Count=400):
        self.Alpha_Value = Alpha_Value
        self.Itr_Count = Itr_Count
    
    # training Model
    def Train_Model(self,Temp_X_Val,Temp_Y_Value):
        weights = np.zeros((np.shape(Temp_X_Val)[1]+1,1))
        Temp_X_Val = np.c_[np.ones((np.shape(Temp_X_Val)[0],1)),Temp_X_Val]
        Itr_COst = np.zeros(self.Itr_Count,)
        for itr in range(self.Itr_Count):
            Z_Dot_Valu = np.dot(Temp_X_Val,weights)
            weights = weights - self.Alpha_Value*np.dot(Temp_X_Val.T,(1/(1+np.exp(-Z_Dot_Valu)))-np.reshape(Temp_Y_Value,(len(Temp_Y_Value),1)))
            Itr_COst[itr] = self.Cost_method(weights, Temp_X_Val, Temp_Y_Value)
        self.weights = weights
    
    # codt method
    def Cost_method(self, Value_Theta, Temp_X_Val, Temp_Y_Value):
        Z_Dot_Valu = np.dot(Temp_X_Val,Value_Theta)
        Xero_Cost_Value = Temp_Y_Value.T.dot(np.log(1/(1+np.exp(-Z_Dot_Valu))))
        cost1 = (1-Temp_Y_Value).T.dot(np.log(1-(1/(1+np.exp(-Z_Dot_Valu)))))
        Cost_method = -((cost1 + Xero_Cost_Value))/len(Temp_Y_Value)
        return Cost_method
    
    # predicting methods
    def Predict_method(self,Temp_X_Val):
        weights_ins = np.zeros((np.shape(Temp_X_Val)[1]+1,1))
        X_ins = np.c_[np.ones((np.shape(Temp_X_Val)[0],1)),Temp_X_Val]
        ans = weights_ins,X_ins
        Z_Dot_Valu = np.dot(ans[1],self.weights)
        Lis = []
        for itr in (1/(1+np.exp(-Z_Dot_Valu))):
            if itr>0.5:
                Lis.append(1)
            else:
                Lis.append(0)
        return Lis

# function to find one out 
def Leave_One_Out(Train_Temp_Data, Train_Label_Data):
    predListdata = []
    
    # get data
    for Train_Temp_Data_Counter in range(len(Train_Temp_Data)):
        instanceFeatureCollection = list(Train_Temp_Data)
        instanceLabelCollection = list(Train_Label_Data)
        
        # filtering one row out
        testFeatureCollection = instanceFeatureCollection[Train_Temp_Data_Counter]
        testlabelCollection = instanceLabelCollection[Train_Temp_Data_Counter]
        del instanceFeatureCollection[Train_Temp_Data_Counter]
        del instanceLabelCollection[Train_Temp_Data_Counter]
        sampleFeatureslCollection = np.array(instanceFeatureCollection)
        samplelabelCollection = np.array(instanceLabelCollection)
        Target_Features = np.array([testFeatureCollection])
        
        # model run
        sampleModelRun = LogidticRegression(Alpha_Value=0.01,Itr_Count=1000)
        sampleModelRun.Train_Model(sampleFeatureslCollection, samplelabelCollection)
        Temp_Prepctions = sampleModelRun.Predict_method(Target_Features)
        predListdata.append([testlabelCollection, Temp_Prepctions[0]])
    List_1_X_Value = [float(Temp_X_Value[0]) for Temp_X_Value in predListdata]
    List_2_X_Value = [float(Temp_X_Value[1]) for Temp_X_Value in predListdata]
    Right_Counter = 0
    for itr in range(len(List_1_X_Value)):
        if List_1_X_Value[itr] == List_2_X_Value[itr]:
            Right_Counter += 1
    returnAns = Right_Counter / float(len(List_2_X_Value)) * 100.0
    return returnAns

if __name__ == "__main__":
    main()