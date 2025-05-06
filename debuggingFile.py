import pandas as pd
from preProcessing_class import DataHandler
from modellingPipeLine import ChurnModelling

if __name__=="__main__":
    df = pd.read_csv("bb_demo_churn/ecommerce_customer_data_large.csv", sep=",")

    processingPipeLine = DataHandler(df, cutting=[18, 30, 70]) # minimum age: 18, maximum age: 70
    processingPipeLine.preProcessing()  

    data=processingPipeLine.dataAgg
    features=processingPipeLine.features
    label=processingPipeLine.label  

    modellingPipeLine = ChurnModelling(data, features, label, test_dimension=0.25, set_seed=1234)
    a, b, c, d, e = modellingPipeLine.modelling()