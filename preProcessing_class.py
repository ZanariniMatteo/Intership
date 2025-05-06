import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import utilitiesFun as ut


class DataHandler:
    def __init__(self, dataframe, cutting=[18, 25, 35, 50, 60, 70]):
        self.dataframe = dataframe
        self.cutting = cutting
    
    def clean_cols_NArows_and_create_isWeekend(self):
        data = self.dataframe.copy()
        # transform type
        data['Purchase Date'] = pd.to_datetime(data['Purchase Date'], format='%Y-%m-%d %H:%M:%S')
        # drop col
        data = data.drop(columns=["Age"])
        # drop Na rows
        data = data.loc[data['Returns'].notnull()]
        # new col using is_weekend function
        data['IsWeekend'] = data['Purchase Date'].apply(lambda x: ut.is_weekend(x))

        return data
    
    def dicotomizationAgeClass(self, dataAgg):
        previous_value = None
        labelsName = []
        for dx_border in self.cutting:
            if previous_value is not None:
                labelsName.append(f"{previous_value}-{dx_border}")
            previous_value = dx_border
 
        dataAgg['AgeCut']=pd.cut(dataAgg['Age'], bins=self.cutting, labels=labelsName)
        dfAgeClass = pd.get_dummies(dataAgg.set_index('Customer ID')['AgeCut']).astype(int).reset_index()

        dataAgg = dataAgg.merge(dfAgeClass, how='left', on='Customer ID')
        dataAgg = dataAgg.drop(columns=['AgeCut'])
        colList = [i for i in dfAgeClass.columns if i!='Customer ID']
        for col in colList: 
            dataAgg = dataAgg.rename(columns={f'{col}': f'AgeClass_{col}'})
        
        return dataAgg
    
    def dicotomProdCategory(self, data, dataAgg):
        # freq dict prod category
        tempData = data.groupby('Customer ID')['Product Category'].value_counts()
        freqProdCategory = {}
        for (code_id, product_category), count in tempData.items():
            if code_id not in freqProdCategory:
                freqProdCategory[code_id] = {}
            freqProdCategory[code_id][product_category] = count

        # df category for merge
        dfProdCat=pd.DataFrame.from_dict(freqProdCategory, orient='index').fillna(0)
        dfProdCat.columns = ['ProdCategory_' + col for col in dfProdCat.columns]
        dfProdCat['sum']=dfProdCat.sum(axis=1)
        dfProdCat=dfProdCat.div(dfProdCat['sum'], axis=0)
        dfProdCat=dfProdCat.drop(columns=['sum'])
        
        # merge
        dataAgg = dataAgg.merge(dfProdCat, how='left', left_on='Customer ID', right_index=True)
        
        return dataAgg
    
    def dicotomPaymentMethod(self, data, dataAgg):
        # freq dict payment method
        tempData = data.groupby('Customer ID')['Payment Method'].value_counts()
        freqPaymentMethod = {}
        for (code_id, payment_method), count in tempData.items():
            if code_id not in freqPaymentMethod:
                freqPaymentMethod[code_id] = {}
            freqPaymentMethod[code_id][payment_method] = count
        
        # df payment for merge
        dfPayMeth=pd.DataFrame.from_dict(freqPaymentMethod, orient='index').fillna(0)
        dfPayMeth.columns = ['PaymentMethod_' + col for col in dfPayMeth.columns]
        dfPayMeth['sum']=dfPayMeth.sum(axis=1)
        dfPayMeth=dfPayMeth.div(dfPayMeth['sum'], axis=0)
        dfPayMeth=dfPayMeth.drop(columns=['sum'])

        # merge
        dataAgg = dataAgg.merge(dfPayMeth, how='left', left_on='Customer ID', right_index=True)

        return dataAgg

    def creation_new_variables(self, data, dataAgg):
        # var: meanFrequencyPurchase
        meanFreqPurch = data.groupby('Customer ID').agg({'Purchase Date': lambda x: ut.meanFreqPurchase(x)})
        dataAgg = dataAgg.merge(round(meanFreqPurch), how='left', on='Customer ID')
        dataAgg = dataAgg.rename(columns={'Purchase Date': 'meanFreqPurchase_day'})
        
        # var: CustomerPurchaseLife_monhtly
        dataAgg['CustomerPurchaseLife_monthly'] = ((dataAgg['last_PurchaseDate'].dt.year - dataAgg['first_PurchaseDate'].dt.year))*12 + (dataAgg['last_PurchaseDate'].dt.month - dataAgg['first_PurchaseDate'].dt.month)
        
        # var: Recency
        dataAgg['Recency'] = (dataAgg['last_PurchaseDate'].max() - dataAgg['last_PurchaseDate']).dt.days

        ## dicotomization
        # vars: AgeClass
        dataAgg = self.dicotomizationAgeClass(dataAgg)
        # vars: ProdCategory
        dataAgg = self.dicotomProdCategory(data, dataAgg)
        # vars: PaymentMethod
        dataAgg = self.dicotomPaymentMethod(data, dataAgg)

        ## economic vars
        # AvgOrderValue: meanPurchase
        # PurchaseFrequency: avgFreqPurchase_days
        dataAgg['PF'] = dataAgg['meanFreqPurchase_day'].max()-dataAgg['meanFreqPurchase_day']
        dataAgg['PF'] = dataAgg['PF'].fillna(0.1)
        # Fidelty: PF * num_order
        dataAgg['Fidelty']=dataAgg['numOrders']*dataAgg['PF']
        # CustomerLifespan
        dataAgg['CL'] = dataAgg['CustomerPurchaseLife_monthly'] + 0.1
        
        # scaling
        scaler=MinMaxScaler()
        dataAgg[['AOV', 'Fidelty', 'CL']] = scaler.fit_transform(dataAgg[['meanPurchase', 'Fidelty', 'CL']])
        
        # CLTV and removing the economic vars
        dataAgg['CLTV']=dataAgg['AOV']*dataAgg['Fidelty']*dataAgg['CL']
        dataAgg = dataAgg.drop(columns=['PF', 'Fidelty', 'CL', 'AOV'])

        return dataAgg



    def preProcessing(self):
        data = self.clean_cols_NArows_and_create_isWeekend()
      
        # new dataset: aggregation by Customer ID
        dataAgg = data.groupby("Customer ID").agg(
            totPurchase = ('Total Purchase Amount', 'sum'), 
            meanPurchase = ('Total Purchase Amount', 'mean'),
            medianProdPrice = ('Product Price', 'median'), 
            totQuantity = ('Quantity', 'sum'), 
            meanQuantity = ('Quantity', 'mean'), 
            meanReturns = ('Returns', 'mean'),  # può essere considerato come prob. di fare Return
            numOrders = ('Customer ID', 'value_counts'), 
            last_PurchaseDate = ('Purchase Date', 'max'), 
            first_PurchaseDate = ('Purchase Date', 'min'), 
            propWeekend = ('IsWeekend', 'mean'), # prob di acquisto nel weekend
            isMale = ('Gender', lambda x: ut.ismale(x)), 
            Churn = ('Churn', 'min'), 
            Age = ('Customer Age', 'min')
            ).reset_index()
        
        dataAgg = self.creation_new_variables(data, dataAgg)


        # output
        self.dataAgg = dataAgg.set_index('Customer ID')

