import pandas as pd
import tensorflow as tf

#preparation du dataset
df = pd.read_csv('CA_State_Fleet___2011-2014_.csv')

df2 = df[['ModelYear','FuelType','EngineConfiguration','DispositionDate','DispositionMileage','PurchasePrice','DispositionSoldAmount']]
df2 = df2.dropna()
df2 = df2[df2['DispositionSoldAmount']!='$0.00']

df2['ModelYear'] = df2['ModelYear'].map(lambda x:int(x))
df2['DispositionMileage'] = df2['DispositionMileage'].map(lambda x:int(x))
df2['DispositionDate'] = df2['DispositionDate'].map(lambda x: int(x.split('/')[2]))
df2['DispositionSoldAmount'] = df2['DispositionSoldAmount'].map(lambda x: float(x.lstrip('$').rstrip('aAbBcC')))
df2['PurchasePrice'] = df2['PurchasePrice'].map(lambda x: float(x.lstrip('$').rstrip('aAbBcC')))

df2 = df2[df2['ModelYear']>=2000]
df2 = df2[df2['PurchasePrice']>= 500]
df2 = df2[df2['DispositionSoldAmount']>= 500]
df2 = df2[df2['DispositionSoldAmount'] <= df2['PurchasePrice']]

df2['age']=df2['DispositionDate'] - df2['ModelYear']
df2 = df2.drop(['DispositionDate','ModelYear'],axis=1)

df2 = pd.get_dummies(df2,prefix=['FuelType','EngineConfiguration'])

#garder les donnees engines et type de fuel ou les supprimer:
Xdata = df2.drop('DispositionSoldAmount',axis=1)
Xdata = df2.drop(['DispositionSoldAmount','EngineConfiguration_PH','EngineConfiguration_HI','EngineConfiguration_DE','EngineConfiguration_BI','FuelType_LPG','FuelType_GAS','FuelType_EVC','FuelType_E85','FuelType_DSL','FuelType_CNG'],axis=1)
ydata = df2[['DispositionSoldAmount']]

# TF
COLUMNS = ["DispositionMileage", "PurchasePrice", "Age", "DispositionSoldAmount"]
FEATURES = ["DispositionMileage", "PurchasePrice", "Age"] # X
LABELS = ["DispositionSoldAmount"] # y == predictions

## Convertions des types
feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]			# creation des feature_collumn PAS ENCORE DE DATA juste les labels

print(feature_cols)
## Création de l'estimateur
estimator = tf.estimator.LinearRegressor(feature_columns=feature_cols,model_dir="train")
def get_input_fn( num_epochs=None, n_batch = 128, shuffle=True):
         return tf.estimator.inputs.pandas_input_fn(
         x= Xdata,
         y = ydata,
         batch_size=n_batch,
         num_epochs=num_epochs,
         shuffle=shuffle)

estimator.train(input_fn=get_input_fn(num_epochs=None,n_batch = 128,shuffle=True),steps=1000)
