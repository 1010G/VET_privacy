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

df2['Age']=df2['DispositionDate'] - df2['ModelYear']
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

def afficherDonnees(): #fonction temporaire pour afficher les data
    print("\n   feature_cols")
    print(feature_cols)
    print("\n")
    print("\nXdata")
    print(Xdata)
    print("\nXdata type")
    print(type(Xdata))
    print("\nydata")
    print(ydata)
    print("\nydata type")
    print(type(ydata))
    print(Xdata.info())
    print("\n")
    print(ydata.info())

#afficherDonnees() # a commenter / decommenter pour afficher les data

## Création de l'estimateur
estimator = tf.estimator.LinearRegressor(feature_columns=feature_cols,model_dir="train")

## Ajout de la data dans l'estimateur pour le mode Train
def get_input_fn_Train( num_epochs=None, n_batch = 128, shuffle=True):
         return tf.estimator.inputs.pandas_input_fn(
            x=Xdata,                                                           # feature data type: pandas.DataFrame
            y=pd.Series(ydata["DispositionSoldAmount"]),                       # label data type pandas.Series
            batch_size=n_batch,                                                # batch 128 by default
            num_epochs=num_epochs,                                             # number of epoch
            shuffle=shuffle)                                                   # shuffle or not, yes by default

## Ajout de la data dans l'estimateur pour le mode Eval
def get_input_fn_Eval():
    return tf.estimator.inputs.pandas_input_fn(
       x=Xdata,                                                           # feature data type: pandas.DataFrame
       y=pd.Series(ydata["DispositionSoldAmount"]),                       # label data type pandas.Series
       shuffle=True)


## Execution du modéle
estimator.train(input_fn=get_input_fn_Train(num_epochs=None,n_batch = 128,shuffle=True),steps=1000000)                    # Entrainement du modéle
result = estimator.evaluate(get_input_fn_Eval())                                                                          # Evaluation du modéle

#tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

for key,value in sorted(result.items()):
  print('%s: %s' % (key, value))
