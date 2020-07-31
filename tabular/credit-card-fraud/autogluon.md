# AutoML

```{.python .input}
!kaggle d download mlg-ulb/creditcardfraud -p ../input/
!cd ../input/; unzip creditcardfraud.zip -d creditcardfraud
```

```{.python .input}
import pandas as pd 
from autogluon import TabularPrediction as task

data = pd.read_csv('../input/creditcardfraud/creditcard.csv')
data.head()
```

```{.python .input}
n = int(len(data)*0.6)
train, test = data.iloc[:n], data.iloc[n:]
```

```{.python .input}
model = task.fit(train_data=task.Dataset(train), eval_metric='roc_auc', label='Class', time_limits=60)
```

```{.python .input}
model.evaluate(test)
```
