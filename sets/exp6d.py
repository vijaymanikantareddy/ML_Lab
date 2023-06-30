#One-Hot Encoding - Using Dummies values approach

import pandas as pd
import numpy as np

bridge_types = ('Arch', 'Beam', 'Truss', 'Cantilever', 'Tied Arch', 'Suspension', 'Cable')
bridge_df = pd.DataFrame(bridge_types, columns=['Bridge_Types'])

bridge_df

dum_df = pd.get_dummies(bridge_df, columns=['Bridge_Types'], prefix=["Type_is"])

dum_df

bridge_df = bridge_df.join(dum_df)
bridge_df