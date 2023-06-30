#Label encoding - Using Sci-kit learn library approach

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

bridge_types = ('Arch', 'Beam', 'Truss', 'Cantilever', 'Tied Arch', 'Suspension', 'Cable')

bridge_df = pd.DataFrame(bridge_types, columns = ['Bridge_Types'])

labelencoder = LabelEncoder()

bridge_df['Bridge_Types_Cat'] = labelencoder.fit_transform(bridge_df['Bridge_Types'])
bridge_df