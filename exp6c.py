#One-Hot encoding - Using Sci-kit learn library approach

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

bridge_types = ('Arch', 'Beam', 'Truss', 'Cantilever', 'Tied Arch', 'Suspension', 'Cable')
bridge_df = pd.DataFrame(bridge_types, columns=['Bridge_Types'])

enc = OneHotEncoder(handle_unknown='ignore')

enc_df = pd.DataFrame(enc.fit_transform(bridge_df['Bridge_Types_Cat']).toarray())

bridge_df = bridge_df.join(enc_df)

bridge_df