#Label encoding - Using Category codes approach

import pandas as pd
import numpy as np

bridge_types = ('Arch', 'Beam', 'Truss', 'Cantilever', 'Tied Arch', 'Suspension', 'Cable')

bridge_df = pd.DataFrame(bridge_types, columns = ['Bridge_Types'])

bridge_df['Bridge_Types'] = bridge_df['Bridge_Types'].astype('category')

bridge_df['Bridge_Types_Car'] = bridge_df['Bridge_Types'].cat.codes
bridge_df