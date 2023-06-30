# Remove Duplicates
import pandas as pd
data = {"A": ["TeamA", "TeamB", "TeamB", "TeamC", "TeamA"], "B": [
 50, 40, 40, 30, 50], "C": [True, False, False, False, True]}
df = pd.DataFrame(data)
print(df)
print(df.drop_duplicates())

# Output:
#  A B C
# 0 TeamA 50 True
# 1 TeamB 40 False
# 2 TeamB 40 False
# 3 TeamC 30 False
# 4 TeamA 50 True
#  A B C
# 0 TeamA 50 True
# 1 TeamB 40 False
# 3 TeamC 30 False
