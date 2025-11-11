import pandas as pd

df = pd.read_csv("./evaluation_details_think.csv", encoding="utf-8")

df.to_csv("./evaluation_details_think.csv", index=False, encoding="utf-8-sig")