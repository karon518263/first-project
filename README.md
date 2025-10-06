```python
# 對等關稅對台汽車零售業的衝擊
* 使用參數
* Statistical_Period:以季度為單位計算
* Mid_term_Population (persons):台灣的人口數
* Exchange_Rate:台幣對美金的匯率
* Economic_Growth_Rate:經濟成長率
* GDP:國內生產毛額，以美金為單位
* GDP_Per_Capita:每人平均國內生產毛額，以美金為單位
* National_Income:國民所得，以美金為單位
* Income_Per_Capita:每人平均國民所得，以美金為單位
* Domestic_vehicles:國產汽車銷量
* Imported_vehicles:進口汽車銷量

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#讀取資料
df_econ = pd.read_csv('/content/2014-2025 Economic Indicators.csv')
df_car = pd.read_csv('/content/2014-2025 Quarterly car sales.csv')

df_econ.head()

df_car.head()

df_total = pd.concat([df_econ, df_car[['Domestic_vehicles', 'Imported_vehicles']]], axis=1)

df_total.head(20)

df_total.info()

print(f'資料比數:{df_total.shape[0]}')
print(f'資料欄位數:{df_total.shape[1]}')
print(f'資料缺失數:{sum(df_total.isna().sum())}')

df_total.head()

df_rename = df_total.rename(columns={'Domestic_vehicles':'Domestic_sales','Imported_vehicles':'Imported_sales'})

df_rename['Total_Sales'] = df_rename['Domestic_sales'] + df_rename['Imported_sales']

df_rename.info()

df_rename.isna().sum()

#視覺化處理缺失值
import missingno as msno
msno.matrix(df_rename)
<img width="2027" height="1067" alt="視覺化缺失值" src="https://github.com/user-attachments/assets/a2145ea6-7032-48ad-8ef1-b95359aa1536" />




df_total_new = df_rename.dropna()

df_total_new.isna().sum()

df_total_new.shape

df_total_new['Statistical_Period'] = pd.PeriodIndex(df_total_new['Statistical_Period'], freq='Q')

df_total_new.set_index('Statistical_Period', inplace=True)

print(df_total_new.index)

df_total_new.head()

df_total_new.info()

df_total_new.describe().T.sort_values(by='std', ascending=False).style.background_gradient(cmap='GnBu')

fig, ax = plt.subplots(figsize=(16, 8), layout='constrained')
plt.plot(df_total_new.index.to_timestamp(), df_total_new['Domestic_sales'],  label='Domestic_sales')
plt.plot(df_total_new.index.to_timestamp(), df_total_new['Imported_sales'],  label='Imported_sales')
plt.plot(df_total_new.index.to_timestamp(), df_total_new['Total_Sales'], label='Total_Sales')
plt.title( "Taiwan's vehecle sales time series trend (2014-2025)", fontsize=18)
plt.xlabel('Statistical_Period', fontsize=12)
plt.ylabel('Quarterly sales (units)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()
<img width="1611" height="811" alt="Taiwan's vehecle sales time series trend (2014-2025)" src="https://github.com/user-attachments/assets/6074901f-aa57-4bba-8f05-f1a7f030afb3" />


sns.heatmap(df_total_new.corr(numeric_only=True), annot=True, square=False, cmap='Blues')
<img width="727" height="619" alt="熱力圖" src="https://github.com/user-attachments/assets/727e3526-82fb-47c5-9cdd-fd09f6224405" />

* 由此熱力圖發現
* 經濟成長與進口車呈現高度正相關，與國產車銷量呈中度負相關，與總體汽車銷量呈現低度正相關，表示經濟成長時，人民的收入增加需求會由國產車偏向進口車
* 國產車與進口車存在競爭關係

```

