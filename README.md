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

fig, ax = plt.subplots()
msno.bar(df_rename)

#缺失值為最新一期經濟數據，無法推估，故直接移除

df_total_new = df_rename.dropna()

df_total_new.isna().sum()

df_total_new.shape

df_total_new['Statistical_Period'] = pd.PeriodIndex(df_total_new['Statistical_Period'], freq='Q')

df_total_new.set_index('Statistical_Period', inplace=True)

print(df_total_new.index)

df_total_new.head()

df_total_new.info()

df_total_new.describe().T.sort_values(by='std', ascending=False).style.background_gradient(cmap='GnBu')

sns.heatmap(df_total_new.corr(numeric_only=True), annot=True, square=False, cmap='Blues')

# 移除沒有用處的欄位，讓資料更清楚，'Mid_term_Population (persons)	','GDP','National_Income'
df_total_new.drop(['Mid_term_Population (persons)','GDP','National_Income'], axis=1, inplace=True)

df_total_new.head()

df_total_new.describe().T.sort_values(by='std', ascending=False).style.background_gradient(cmap='GnBu')

sns.heatmap(df_total_new.corr(numeric_only=True), annot=True, square=False, cmap='Blues')

* 由此熱力圖發現
* 經濟成長與進口車呈現高度正相關，與國產車銷量呈中度負相關，與總體汽車銷量呈現低度正相關，表示經濟成長時，人民的收入增加需求會由國產車偏向進口車
* 國產車與進口車存在競爭關係

df_total_new.info()

fig, ax = plt.subplots(figsize=(16, 8), layout='constrained')
plt.plot(df_total_new.index.to_timestamp(), df_total_new['Domestic_sales'], color='royalblue',  label='Domestic_sales')
plt.plot(df_total_new.index.to_timestamp(), df_total_new['Imported_sales'], color='darkorange',  label='Imported_sales')
plt.plot(df_total_new.index.to_timestamp(), df_total_new['Total_Sales'], color='tab:purple', label='Total_Sales')
plt.title( "Taiwan's vehecle sales time series trend (2014-2025)", fontsize=18)
plt.xlabel('Statistical_Period', fontsize=12)
plt.ylabel('Quarterly sales (units)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()
<img width="1611" height="811" alt="台灣總體汽車銷售趨勢圖" src="https://github.com/user-attachments/assets/accdcb85-e264-457f-8226-ba0e54170ac1" />

* 總體車輛銷售數量存在明顯的週期性及有明顯的正相關，數據通常都是同時上升或下降，，不過可以看出進口車銷售數量持續上升，國產車則是持續下降，在2017呈現交叉，且差距有越來越大的趨勢
* 依圖表顯示，總體汽車銷量的起伏大致都在年前獲年後大約在Q1及Q4的位置會達到高峰，與名俗上過年前後會有買氣大致相符

fig, ax1 = plt.subplots(figsize=(18, 8), layout='constrained')

ax1.plot(df_total_new.index.to_timestamp(), df_total_new['Domestic_sales'], color='royalblue', label='Domestic_sales')
ax1.plot(df_total_new.index.to_timestamp(), df_total_new['Imported_sales'], color='darkorange', label='Imported_sales')
ax1.set_xlabel('Year', fontsize=14)
ax1.set_ylabel('Statistical_Period', color='black', fontsize=14)
ax1.tick_params(axis='y', labelcolor='black', labelsize=12)
ax1.legend(loc='upper left', fontsize=12)

ax2 = ax1.twinx() 
ax2.plot(df_total_new.index.to_timestamp(), df_total_new['Income_Per_Capita'], color='tomato', label='Income_Per_Capita')
ax2.set_ylabel('Income_Per_Capita', color='tomato', fontsize=14)
ax2.tick_params(axis='y', labelcolor='tomato', labelsize=12)
ax2.legend(loc='upper right', fontsize=12)


plt.title("Taiwan's vehecle sales time series trend (2014-2025) VS Income_Per_Capita", fontsize=20)
fig.tight_layout() 
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()
<img width="1790" height="790" alt="台灣汽車銷售趨勢與人均收入趨勢圖" src="https://github.com/user-attachments/assets/8eee3815-94f3-4d62-b287-6b0ff78fa8ee" />

* 人均收入與進口車銷售數量有正相關，與國產車銷售數量呈負相關
* 代表收入提升時，台灣人民會更傾向於購買進口車

fig, ax1 = plt.subplots(figsize=(18, 8), layout='constrained')

ax1.plot(df_total_new.index.to_timestamp(), df_total_new['Domestic_sales'], color='royalblue', label='Domestic_sales')
ax1.plot(df_total_new.index.to_timestamp(), df_total_new['Imported_sales'], color='darkorange', label='Imported_sales')
ax1.set_xlabel('Year', fontsize=14)
ax1.set_ylabel('Statistical_Period', color='black', fontsize=14)
ax1.tick_params(axis='y', labelcolor='black', labelsize=12)
ax1.legend(loc='upper left', fontsize=12)

ax2 = ax1.twinx() 
ax2.plot(df_total_new.index.to_timestamp(), df_total_new['Economic_Growth_Rate'], color='tab:green', label='Economic_Growth_Rate')
ax2.set_ylabel('Economic_Growth_Rate', color='tab:green', fontsize=14)
ax2.tick_params(axis='y', labelcolor='tab:green', labelsize=12)
ax2.legend(loc='upper right', fontsize=12)


plt.title("Taiwan's vehecle sales time series trend (2014-2025) VS Economic_Growth_Rate", fontsize=20)
fig.tight_layout() 
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

<img width="1790" height="790" alt="台灣汽車銷售趨勢與經濟成長率趨勢圖" src="https://github.com/user-attachments/assets/281dadf7-4840-4043-86cf-be12733e43d6" />

fig, ax1 = plt.subplots(figsize=(18, 8), layout='constrained')

ax1.plot(df_total_new.index.to_timestamp(), df_total_new['Domestic_sales'], color='royalblue', label='Domestic_sales')
ax1.plot(df_total_new.index.to_timestamp(), df_total_new['Imported_sales'], color='darkorange', label='Imported_sales')
ax1.plot(df_total_new.index.to_timestamp(), df_total_new['Total_Sales'], color='tab:purple', label='Total_Sales ')
ax1.set_xlabel('Year', fontsize=14)
ax1.set_ylabel('Statistical_Period', color='black', fontsize=14)
ax1.tick_params(axis='y', labelcolor='black', labelsize=12)
ax1.legend(loc='upper left', fontsize=12)

ax2 = ax1.twinx() 
ax2.plot(df_total_new.index.to_timestamp(), df_total_new['Economic_Growth_Rate'], color='tab:green', label='Economic_Growth_Rate')
ax2.set_ylabel('Economic_Growth_Rate', color='tab:green', fontsize=14)
ax2.tick_params(axis='y', labelcolor='tab:green', labelsize=12)
ax2.legend(loc='upper right', fontsize=12)


plt.title("Taiwan's vehecle sales time series trend (2014-2025) VS Economic_Growth_Rate", fontsize=20)
fig.tight_layout() 
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

<img width="1790" height="790" alt="台灣總體汽車銷售趨勢與經濟成長率趨勢圖" src="https://github.com/user-attachments/assets/82d7011d-71ed-4604-b8ab-7bc78d911b04" />

* 經濟成長率與整體汽車銷售及進口車銷售數量有高度正相關，與國產車銷售數量則是低度相關。
* 2023年，台灣整體車市銷量顯著上升，經濟成長率卻大幅下滑。此現象的原因在於，車市受惠於疫後供應鏈回穩，先前因晶片短缺的訂單得以交付，帶動銷量激增；然而，整體經濟卻因全球需求普遍疲軟，導致台灣出口大幅減少，成長動能因而受挫。

fig, ax1 = plt.subplots(figsize=(18, 8), layout='constrained')

ax1.plot(df_total_new.index.to_timestamp(), df_total_new['Domestic_sales'], color='royalblue', label='Domestic_sales')
ax1.plot(df_total_new.index.to_timestamp(), df_total_new['Imported_sales'], color='darkorange', label='Imported_sales')
ax1.set_xlabel('Year', fontsize=14)
ax1.set_ylabel('Statistical_Period', color='black', fontsize=14)
ax1.tick_params(axis='y', labelcolor='black', labelsize=12)
ax1.legend(loc='upper left', fontsize=12)

ax2 = ax1.twinx() 
ax2.plot(df_total_new.index.to_timestamp(), df_total_new['Exchange_Rate'], color='chocolate', label='Exchange_Rate')
ax2.set_ylabel('Exchange_Rate', color='chocolate', fontsize=14)
ax2.tick_params(axis='y', labelcolor='chocolate', labelsize=12)
ax2.legend(loc='upper right', fontsize=12)


plt.title("Taiwan's vehecle sales time series trend (2014-2025) VS Exchange_Rate", fontsize=20)
fig.tight_layout() 
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

<img width="1790" height="790" alt="台灣汽車銷售趨勢與匯率趨勢圖" src="https://github.com/user-attachments/assets/9fc568fe-a081-4fca-be36-8ce5e696af88" />

* 匯率與進口車銷售數量有低度相關，並沒有顯著的關聯；與國產車銷售數量則無相關

```

