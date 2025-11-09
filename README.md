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

df_total_new['quarter'] = df_total_new.index.quarter

df_total_new['quarter_str'] = 'Q' + df_total_new['quarter'].astype(str)

df_total_new.head()

df_total_new.info()

df_total_new.describe().T.sort_values(by='std', ascending=False).style.background_gradient(cmap='GnBu')

sns.heatmap(df_total_new.corr(numeric_only=True), annot=True, square=False, cmap='Blues')

# 移除沒有用處的欄位，讓資料更清楚，'Mid_term_Population (persons)	','GDP','National_Income',
df_total_new.drop(['Mid_term_Population (persons)','GDP','National_Income'], axis=1, inplace=True)

df_total_new.head()

df_total_new.describe().T.sort_values(by='std', ascending=False).style.background_gradient(cmap='GnBu')

sns.heatmap(df_total_new.corr(numeric_only=True), annot=True, square=False, cmap='Blues')

* 由此熱力圖發現
* 經濟成長與進口車呈現高度正相關，與國產車銷量呈中度負相關，與總體汽車銷量呈現低度正相關，表示經濟成長時，人民的收入增加需求會由國產車偏向進口車
* 國產車與進口車存在競爭關係

df_total_new.info()

#計算個季度的銷售數量平均數
quarterly_avg_sales = df_total_new.groupby('quarter_str')[['Domestic_sales', 'Imported_sales', 'Total_Sales']].mean()

quarterly_avg_sales = quarterly_avg_sales.round(0)

print("\n--- 各季度平均銷量統計 ---")
print(quarterly_avg_sales)

plt.figure(figsize=(12, 8), layout='constrained')
sns.boxplot(
    x='quarter_str',
    y='Total_Sales',
    data=df_total_new,
    palette='viridis'
)

plt.title('Box Plot of Total Sales Distribution by Quarter', fontsize=20)
plt.xlabel('quarter_str', fontsize=14)
plt.ylabel('Total_Sales', fontsize=14)
plt.show()

<img width="1211" height="811" alt="箱型圖-車輛銷售數量季度" src="https://github.com/user-attachments/assets/7f27d9b6-d968-44fb-8c58-895afc9eee2b" />

* 可以觀察出Q1及Q4是車市的旺季，Q2則是相對不穩定，Q3是淡季

#進口車、國產車及總體汽車銷量的時間序列圖
fig, ax = plt.subplots(figsize=(16, 8), layout='constrained')
plt.plot(df_total_new.index.to_timestamp(), df_total_new['Domestic_sales'], color='royalblue',  label='Domestic_sales')
plt.plot(df_total_new.index.to_timestamp(), df_total_new['Imported_sales'], color='darkorange',  label='Imported_sales')
plt.plot(df_total_new.index.to_timestamp(), df_total_new['Total_Sales'], color='tab:purple', label='Total_Sales')
plt.title( "Taiwan's vehecle sales time series trend (2014-2025)", fontsize=18)
plt.xlabel('Statistical_Period', fontsize=12)
plt.ylabel('Quarterly sales (units)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

<img width="1611" height="811" alt="台灣總體汽車銷售趨勢圖" src="https://github.com/user-attachments/assets/8056f19c-3010-4259-9d67-82f5b66b52a7" />

* 總體車輛銷售數量存在明顯的週期性及有明顯的正相關，數據通常都是同時上升或下降，，不過可以看出進口車銷售數量持續上升，國產車則是持續下降，在2017呈現交叉，且差距有越來越大的趨勢
* 依圖表顯示，國產車及進口車的起伏大致都在年前獲年後大約在Q1及Q4的位置會達到高峰，與季度及汽車銷售數量箱型圖一致

#進口車、國產車及人均收入的時間序列圖
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

<img width="1790" height="790" alt="台灣汽車銷售趨勢與人均收入趨勢圖" src="https://github.com/user-attachments/assets/28699952-6377-4e7a-bd95-76a5456f93e9" />

* 人均收入與進口車銷售數量有正相關，與國產車銷售數量呈負相關
* 代表收入提升時，台灣人民會更傾向於購買進口車

#進口車、國產車及經濟成長率的時間序列圖
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

<img width="1790" height="790" alt="台灣汽車銷售趨勢與經濟成長率趨勢圖" src="https://github.com/user-attachments/assets/e4a20b68-ceec-49c7-8dfd-a4e08ca67696" />

##進口車、國產車、總體汽車銷售及經濟成長率的時間序列圖
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

* 經濟成長率與整體汽車銷售及進口車銷售數量有高度正相關，與國產車銷售數量則是低度相關。
* 2023年，台灣整體車市銷量顯著上升，經濟成長率卻大幅下滑。此現象的原因在於，車市受惠於疫後供應鏈回穩，先前因晶片短缺的訂單得以交付，帶動銷量激增；然而，整體經濟卻因全球需求普遍疲軟，導致台灣出口大幅減少，成長動能因而受挫。

<img width="1790" height="790" alt="台灣總體汽車銷售趨勢與經濟成長率趨勢圖" src="https://github.com/user-attachments/assets/6543ec28-bafc-4d2c-96a7-00f505646cb2" />

#進口車、國產車及匯率的時間序列圖

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

<img width="1211" height="811" alt="箱型圖-車輛銷售數量季度" src="https://github.com/user-attachments/assets/f70d70ea-78cf-4c88-8b37-86125c04c61e" />

* 匯率與進口車銷售數量有低度相關，並沒有顯著的關聯；與國產車銷售數量則無相關

# ARIMA

pip install "numpy<2.0"

!pip install pmdarima

from pmdarima import auto_arima

# 切分資料
df_train = df_total_new[df_total_new.index < '2025Q2']
df_test = df_total_new[df_total_new.index >= '2025Q2']

print(f'訓練集資料期間: {df_train.index.min()} 到 {df_train.index.max()}')
print(f'測試集期間: {df_test.index.min()} 到 {df_test.index.max()}')

# 準備數據
features = ['Exchange_Rate', 'Income_Per_Capita', 'Economic_Growth_Rate']
X_test = df_test[features]
X_train = df_train[features]
y_train = df_train['Total_Sales']
y_test = df_test['Total_Sales']

# 建立模型
arima_model = auto_arima(y_train, exogenous=X_train, trace=False, error_action='ignore', suppress_warnings=True, auppress_warings=True, seasonal=True, m=4)

# 預測與評估
arima_pred = arima_model.predict(n_periods=len(X_test), exogenous=X_test)

from sklearn.metrics import mean_absolute_error
mae_arima = mean_absolute_error(y_test, arima_pred)
mape_arima = np.mean(np.abs((y_test - arima_pred) / y_test)) *100
print(f'MAE: {mae_arima:.2f}, MAPE: {mape_arima:.2f}%')

# 計算ARIMA缺口
arima_gap = y_test - arima_pred

print('\n' + '='*50)
print('        ARIMA預測分析')

for period, actual, pred, gap in zip(y_test.index, y_test, arima_pred, arima_gap):
    print(f'季度: {period}')
    print(f'實際銷售量: {actual:.0f}')
    print(f'ARIMA預測銷量: {pred:.0f}')
    print(f'ARIMA預測缺口: {gap:.0f}輛')
print('='*50)


# Prophet

pip uninstall prophet holidays -y

pip install holidays==0.41

pip install prophet

from prophet import Prophet

# 初始化模型
fbp_model = Prophet(seasonality_mode='multiplicative')

for feature in features:
    fbp_model.add_regressor(feature)

# 準備prophet格式數據
fbp_train = df_train.reset_index().rename(columns={'Statistical_Period': 'ds', 'Total_Sales': 'y'})
fbp_test = df_test.reset_index().rename(columns={'Statistical_Period': 'ds', 'Total_Sales': 'y'})

fbp_train['ds'] = fbp_train['ds'].dt.to_timestamp(how='end')
fbp_test['ds'] = fbp_test['ds'].dt.to_timestamp(how='end')

# 訓練模型
fbp_model.fit(fbp_train[['ds','y'] + features])

# 預測與評估
fbp_pred = fbp_model.predict(fbp_test[['ds'] + features])
mae_fbp = mean_absolute_error(fbp_test['y'], fbp_pred['yhat'])
mape_fbp = np.mean(np.abs((fbp_test['y'] - fbp_pred['yhat']) / fbp_test['y'])) * 100
print(f'MAE: {mae_fbp:.2f}, MAPE: {mape_fbp:.2f}%')

# 計算Prophet缺口
fbp_gap = pd.merge(fbp_test[['ds', 'y']], fbp_pred[['ds', 'yhat']], on='ds')
print(fbp_gap)

fbp_gap['gap'] = fbp_gap['y'] - fbp_gap['yhat']
print('\n' + '='*50)
print('            Prophet預測分析')
print('='*50)

for index, row in fbp_gap.iterrows():
    period = pd.Period(row['ds'], freq='Q')
    print(f'季度: {period}')
    print(f'實際銷售量: {row['y']:.0f}')
    print(f'Prophet預測銷售量: {row['yhat']:.0f}')
    print(f'Prophet預測缺口: {row['gap']:.0f}')
print('='*50)

# LightGBM

import lightgbm as lgb

from sklearn.model_selection import GridSearchCV

lgbm_features = features + ['quarter']

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5, 7]
}

lgbm = lgb.LGBMRegressor(random_state=42)

# 超參數調整
grid_search = GridSearchCV(estimator=lgbm, param_grid = param_grid, scoring = 'neg_mean_absolute_error', cv=3, verbose=1, n_jobs=-1)

grid_search.fit(df_train[lgbm_features], df_train['Total_Sales'])

print(f"最佳參數組合: {grid_search.best_params_}")

# 預測及評估

lgbmr_pred = grid_search.best_estimator_.predict(df_test[lgbm_features])
mae_lgbm = mean_absolute_error(y_test, lgbmr_pred)
mape_lgbm = np.mean(np.abs((y_test - lgbmr_pred) / y_test))*100
print(f'MAE: {mae_lgbm:.2f}, MAPE: {mape_lgbm:.2f}%')

# 計算LightGBM的缺口值
lgbm_gap = y_test - lgbmr_pred

print('\n' + '='*50)
print('          LightGBM 預測分析 ')
print('='*50)

for period, actual, pred, gap in zip(y_test.index, y_test, lgbmr_pred, lgbm_gap):
    print(f'季度: {period}')
    print(f'實際銷售量: {actual:.0f}')
    print(f'LGBM預測銷售量: {pred:.0f}')
    print(f'LGBM預測缺口: {gap:.0f}')
print('='*50)

import plotly.graph_objects as go

# 繪製測試集的綜合比較圖
fig_compare = go.Figure()
fig_compare.add_trace(go.Bar(x=df_test.index.to_timestamp(how='end'), y=y_test, name='實際銷量', marker_color='black'))
fig_compare.add_trace(go.Bar(x=df_test.index.to_timestamp(how='end'), y=arima_pred, name='ARIMA預測'))
fig_compare.add_trace(go.Bar(x=df_test.index.to_timestamp(how='end'), y=fbp_pred['yhat'], name='Prophet預測'))
fig_compare.add_trace(go.Bar(x=df_test.index.to_timestamp(how='end'), y=lgbmr_pred, name='LGBM預測'))
fig_compare.update_layout(title='綜合比較圖', xaxis_title='季度', yaxis_title='總銷量(輛)', barmode='group')
fig_compare.show()

# 計算平均值
#arima_gap = y_test - arima_pred
#lgbm_gap = y_test - lgbmr_pred
fbp_pred_series = fbp_pred['yhat'].copy()
fbp_pred_series.index = y_test.index
fbp_gap_series = (y_test - fbp_pred_series)

avg_sales_forecast = (arima_pred + lgbmr_pred + fbp_pred_series) / 3

avg_gap_series = (arima_gap + fbp_gap_series + lgbm_gap) / 3

avg_gap_total_mean = avg_gap_series.mean()

final_summary_df = pd.DataFrame({
    '實際銷量': y_test,
    'ARIMA預測': arima_pred,
    'Prophet預測': fbp_pred_series,
    'LGBM預測': lgbmr_pred,
    '共識銷量 (Avg)': avg_sales_forecast,
    '共識缺口 (Gap)': avg_gap_series
})

print('\n' + '='*60)
print('                  最終綜合分析總表')
print('='*60)
print(final_summary_df.round(0).to_markdown(index=True))

print(f'\n' + '='*60)
print(f'                         最終綜合結論')
print('='*60)
print(f'''本研究整合了 ARIMA、Prophet 及 LightGBM 三種模型的預測結果，以產生一個更穩健的共識預測，
在對等關稅後，市場實際銷售量皆低於預期的水準，根據三個模型的預測值，可以發現2025Q2的對等關稅確實對車市造成明顯的負面衝擊，
消費者預期政府對因應美國的關稅壓力下調降對美進口的車輛進口貨物稅，預期車輛售價會下降，短期關稅戰未明朗前會呈現觀望的態度，
所以即使台幣升值且人均所得及經濟成長率都呈現大幅成長的情況下,總體車輛銷售額卻是呈現下滑趨勢，
依造三模型的平均預估值可以得知，市場總銷售量每季減少{abs(int(avg_gap_total_mean))} 輛。''')
print('='*60)
```

