import pandas as pd

df = pd.read_excel("D:/stock/_data/news/NewsResult_20251201-20251231.xlsx", index_col=0)
# df = df.drop(['언론사', 'URL', '분석제외 여부'], axis=1)
print(df.columns)
exit()
keyword = df['키워드']
institute = df['기관']
features = df['특성추출(가중치순 상위 50개)']
person = df['인물']
place = df['위치']
print(df['통합 분류1'].isna().sum())        #0
print(df['통합 분류2'].isna().sum())        #2952
print(df['통합 분류3'].isna().sum())        #6512
print(df['사건/사고 분류1'].isna().sum())   #11910
print(df['사건/사고 분류2'].isna().sum())   #14075
print(df['사건/사고 분류3'].isna().sum())   #14529
print(df['인물'].isna().sum())             #4056
print(df['위치'].isna().sum())             #2358

keyword_sep = keyword.astype(str).str.split(',', expand=True)       #2441
institute_sep = institute.astype(str).str.split(',', expand=True)   #137
features_sep = features.astype(str).str.split(',', expand=True)     #50
person_sep = person.astype(str).str.split(',', expand=True)         #460
place_sep = place.astype(str).str.split(',', expand=True)           #172

print(keyword_sep.head())
print(institute_sep.head())
print(features_sep.head())
print(person_sep.head())
print(place_sep.head())


