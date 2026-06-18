import pandas as pd

df = pd.read_excel(
    r'G:\Mi unidad\NEUROCIENCIA\AFH\EXPERIMENTO\REGISTERED REPORT\TSUCHIYA\S1_Data.xlsx',
    sheet_name='Sheet1',
    usecols=['Name', 'channel', 'class_D_WxA', 'class_E3_WxSE']
)

df_ch1 = df[df['channel'] == 1]

ac = ['AC_29', 'AC_30', 'AC_31', 'AC_32', 'AC_33', 'AC_34']
r = df_ch1[df_ch1['Name'].isin(ac)][['Name', 'class_D_WxA', 'class_E3_WxSE']].copy()
r['delta'] = r['class_D_WxA'] - r['class_E3_WxSE']

print(r.to_string(index=False))
print(f'\nPositivos (anestesia > sueño): {(r["delta"]>0).sum()}')
print(f'Negativos (sueño > anestesia): {(r["delta"]<0).sum()}')
print(f'Delta promedio: {r["delta"].mean():.4f}')