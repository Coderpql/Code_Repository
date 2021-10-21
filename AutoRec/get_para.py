import pandas as pd

col_names = ['h', 'lambda', 'mae']
Domain_name = ['Cell_Phones_and_Accessories', 'Industrial_and_Scientific', 'Software']
result = pd.DataFrame(columns=col_names, index=Domain_name)

for domain_name in Domain_name:
    df_mae = pd.read_csv('data/amazon/correlated/' + domain_name + '/' + 'log/k_fold_mae.csv')
    optimal_para = df_mae.loc[df_mae['mae'] == df_mae['mae'].min()].values.reshape(-1)
    print(optimal_para)
    result.loc[domain_name, 'h'] = optimal_para[0]
    result.loc[domain_name, 'lambda'] = optimal_para[1]
    result.loc[domain_name, 'mae'] = optimal_para[2]

result.index.name = 'domain_name'
result.to_csv('data/amazon/correlated/optimal_parameter.csv')