# freeCodeCamp - Linear Regression Health Costs Calculator

Solução para o projeto **Linear Regression Health Costs Calculator** da certificação **Machine Learning with Python** do freeCodeCamp.

## Objetivo

Criar um modelo de regressão para prever custos médicos (`expenses`) usando o dataset `insurance.csv`.

O teste oficial exige:

```python
mae < 3500
```

Ou seja, o erro absoluto médio precisa ficar abaixo de 3500.

## Estratégia usada

A solução usa:

- Pandas para leitura e preparação dos dados;
- `pd.get_dummies()` para transformar variáveis categóricas em numéricas;
- TensorFlow/Keras;
- camada `Normalization`;
- rede neural densa para regressão;
- função de perda `mae`;
- métricas `mae` e `mse`;
- `EarlyStopping` para evitar overfitting.

## Arquivos

- `linear_regression_health_costs_solution.py`: versão script.
- `README.md`: explicação do projeto.


## Dataset

O dataset é baixado automaticamente de:

```python
https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv
```

## Observação

O dataset não está incluído porque o notebook baixa o arquivo diretamente pelo link oficial do freeCodeCamp.
