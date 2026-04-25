#!/usr/bin/env python3
import os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

BASE = os.path.join(os.path.dirname(__file__), '..')
DATA = os.path.join(BASE, 'data', 'apartment_final_v6_dong.csv')
OUT = os.path.join(BASE, 'results', 'log_ols_robustness.json')

FEATURES = ['전용면적', '층', '건물연령', '강남구분',
            '초등학교수', '중학교수', '고등학교수',
            'CCTV수', '백화점수', '지하철역수',
            '공원수', '도서관수', '학원수', '어린이집수',
            '기준금리', 'CD금리', '소비자물가지수', 'M2']
TARGET = '거래금액'


def main():
    df = pd.read_csv(DATA)
    X = df[FEATURES].values
    y = np.log(df[TARGET].values)

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    r2 = r2_score(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mae = float(mean_absolute_error(y_test, pred))

    coef = pd.DataFrame({'변수': FEATURES, '계수': model.coef_}).sort_values('계수', ascending=False)

    result = {
        'target': 'log(거래금액)',
        'split': {'train': int(len(X_train)), 'val': int(len(X_val)), 'test': int(len(X_test))},
        'metrics': {'R2': float(r2), 'RMSE_log': rmse, 'MAE_log': mae},
        'top_positive': coef.head(8).to_dict(orient='records'),
        'top_negative': coef.tail(8).to_dict(orient='records')
    }

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
