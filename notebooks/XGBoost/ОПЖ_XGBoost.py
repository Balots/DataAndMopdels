import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib
import os

output_dir = 'notebooks/XGBoost'

def standardize_region_names(df):
    """Стандартизация названий регионов"""
    df_clean = df.copy()
    
    # Создаем словарь замен
    replacements = {
        'Ненецкий авт.округ': 'Ненецкий автономный округ',
        'Hенецкий авт.округ': 'Ненецкий автономный округ',
        '  Ненецкий автономный округ': 'Ненецкий автономный округ',
        
        'Ямало-Ненецкий авт.округ': 'Ямало-Ненецкий автономный округ',
        'Ямало-Hенецкий авт.округ': 'Ямало-Ненецкий автономный округ',
        '  Ямало-Ненецкий автономный округ': 'Ямало-Ненецкий автономный округ',
        
        'Ханты-Мансийский авт.округ-Югра': 'Ханты-Мансийский автономный округ - Югра',
        '  Ханты-Мансийский автономный округ - Югра': 'Ханты-Мансийский автономный округ - Югра',
        
        'Республика Татарстан(Татарстан)': 'Республика Татарстан',
        'Чувашская Республика(Чувашия)': 'Чувашская Республика',
        'Республика Северная Осетия- Алания': 'Республика Северная Осетия-Алания',
        
        'Oмская область': 'Омская область',
        'Hижегородская область': 'Нижегородская область',
        
        'г. Севастополь': 'г.Севастополь',
        'г.Москва': 'г.Москва',
        'г.Санкт-Петербург': 'г.Санкт-Петербург',
        
        'Чукотский авт.округ': 'Чукотский автономный округ',
        'Чукотский автономный округ': 'Чукотский автономный округ'
    }
    
    # Применяем замены
    df_clean['Регион'] = df_clean['Регион'].replace(replacements)
    df_clean['Регион'] = df_clean['Регион'].str.strip()
    
    return df_clean

class OPJXGBoostForecaster:
    def __init__(self, random_state=42):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_fitted = False
        self.last_known_data = None
        self.random_state = random_state
        self.first_year = None
        
    def clean_numeric_columns(self, df):
        """
        Очистка числовых колонок
        """
        df = df.copy()
        
        numeric_columns = [col for col in df.columns if col not in ['Регион', 'Год', 'ОПЖ']]
        
        for col in numeric_columns:
            if df[col].dtype == 'object':
                df[col] = (df[col]
                          .astype(str)
                          .str.replace('\xa0', '')
                          .str.replace(' ', '')
                          .str.replace(',', '.')
                          .str.replace('−', '-')
                          .str.replace('–', '-')
                          )
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'ОПЖ' in df.columns:
            df['ОПЖ'] = pd.to_numeric(df['ОПЖ'], errors='coerce')
        
        return df
        
    def prepare_features(self, df, is_training=True):
        """
        Подготовка признаков для ОПЖ с сохранением данных с 2014 года
        """
        df = df.copy().sort_values(['Регион', 'Год'])
        
        # Определяем первый год для расчета тренда
        if self.first_year is None:
            self.first_year = df['Год'].min()
        
        print(f"Исходные данные: {df['Год'].min()}-{df['Год'].max()}, {len(df)} строк")
        
        # Создание лаговых признаков
        if is_training or df['Год'].min() < 2024:
            df['lag1_ОПЖ'] = df.groupby('Регион')['ОПЖ'].shift(1)
            df['lag2_ОПЖ'] = df.groupby('Регион')['ОПЖ'].shift(2)
            df['lag1_Число умерших'] = df.groupby('Регион')['Число умерших'].shift(1)
            df['lag1_Младенческая смертность коэф'] = df.groupby('Регион')['Младенческая смертность коэф'].shift(1)
            df['lag1_Общая численность инвалидов'] = df.groupby('Регион')['Общая численность инвалидов'].shift(1)
            
            # Скользящие средние с min_periods=1 для сохранения данных
            df['ОПЖ_MA2'] = df.groupby('Регион')['ОПЖ'].transform(lambda x: x.rolling(2, min_periods=1).mean())
            df['ОПЖ_MA3'] = df.groupby('Регион')['ОПЖ'].transform(lambda x: x.rolling(3, min_periods=1).mean())
        
        # Создание тренда времени
        df['year_trend'] = df['Год'] - self.first_year
        df['год_от_начала'] = df['Год'] - self.first_year
        
        # Создание производных медицинских и социальных показателей
        df['Врачей_на_10k'] = df['Численность врачей всех специальностей'] / df['Численность населения'] * 10000
        df['Умерших_на_1000'] = df['Число умерших'] / df['Численность населения'] * 1000
        df['Инвалидов_на_1000'] = df['Общая численность инвалидов'] / df['Численность населения'] * 1000
        df['Преступлений_на_1000'] = df['Кол-во преступлений'] / df['Численность населения'] * 1000
        df['Браков_на_1000'] = df['Браков'] / df['Численность населения'] * 1000
        df['Разводов_на_1000'] = df['Разводов'] / df['Численность населения'] * 1000
        
        # Индексы развития
        df['Индекс_здравоохранения'] = (
            df['Врачей_на_10k'] + 
            df['Число больничных организаций на конец отчетного года'] + 
            df['Число санаторно-курортных организаций']
        ) / 3
        
        df['Социальный_индекс'] = (
            df['Средняя ЗП'] / df['Величина прожиточного минимума'] - 
            (df['Уровень бедности'] / 100)
        )
        
        # Относительные изменения
        df['изменение_населения'] = df.groupby('Регион')['Численность населения'].pct_change()
        df['изменение_ВРП'] = df.groupby('Регион')['Валовой региональный продукт на душу населения (ОКВЭД 2)'].pct_change()
        
        if is_training:
            print(f"Данные до обработки NaN: {len(df)} строк")
            
            # Заполняем числовые колонки медианами по регионам
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in ['Год']:
                    df[col] = df.groupby('Регион')[col].transform(
                        lambda x: x.fillna(x.median()) if not x.isnull().all() else x
                    )
            
            # Если все еще есть NaN, заполняем общими медианами
            df = df.fillna(df.median(numeric_only=True))
            
            print(f"Данные после обработки NaN: {len(df)} строк")
            print(f"Годы после обработки: {df['Год'].min()}-{df['Год'].max()}")
        
        return df
    
    def train_test_split_temporal(self, df, test_years=[2021, 2022, 2023]):
        """
        Разделение на train/test по времени
        """
        train_mask = ~df['Год'].isin(test_years)
        test_mask = df['Год'].isin(test_years)
        
        X_train = df[train_mask][self.feature_names]
        X_test = df[test_mask][self.feature_names]
        y_train = df[train_mask]['ОПЖ']
        y_test = df[test_mask]['ОПЖ']
        
        return X_train, X_test, y_train, y_test, train_mask, test_mask
    
    def fit(self, df):
        """
        Обучение XGBoost модели для ОПЖ на расширенных данных
        """
        # Очистка и подготовка данных
        df_clean = self.clean_numeric_columns(df)
        df_processed = self.prepare_features(df_clean, is_training=True)
        
        print(f"Данные за период: {df_processed['Год'].min()}-{df_processed['Год'].max()}")
        
        # Сохраняем последние известные данные для прогноза
        self.last_known_data = df_processed[df_processed['Год'] == 2023].copy()
        
        # Определение признаков для ОПЖ
        self.feature_names = [
            # Базовые демографические
            'Численность населения', 
            'Число умерших', 
            'Общая численность инвалидов',
            'Браков',
            'Разводов',
            
            # Медицинские факторы
            'Младенческая смертность коэф', 
            'Численность врачей всех специальностей', 
            'Число больничных организаций на конец отчетного года', 
            'Число санаторно-курортных организаций',
            
            # Социально-экономические
            'Валовой региональный продукт на душу населения (ОКВЭД 2)',
            'Величина прожиточного минимума', 
            'Уровень бедности', 
            'Средняя ЗП',
            'Кол-во преступлений',
            
            # Лаговые признаки
            'lag1_ОПЖ', 
            'lag2_ОПЖ',
            'lag1_Число умерших', 
            'lag1_Младенческая смертность коэф', 
            'lag1_Общая численность инвалидов',
            
            # Скользящие средние
            'ОПЖ_MA2',
            'ОПЖ_MA3',
            
            # Тренд и время
            'year_trend',
            'год_от_начала',
            
            # Производные показатели
            'Врачей_на_10k', 
            'Умерших_на_1000', 
            'Инвалидов_на_1000', 
            'Преступлений_на_1000',
            'Браков_на_1000',
            'Разводов_на_1000',
            'Индекс_здравоохранения', 
            'Социальный_индекс',
            'изменение_населения',
            'изменение_ВРП'
        ]
        
        # Убираем признаки, которых нет в данных
        available_features = [f for f in self.feature_names if f in df_processed.columns]
        self.feature_names = available_features
        
        print(f"Используется {len(self.feature_names)} признаков")
        
        # Разделение на train/test
        X_train, X_test, y_train, y_test, train_mask, test_mask = self.train_test_split_temporal(df_processed)
        
        print(f"Размер train: {X_train.shape}, test: {X_test.shape}")
        print(f"Период обучения: {df_processed[train_mask]['Год'].min()}-{df_processed[train_mask]['Год'].max()}")
        print(f"Период тестирования: {df_processed[test_mask]['Год'].min()}-{df_processed[test_mask]['Год'].max()}")
        
        # Масштабирование признаков
        scale_features = [f for f in self.feature_names if not f.startswith(('lag', 'ОПЖ_MA', 'изменение_'))]
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[scale_features] = self.scaler.fit_transform(X_train[scale_features])
        X_test_scaled[scale_features] = self.scaler.transform(X_test[scale_features])
        
        # Создание и обучение XGBoost модели
        self.model = xgb.XGBRegressor(
            max_depth=7,
            learning_rate=0.05,
            n_estimators=500,
            random_state=self.random_state,
            n_jobs=-1,
            subsample=0.8,
            colsample_bytree=0.8
        )
        
        print("Обучение XGBoost модели на расширенных данных...")
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        # Предсказание на тесте
        y_pred = self.model.predict(X_test_scaled)
        
        # Метрики качества
        self.calculate_metrics(y_test, y_pred)
        
        # Дополнительная валидация
        self.cross_validation(X_train_scaled, y_train)
        
        # Сохранение результатов
        self.results = {
            'X_train': X_train_scaled, 'X_test': X_test_scaled,
            'y_train': y_train, 'y_test': y_test, 'y_pred': y_pred,
            'df_processed': df_processed
        }
        
        return self
    
    def calculate_metrics(self, y_true, y_pred):
        """
        Расчет метрик качества для ОПЖ
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        print("\n" + "="*50)
        print("МЕТРИКИ КАЧЕСТВА XGBoost ДЛЯ ОПЖ (2014-2023)")
        print("="*50)
        print(f"RMSE: {rmse:.4f} лет")
        print(f"MAE: {mae:.4f} лет")
        print(f"R²: {r2:.4f}")
        print(f"Средняя ОПЖ в тесте: {y_true.mean():.2f} лет")
        print(f"Относительная ошибка: {rmse/y_true.mean()*100:.2f}%")
        
        # Дополнительные метрики
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        max_error = np.max(np.abs(y_true - y_pred))
        median_error = np.median(np.abs(y_true - y_pred))
        print(f"MAPE: {mape:.2f}%")
        print(f"Максимальная ошибка: {max_error:.2f} лет")
        print(f"Медианная ошибка: {median_error:.2f} лет")
        
        self.metrics = {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape, 'MaxError': max_error, 'MedianError': median_error}
    
    def cross_validation(self, X_train, y_train, cv=5):
        """
        Кросс-валидация для оценки устойчивости модели
        """
        print("\nКросс-валидация XGBoost (RMSE):")
        
        temp_model = xgb.XGBRegressor(
            max_depth=7,
            learning_rate=0.05,
            n_estimators=200,
            random_state=self.random_state,
            n_jobs=-1,
            subsample=0.8,
            colsample_bytree=0.8
        )
        
        try:
            scores = cross_val_score(temp_model, X_train, y_train, 
                                   scoring='neg_mean_squared_error', cv=cv)
            rmse_scores = np.sqrt(-scores)
            print(f"Среднее: {rmse_scores.mean():.4f} (+/- {rmse_scores.std() * 2:.4f})")
        except Exception as e:
            print(f"Ошибка при кросс-валидации: {e}")
    
    def save_model(self, filepath=None):
        """
        Сохранение обученной модели и всех компонентов
        """
        if not self.is_fitted:
            print("Модель не обучена!")
            return False
        
        if filepath is None:
            filepath = os.path.join(output_dir, 'opj_xgboost_model.pkl')
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'first_year': self.first_year,
            'last_known_data': self.last_known_data,
            'metrics': self.metrics
        }
        
        joblib.dump(model_data, filepath)
        print(f"XGBoost модель сохранена в {filepath}")
        return True
    
    def load_model(self, filepath=None):
        """
        Загрузка обученной модели
        """
        if filepath is None:
            filepath = os.path.join(output_dir, 'opj_xgboost_model.pkl')
            
        if not os.path.exists(filepath):
            print(f"Файл модели {filepath} не найден!")
            return False
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.first_year = model_data['first_year']
        self.last_known_data = model_data['last_known_data']
        self.metrics = model_data['metrics']
        self.is_fitted = True
        
        print(f"XGBoost модель загружена из {filepath}")
        return True

    def prepare_final_output(self, predictions_df, target_year=2024):
        """
        Подготовка финального файла для GitHub пайплайна
        """
        final_output = pd.DataFrame({
            'Регион': predictions_df['Регион'],
            'Год': predictions_df['Год'],
            'ОПЖ': predictions_df['ОПЖ_предыдущий'],
            'predictions': predictions_df['ОПЖ_прогноз_XGBoost']
        })
        
        return final_output

    def predict_future(self, df, future_years=[2024]):
        """
        Прогноз ОПЖ на будущие периоды
        """
        if not self.is_fitted:
            print("Сначала обучите модель!")
            return None
        
        if self.last_known_data is None:
            print("Нет данных для прогноза!")
            return None
        
        print(f"\nСоздание прогноза ОПЖ на {future_years} год (XGBoost)...")
        
        all_predictions = []
        
        for year in future_years:
            year_predictions = []
            
            for _, last_row in self.last_known_data.iterrows():
                # Создаем строку для прогноза
                future_row = last_row.copy()
                future_row['Год'] = year
                future_row['year_trend'] = year - self.first_year
                future_row['год_от_начала'] = year - self.first_year
                
                # Прогноз основных показателей
                future_row['Численность населения'] = last_row['Численность населения'] * 1.003
                future_row['Число умерших'] = last_row['Число умерших'] * 0.995
                future_row['Общая численность инвалидов'] = last_row['Общая численность инвалидов']
                future_row['Младенческая смертность коэф'] = last_row['Младенческая смертность коэф'] * 0.98
                future_row['Валовой региональный продукт на душу населения (ОКВЭД 2)'] = last_row['Валовой региональный продукт на душу населения (ОКВЭД 2)'] * 1.02
                future_row['Средняя ЗП'] = last_row['Средняя ЗП'] * 1.04
                future_row['Численность врачей всех специальностей'] = last_row['Численность врачей всех специальностей'] * 1.02
                future_row['Величина прожиточного минимума'] = last_row['Величина прожиточного минимума'] * 1.03
                future_row['Уровень бедности'] = last_row['Уровень бедности'] * 0.98
                future_row['Браков'] = last_row['Браков'] * 1.01
                future_row['Разводов'] = last_row['Разводов'] * 1.005
                
                # Лаги берем из последнего известного года
                future_row['lag1_ОПЖ'] = last_row['ОПЖ']
                future_row['lag2_ОПЖ'] = last_row['lag1_ОПЖ']
                future_row['lag1_Число умерших'] = last_row['Число умерших']
                future_row['lag1_Младенческая смертность коэф'] = last_row['Младенческая смертность коэф']
                future_row['lag1_Общая численность инвалидов'] = last_row['Общая численность инвалидов']
                
                # Пересчитываем скользящие средние
                future_row['ОПЖ_MA2'] = (future_row['lag1_ОПЖ'] + future_row['ОПЖ']) / 2
                future_row['ОПЖ_MA3'] = (future_row['lag2_ОПЖ'] + future_row['lag1_ОПЖ'] + future_row['ОПЖ']) / 3
                
                # Пересчитываем производные показатели
                future_row['Врачей_на_10k'] = future_row['Численность врачей всех специальностей'] / future_row['Численность населения'] * 10000
                future_row['Умерших_на_1000'] = future_row['Число умерших'] / future_row['Численность населения'] * 1000
                future_row['Инвалидов_на_1000'] = future_row['Общая численность инвалидов'] / future_row['Численность населения'] * 1000
                future_row['Преступлений_на_1000'] = future_row['Кол-во преступлений'] / future_row['Численность населения'] * 1000
                future_row['Браков_на_1000'] = future_row['Браков'] / future_row['Численность населения'] * 1000
                future_row['Разводов_на_1000'] = future_row['Разводов'] / future_row['Численность населения'] * 1000
                future_row['Индекс_здравоохранения'] = (
                    future_row['Врачей_на_10k'] + 
                    future_row['Число больничных организаций на конец отчетного года'] + 
                    future_row['Число санаторно-курортных организаций']
                ) / 3
                future_row['Социальный_индекс'] = (
                    future_row['Средняя ЗП'] / future_row['Величина прожиточного минимума'] - 
                    (future_row['Уровень бедности'] / 100)
                )
                future_row['изменение_населения'] = (future_row['Численность населения'] - last_row['Численность населения']) / last_row['Численность населения']
                future_row['изменение_ВРП'] = (future_row['Валовой региональный продукт на душу населения (ОКВЭД 2)'] - last_row['Валовой региональный продукт на душу населения (ОКВЭД 2)']) / last_row['Валовой региональный продукт на душу населения (ОКВЭД 2)']
                
                year_predictions.append(future_row)
            
            # Прогноз на конкретный год
            future_df = pd.DataFrame(year_predictions)
            X_future = future_df[self.feature_names]
            
            # Масштабирование
            scale_features = [f for f in self.feature_names if not f.startswith(('lag', 'ОПЖ_MA', 'изменение_'))]
            X_future_scaled = X_future.copy()
            X_future_scaled[scale_features] = self.scaler.transform(X_future[scale_features])
            
            # Прогноз ОПЖ
            predictions = self.model.predict(X_future_scaled)
            
            # Сохраняем результаты
            for i, (_, last_row) in enumerate(self.last_known_data.iterrows()):
                all_predictions.append({
                    'Регион': last_row['Регион'],
                    'Год': year,
                    'ОПЖ_прогноз_XGBoost': predictions[i],
                    'ОПЖ_предыдущий': last_row['ОПЖ'],
                    'Изменение_ОПЖ_XGBoost': predictions[i] - last_row['ОПЖ']
                })
            
            # Обновляем последние данные
            self.last_known_data = future_df.copy()
            self.last_known_data['ОПЖ'] = predictions
        
        results_df = pd.DataFrame(all_predictions)
        
        print("Прогноз XGBoost успешно создан")
        return results_df

# ИСПОЛЬЗОВАНИЕ:
if __name__ == "__main__":
    # Загрузка данных для ОПЖ 
    df_opj = pd.read_excel('Финальный вариант/общая_ОПЖ (2).xlsx')
    
    # Стандартизация названий регионов
    print("Стандартизация названий регионов...")
    df_opj = standardize_region_names(df_opj)
    print(f"Уникальных регионов после очистки: {df_opj['Регион'].nunique()}")
    
    print("="*70)
    print("XGBOOST МОДЕЛЬ ДЛЯ ПРОГНОЗА ОПЖ (2014-2023)")
    print("="*70)
    
    # Диагностика данных
    print(f"Размер данных: {df_opj.shape}")
    print(f"Период данных: {df_opj['Год'].min()}-{df_opj['Год'].max()}")
    print(f"Количество регионов: {df_opj['Регион'].nunique()}")
    
    # Обучение модели
    xgb_forecaster = OPJXGBoostForecaster()
    xgb_forecaster.fit(df_opj)
    
    # Сохранение модели
    xgb_forecaster.save_model()
    
    # Прогноз на 2024 год
    future_predictions = xgb_forecaster.predict_future(df_opj, [2024])
    
    if future_predictions is not None:
        # Подготовка финального вывода для GitHub пайплайна
        final_output = xgb_forecaster.prepare_final_output(future_predictions, 2024)
        
        # Сохранение в формате для пайплайна
        predictions_filepath = os.path.join(output_dir, 'predictions_ele.xlsx')
        final_output.to_excel(predictions_filepath, index=False)
        print(f"Финальный файл для пайплайна сохранен как '{predictions_filepath}'")
        
        # Вывод статистики
        print(f"\nСТАТИСТИКА ПРОГНОЗА XGBoost:")
        print(f"Средняя ОПЖ в 2024: {future_predictions['ОПЖ_прогноз_XGBoost'].mean():.2f} лет")
        print(f"Регионов с ростом ОПЖ: {(future_predictions['Изменение_ОПЖ_XGBoost'] > 0).sum()}")
        print(f"Регионов со снижением ОПЖ: {(future_predictions['Изменение_ОПЖ_XGBoost'] < 0).sum()}")
        
        # Пример первых строк финального файла
        print(f"\nПервые 5 строк финального файла:")
        print(final_output.head())
        
        # Дополнительно: сохранение полных результатов для анализа
        full_results_filepath = os.path.join(output_dir, 'full_opj_predictions_results_xgboost.csv')
        future_predictions.to_csv(full_results_filepath, index=False)
        print(f"Полные результаты XGBoost сохранены в '{full_results_filepath}'")