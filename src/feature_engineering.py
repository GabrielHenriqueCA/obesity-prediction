"""
Feature Engineering Module
Criação e transformação de features para o modelo de obesidade
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Classe para engenharia de features do dataset de obesidade
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def create_bmi(self, df):
        """
        Cria feature de IMC (Índice de Massa Corporal)
        
        IMC = Peso (kg) / Altura² (m²)
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset com colunas 'Weight' e 'Height'
            
        Returns:
        --------
        pd.DataFrame
            Dataset com nova coluna 'BMI'
        """
        df = df.copy()
        df['BMI'] = df['Weight'] / (df['Height'] ** 2)
        return df
    
    def create_bmi_category(self, df):
        """
        Categoriza IMC segundo padrões da OMS
        
        Categorias:
        - Underweight: < 18.5
        - Normal: 18.5 - 24.9
        - Overweight: 25 - 29.9
        - Obesity_1: 30 - 34.9
        - Obesity_2: 35 - 39.9
        - Obesity_3: >= 40
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset com coluna 'BMI'
            
        Returns:
        --------
        pd.DataFrame
            Dataset com nova coluna 'BMI_Category'
        """
        df = df.copy()
        
        conditions = [
            (df['BMI'] < 18.5),
            (df['BMI'] >= 18.5) & (df['BMI'] < 25),
            (df['BMI'] >= 25) & (df['BMI'] < 30),
            (df['BMI'] >= 30) & (df['BMI'] < 35),
            (df['BMI'] >= 35) & (df['BMI'] < 40),
            (df['BMI'] >= 40)
        ]
        
        categories = [
            'Underweight', 
            'Normal', 
            'Overweight', 
            'Obesity_1', 
            'Obesity_2', 
            'Obesity_3'
        ]
        
        df['BMI_Category'] = np.select(conditions, categories, default='Normal')
        
        return df
    
    def create_caloric_balance_score(self, df):
        """
        Cria score de balanço calórico
        Combina: consumo de alimentos calóricos, vegetais e atividade física
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset com colunas FAVC, FCVC, FAF
            
        Returns:
        --------
        pd.DataFrame
            Dataset com nova coluna 'Caloric_Balance_Score'
        """
        df = df.copy()
        
        # Converter para numérico se necessário
        favc_map = {'yes': 1, 'no': 0}
        df['FAVC_num'] = df['FAVC'].map(favc_map) if df['FAVC'].dtype == 'object' else df['FAVC']
        
        # Score: alimentos calóricos (+) - vegetais (-) - atividade física (-)
        # Normalizado entre 0-10
        df['Caloric_Balance_Score'] = (
            df['FAVC_num'] * 3 +  # Alto impacto de alimentos calóricos
            (3 - df['FCVC']) +     # Menos vegetais = pior
            (3 - df['FAF'])        # Menos atividade física = pior
        )
        
        # Normalizar para 0-10
        df['Caloric_Balance_Score'] = (
            df['Caloric_Balance_Score'] / df['Caloric_Balance_Score'].max() * 10
        )
        
        return df
    
    def create_risk_score(self, df):
        """
        Cria score de risco de obesidade
        Considera: histórico familiar, tabagismo, álcool, monitoramento calórico
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset
            
        Returns:
        --------
        pd.DataFrame
            Dataset com nova coluna 'Risk_Score'
        """
        df = df.copy()
        
        risk_score = 0
        
        # Histórico familiar (+3 pontos)
        family_map = {'yes': 3, 'no': 0}
        risk_score += df['family_history'].map(family_map) if df['family_history'].dtype == 'object' else df['family_history'] * 3
        
        # Tabagismo (+2 pontos)
        smoke_map = {'yes': 2, 'no': 0}
        risk_score += df['SMOKE'].map(smoke_map) if df['SMOKE'].dtype == 'object' else df['SMOKE'] * 2
        
        # Não monitora calorias (+2 pontos)
        scc_map = {'yes': 0, 'no': 2}
        risk_score += df['SCC'].map(scc_map) if df['SCC'].dtype == 'object' else (1 - df['SCC']) * 2
        
        # Consumo de álcool (baseado em frequência)
        calc_map = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
        if df['CALC'].dtype == 'object':
            risk_score += df['CALC'].map(calc_map).fillna(0)
        
        df['Risk_Score'] = risk_score
        
        return df
    
    def create_hydration_level(self, df):
        """
        Categoriza nível de hidratação
        
        Níveis:
        - Low: < 1.5L
        - Medium: 1.5 - 2.5L
        - High: > 2.5L
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset com coluna 'CH2O'
            
        Returns:
        --------
        pd.DataFrame
            Dataset com nova coluna 'Hydration_Level'
        """
        df = df.copy()
        
        conditions = [
            (df['CH2O'] < 1.5),
            (df['CH2O'] >= 1.5) & (df['CH2O'] <= 2.5),
            (df['CH2O'] > 2.5)
        ]
        
        levels = ['Low', 'Medium', 'High']
        
        df['Hydration_Level'] = np.select(conditions, levels, default='Medium')
        
        return df
    
    def create_activity_level(self, df):
        """
        Categoriza nível de atividade física
        
        Níveis:
        - Sedentary: 0-1
        - Light: 1-2
        - Moderate: 2-3
        - Active: > 3
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset com coluna 'FAF'
            
        Returns:
        --------
        pd.DataFrame
            Dataset com nova coluna 'Activity_Level'
        """
        df = df.copy()
        
        conditions = [
            (df['FAF'] <= 1),
            (df['FAF'] > 1) & (df['FAF'] <= 2),
            (df['FAF'] > 2) & (df['FAF'] <= 3),
            (df['FAF'] > 3)
        ]
        
        levels = ['Sedentary', 'Light', 'Moderate', 'Active']
        
        df['Activity_Level'] = np.select(conditions, levels, default='Light')
        
        return df
    
    def create_screen_time_category(self, df):
        """
        Categoriza tempo de tela (dispositivos tecnológicos)
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset com coluna 'TUE' (Technology Use)
            
        Returns:
        --------
        pd.DataFrame
            Dataset com nova coluna 'Screen_Time_Category'
        """
        df = df.copy()
        
        conditions = [
            (df['TUE'] <= 1),
            (df['TUE'] > 1) & (df['TUE'] <= 2),
            (df['TUE'] > 2)
        ]
        
        categories = ['Low', 'Moderate', 'High']
        
        df['Screen_Time_Category'] = np.select(conditions, categories, default='Moderate')
        
        return df
    
    def encode_categorical_features(self, df, categorical_cols, fit=True):
        """
        Codifica features categóricas
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset
        categorical_cols : list
            Lista de colunas categóricas
        fit : bool
            Se True, ajusta os encoders
            
        Returns:
        --------
        pd.DataFrame
            Dataset com features codificadas
        """
        df = df.copy()
        
        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.label_encoders:
                        # Handle unseen labels
                        df[col] = df[col].astype(str)
                        df[col] = df[col].apply(
                            lambda x: x if x in self.label_encoders[col].classes_ 
                            else self.label_encoders[col].classes_[0]
                        )
                        df[col] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def scale_features(self, df, numeric_cols, fit=True):
        """
        Normaliza features numéricas
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset
        numeric_cols : list
            Lista de colunas numéricas
        fit : bool
            Se True, ajusta o scaler
            
        Returns:
        --------
        pd.DataFrame
            Dataset com features normalizadas
        """
        df = df.copy()
        
        if fit:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        else:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        
        return df
    
    def create_all_features(self, df):
        """
        Cria todas as features derivadas
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset original
            
        Returns:
        --------
        pd.DataFrame
            Dataset com todas as features
        """
        df = df.copy()
        
        # ============================================================================
        # IMPORTANTE: BMI REMOVIDO PARA EVITAR DATA LEAKAGE
        # ============================================================================
        # Decisão técnica: NÃO criar features baseadas em BMI (Índice de Massa Corporal)
        # 
        # MOTIVO:
        # A classificação médica de obesidade É DEFINIDA pelo BMI:
        #   - BMI < 18.5  → Insufficient Weight
        #   - BMI 18.5-25 → Normal Weight
        #   - BMI 25-30   → Overweight
        #   - BMI 30-35   → Obesity I
        #   - BMI 35-40   → Obesity II
        #   - BMI > 40    → Obesity III
        #
        # PROBLEMA:
        # Usar BMI como feature causa TARGET LEAKAGE porque:
        # 1. Um modelo usando APENAS BMI atinge 95%+ de acurácia
        # 2. BMI é calculado como: Weight / Height²
        # 3. O target (categoria de obesidade) é praticamente derivado do BMI
        # 4. Isso torna o modelo uma "calculadora sofisticada" em vez de ML real
        #
        # SOLUÇÃO:
        # Removemos BMI para FORÇAR o modelo a aprender padrões comportamentais:
        #   - Hábitos alimentares (FAVC, FCVC, NCP, CAEC)
        #   - Atividade física (FAF)
        #   - Estilo de vida (TUE, CALC, SMOKE)
        #   - Genética (family_history)
        #
        # IMPACTO:
        # - Acurácia esperada: 70-85% (vs 95%+ com BMI)
        # - Valor do modelo: MUITO MAIOR (aprende fatores de risco reais)
        # - Aplicação prática: Prevenção e identificação de risco ANTES de medir peso
        #
        # CÓDIGO ORIGINAL COMENTADO:
        # df = self.create_bmi(df)
        # df = self.create_bmi_category(df)
        # ============================================================================
        
        # Features comportamentais (foco principal do modelo)
        df = self.create_caloric_balance_score(df)
        df = self.create_risk_score(df)
        df = self.create_hydration_level(df)
        df = self.create_activity_level(df)
        df = self.create_screen_time_category(df)

        cols_to_drop = ['Weight', 'Height']
        existing_cols = [c for c in cols_to_drop if c in df.columns]
        
        if existing_cols:
            df = df.drop(columns=existing_cols)
        
        return df
    
    def get_feature_names(self):
        """
        Retorna nomes das features
        
        Returns:
        --------
        list
            Lista de nomes de features
        """
        return self.feature_names


def create_feature_engineering_pipeline():
    """
    Cria pipeline de feature engineering
    
    Returns:
    --------
    FeatureEngineer
        Instância do feature engineer
    """
    return FeatureEngineer()


if __name__ == "__main__":
    # Teste do módulo
    engineer = FeatureEngineer()
    print("Módulo de feature engineering carregado com sucesso!")
