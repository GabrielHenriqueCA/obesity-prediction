"""
Preprocessing Module
Funções para limpeza e preparação de dados
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Classe para preprocessamento de dados do dataset de obesidade
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.target_encoder = LabelEncoder()
        
    def load_data(self, filepath):
        """
        Carrega o dataset
        
        Parameters:
        -----------
        filepath : str
            Caminho para o arquivo CSV
            
        Returns:
        --------
        pd.DataFrame
            Dataset carregado
        """
        df = pd.read_csv(filepath)
        return df
    
    def check_data_quality(self, df):
        """
        Verifica qualidade dos dados
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset a ser verificado
            
        Returns:
        --------
        dict
            Relatório de qualidade
        """
        quality_report = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict()
        }
        
        return quality_report
    
    def handle_missing_values(self, df):
        """
        Trata valores ausentes
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset com possíveis valores ausentes
            
        Returns:
        --------
        pd.DataFrame
            Dataset sem valores ausentes
        """
        # Verificar se há valores missing
        if df.isnull().sum().sum() > 0:
            # Para variáveis numéricas: preencher com mediana
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            
            # Para variáveis categóricas: preencher com moda
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        return df
    
    def remove_duplicates(self, df):
        """
        Remove linhas duplicadas
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset com possíveis duplicatas
            
        Returns:
        --------
        pd.DataFrame
            Dataset sem duplicatas
        """
        df_clean = df.drop_duplicates()
        n_removed = len(df) - len(df_clean)
        
        if n_removed > 0:
            print(f"Removidas {n_removed} linhas duplicadas")
        
        return df_clean
    
    def detect_outliers(self, df, columns, method='iqr', threshold=1.5):
        """
        Detecta outliers usando IQR
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset
        columns : list
            Colunas numéricas para análise
        method : str
            Método de detecção ('iqr' ou 'zscore')
        threshold : float
            Threshold para IQR (padrão: 1.5)
            
        Returns:
        --------
        dict
            Dicionário com outliers por coluna
        """
        outliers = {}
        
        for col in columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
            
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers[col] = df[z_scores > 3].index.tolist()
        
        return outliers
    
    def split_data(self, df, target_col, test_size=0.2, random_state=42):
        """
        Separa dados em treino e teste
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset completo
        target_col : str
            Nome da coluna alvo
        test_size : float
            Proporção do conjunto de teste
        random_state : int
            Seed para reprodutibilidade
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y  # Mantém proporção das classes
        )
        
        return X_train, X_test, y_train, y_test
    
    def encode_target(self, y_train, y_test):
        """
        Codifica variável alvo
        
        Parameters:
        -----------
        y_train : pd.Series
            Target de treino
        y_test : pd.Series
            Target de teste
            
        Returns:
        --------
        tuple
            (y_train_encoded, y_test_encoded)
        """
        y_train_encoded = self.target_encoder.fit_transform(y_train)
        y_test_encoded = self.target_encoder.transform(y_test)
        
        return y_train_encoded, y_test_encoded
    
    def get_target_mapping(self):
        """
        Retorna mapeamento da variável alvo
        
        Returns:
        --------
        dict
            Mapeamento de classes
        """
        if hasattr(self.target_encoder, 'classes_'):
            return {i: label for i, label in enumerate(self.target_encoder.classes_)}
        return None


def create_preprocessing_pipeline():
    """
    Cria pipeline de preprocessamento
    
    Returns:
    --------
    DataPreprocessor
        Instância do preprocessador
    """
    return DataPreprocessor()


if __name__ == "__main__":
    # Teste do módulo
    preprocessor = DataPreprocessor()
    print("Módulo de preprocessamento carregado com sucesso!")
