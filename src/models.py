"""
Models Module
Treinamento, avaliação e otimização de modelos de Machine Learning
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)
import xgboost as xgb
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Classe para treinamento e avaliação de modelos
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def create_baseline_models(self):
        """
        Cria modelos baseline
        
        Returns:
        --------
        dict
            Dicionário com modelos baseline
        """
        models = {
            'Dummy_Stratified': DummyClassifier(strategy='stratified', random_state=self.random_state),
            'Dummy_MostFrequent': DummyClassifier(strategy='most_frequent', random_state=self.random_state),
            'Logistic_Regression': LogisticRegression(max_iter=1000, random_state=self.random_state)
        }
        
        return models
    
    def create_advanced_models(self):
        """
        Cria modelos avançados
        
        Returns:
        --------
        dict
            Dicionário com modelos avançados
        """
        models = {
            'Random_Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='mlogloss'
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            ),
            'Gradient_Boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state
            )
        }
        
        return models
    
    def train_model(self, model, X_train, y_train, model_name):
        """
        Treina um modelo
        
        Parameters:
        -----------
        model : sklearn model
            Modelo a ser treinado
        X_train : array-like
            Features de treino
        y_train : array-like
            Target de treino
        model_name : str
            Nome do modelo
            
        Returns:
        --------
        model
            Modelo treinado
        """
        print(f"\n{'='*50}")
        print(f"Treinando {model_name}...")
        print(f"{'='*50}")
        
        model.fit(X_train, y_train)
        self.models[model_name] = model
        
        print(f"✓ {model_name} treinado com sucesso!")
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name, class_names=None):
        """
        Avalia um modelo
        
        Parameters:
        -----------
        model : sklearn model
            Modelo treinado
        X_test : array-like
            Features de teste
        y_test : array-like
            Target de teste
        model_name : str
            Nome do modelo
        class_names : list
            Nomes das classes
            
        Returns:
        --------
        dict
            Dicionário com métricas
        """
        y_pred = model.predict(X_test)
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision': precision,
            'recall': recall,
            'predictions': y_pred,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        # Classification report
        target_names = None
        if class_names is not None:
            target_names = list(class_names)

        results['classification_report'] = classification_report(
            y_test,
            y_pred,
            target_names=target_names,
            output_dict=True
        )

        
        self.results[model_name] = results
        
        print(f"\n{'='*50}")
        print(f"Resultados - {model_name}")
        print(f"{'='*50}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score (Macro): {f1_macro:.4f}")
        print(f"F1-Score (Weighted): {f1_weighted:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        
        return results
    
    def cross_validate_model(self, model, X, y, cv=5, scoring='accuracy'):
        """
        Realiza validação cruzada
        
        Parameters:
        -----------
        model : sklearn model
            Modelo a ser validado
        X : array-like
            Features
        y : array-like
            Target
        cv : int
            Número de folds
        scoring : str
            Métrica de avaliação
            
        Returns:
        --------
        dict
            Resultados da validação cruzada
        """
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        results = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
        
        print(f"\nValidação Cruzada ({cv}-fold):")
        print(f"Mean {scoring}: {results['mean']:.4f} (+/- {results['std']:.4f})")
        
        return results
    
    def optimize_hyperparameters(self, model_type, X_train, y_train, method='random', cv=5):
        """
        Otimiza hiperparâmetros
        
        Parameters:
        -----------
        model_type : str
            Tipo de modelo ('random_forest', 'xgboost', 'lightgbm')
        X_train : array-like
            Features de treino
        y_train : array-like
            Target de treino
        method : str
            Método de busca ('grid' ou 'random')
        cv : int
            Número de folds
            
        Returns:
        --------
        best_model
            Model com melhores hiperparâmetros
        """
        print(f"\n{'='*50}")
        print(f"Otimizando hiperparâmetros - {model_type}")
        print(f"Método: {method.upper()}")
        print(f"{'='*50}")
        
        param_grids = self._get_param_grids()
        
        if model_type.lower() not in param_grids:
            raise ValueError(f"Modelo {model_type} não suportado")
        
        model = self._get_base_model(model_type)
        param_grid = param_grids[model_type.lower()]
        
        if method == 'grid':
            search = GridSearchCV(
                model, 
                param_grid, 
                cv=cv, 
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
        else:  # random
            search = RandomizedSearchCV(
                model, 
                param_grid, 
                n_iter=20,
                cv=cv, 
                scoring='accuracy',
                n_jobs=-1,
                verbose=1,
                random_state=self.random_state
            )
        
        search.fit(X_train, y_train)
        
        print(f"\nMelhores parâmetros:")
        for param, value in search.best_params_.items():
            print(f"  {param}: {value}")
        print(f"\nMelhor score (CV): {search.best_score_:.4f}")
        
        return search.best_estimator_
    
    def _get_param_grids(self):
        """
        Retorna grids de parâmetros para otimização
        """
        return {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            },
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1, 0.3],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'lightgbm': {
                'n_estimators': [100, 200, 300],
                'num_leaves': [31, 50, 70],
                'learning_rate': [0.01, 0.05, 0.1, 0.3],
                'max_depth': [5, 10, 15, -1],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
    
    def _get_base_model(self, model_type):
        """
        Retorna modelo base para otimização
        """
        models = {
            'random_forest': RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
            'xgboost': xgb.XGBClassifier(random_state=self.random_state, n_jobs=-1, eval_metric='mlogloss'),
            'lightgbm': lgb.LGBMClassifier(random_state=self.random_state, n_jobs=-1, verbose=-1)
        }
        return models[model_type.lower()]
    
    def compare_models(self):
        """
        Compara todos os modelos treinados
        
        Returns:
        --------
        pd.DataFrame
            Tabela comparativa
        """
        if not self.results:
            print("Nenhum modelo foi avaliado ainda!")
            return None
        
        comparison = []
        
        for model_name, results in self.results.items():
            comparison.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'F1-Score (Macro)': results['f1_macro'],
                'F1-Score (Weighted)': results['f1_weighted'],
                'Precision': results['precision'],
                'Recall': results['recall']
            })
        
        df_comparison = pd.DataFrame(comparison)
        df_comparison = df_comparison.sort_values('Accuracy', ascending=False)
        
        print("\n{'='*80}")
        print("COMPARAÇÃO DE MODELOS")
        print(f"{'='*80}")
        print(df_comparison.to_string(index=False))
        print(f"{'='*80}")
        
        return df_comparison
    
    def select_best_model(self, metric='accuracy'):
        """
        Seleciona o melhor modelo baseado em uma métrica
        
        Parameters:
        -----------
        metric : str
            Métrica para seleção
            
        Returns:
        --------
        tuple
            (model_name, model, results)
        """
        if not self.results:
            print("Nenhum modelo foi avaliado ainda!")
            return None
        
        best_score = -1
        best_name = None
        
        for model_name, results in self.results.items():
            score = results[metric]
            if score > best_score:
                best_score = score
                best_name = model_name
        
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        print(f"\n{'='*50}")
        print(f"MELHOR MODELO: {best_name}")
        print(f"{metric.upper()}: {best_score:.4f}")
        print(f"{'='*50}")
        
        return best_name, self.best_model, self.results[best_name]
    
    def plot_confusion_matrix(self, model_name, class_names=None, figsize=(10, 8)):
        """
        Plota matriz de confusão
        
        Parameters:
        -----------
        model_name : str
            Nome do modelo
        class_names : list
            Nomes das classes
        figsize : tuple
            Tamanho da figura
        """
        if model_name not in self.results:
            print(f"Modelo {model_name} não encontrado!")
            return
        
        cm = self.results[model_name]['confusion_matrix']
        
        plt.figure(figsize=figsize)
        labels = None
        if class_names is not None:
            labels = list(class_names)

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels if labels is not None else 'auto',
            yticklabels=labels if labels is not None else 'auto'
        )

        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_feature_importance(self, model_name, feature_names, top_n=20, figsize=(10, 8)):
        """
        Plota importância das features
        
        Parameters:
        -----------
        model_name : str
            Nome do modelo
        feature_names : list
            Nomes das features
        top_n : int
            Número de features mais importantes
        figsize : tuple
            Tamanho da figura
        """
        if model_name not in self.models:
            print(f"Modelo {model_name} não encontrado!")
            return
        
        model = self.models[model_name]
        
        if not hasattr(model, 'feature_importances_'):
            print(f"Modelo {model_name} não possui feature_importances_!")
            return
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=figsize)
        plt.title(f'Feature Importance - {model_name} (Top {top_n})')
        plt.barh(range(top_n), importances[indices])
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        return plt.gcf()
    
    def save_model(self, model_name, filepath):
        """
        Salva modelo treinado
        
        Parameters:
        -----------
        model_name : str
            Nome do modelo
        filepath : str
            Caminho para salvar
        """
        if model_name not in self.models:
            print(f"Modelo {model_name} não encontrado!")
            return
        
        joblib.dump(self.models[model_name], filepath)
        print(f"✓ Modelo {model_name} salvo em: {filepath}")
    
    def load_model(self, filepath):
        """
        Carrega modelo salvo
        
        Parameters:
        -----------
        filepath : str
            Caminho do modelo
            
        Returns:
        --------
        model
            Modelo carregado
        """
        model = joblib.load(filepath)
        print(f"✓ Modelo carregado de: {filepath}")
        return model


if __name__ == "__main__":
    # Teste do módulo
    trainer = ModelTrainer()
    print("Módulo de modelagem carregado com sucesso!")
