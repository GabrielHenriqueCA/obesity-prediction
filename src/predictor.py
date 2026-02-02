"""
Predictor Module
Sistema preditivo para diagnóstico de obesidade
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ObesityPredictor:
    """
    Sistema preditivo para níveis de obesidade
    """
    
    def __init__(self, model_path=None):
        """
        Inicializa o preditor
        
        Parameters:
        -----------
        model_path : str
            Caminho para o modelo salvo
        """
        self.model = None
        self.preprocessor = None
        self.feature_engineer = None
        self.target_mapping = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        Carrega modelo treinado
        
        Parameters:
        -----------
        model_path : str
            Caminho do modelo
        """
        try:
            saved_data = joblib.load(model_path)
            
            if isinstance(saved_data, dict):
                self.model = saved_data.get('model')
                self.preprocessor = saved_data.get('preprocessor')
                self.feature_engineer = saved_data.get('feature_engineer')
                self.target_mapping = saved_data.get('target_mapping')
            else:
                self.model = saved_data
            
            print(f"✓ Modelo carregado com sucesso de: {model_path}")
        except Exception as e:
            print(f"✗ Erro ao carregar modelo: {e}")
    
    def set_components(self, model, preprocessor=None, feature_engineer=None, target_mapping=None):
        """
        Define componentes do sistema
        
        Parameters:
        -----------
        model : sklearn model
            Modelo treinado
        preprocessor : DataPreprocessor
            Preprocessador
        feature_engineer : FeatureEngineer
            Feature engineer
        target_mapping : dict
            Mapeamento de classes
        """
        self.model = model
        self.preprocessor = preprocessor
        self.feature_engineer = feature_engineer
        self.target_mapping = target_mapping
    
    def validate_input(self, patient_data):
        """
        Valida dados de entrada
        
        Parameters:
        -----------
        patient_data : dict or pd.DataFrame
            Dados do paciente
            
        Returns:
        --------
        bool
            True se válido
        """
        required_fields = [
            'Gender', 'Age', 'Height', 'Weight', 
            'family_history_with_overweight', 'FAVC', 'FCVC', 
            'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 
            'FAF', 'TUE', 'CALC', 'MTRANS'
        ]
        
        if isinstance(patient_data, dict):
            missing_fields = [field for field in required_fields if field not in patient_data]
            if missing_fields:
                print(f"✗ Campos ausentes: {missing_fields}")
                return False
        elif isinstance(patient_data, pd.DataFrame):
            missing_fields = [field for field in required_fields if field not in patient_data.columns]
            if missing_fields:
                print(f"✗ Colunas ausentes: {missing_fields}")
                return False
        
        return True
    
    def preprocess_input(self, patient_data):
        """
        Preprocessa dados de entrada
        
        Parameters:
        -----------
        patient_data : dict or pd.DataFrame
            Dados do paciente
            
        Returns:
        --------
        pd.DataFrame
            Dados preprocessados
        """
        # Converter para DataFrame se necessário
        if isinstance(patient_data, dict):
            df = pd.DataFrame([patient_data])
        else:
            df = patient_data.copy()
        
        # Aplicar feature engineering se disponível
        if self.feature_engineer:
            df = self.feature_engineer.create_all_features(df)
        
        return df
    
    def predict(self, patient_data):
        """
        Faz predição para um paciente
        
        Parameters:
        -----------
        patient_data : dict or pd.DataFrame
            Dados do paciente
            
        Returns:
        --------
        dict
            Resultado da predição
        """
        if not self.model:
            return {"error": "Modelo não carregado"}
        
        # Validar entrada
        if not self.validate_input(patient_data):
            return {"error": "Dados de entrada inválidos"}
        
        # Preprocessar
        df_processed = self.preprocess_input(patient_data)
        
        # Predição
        try:
            prediction = self.model.predict(df_processed)[0]
            probabilities = self.model.predict_proba(df_processed)[0]
            
            # Mapear classe predita
            if self.target_mapping:
                predicted_class = self.target_mapping.get(prediction, f"Class_{prediction}")
            else:
                predicted_class = prediction
            
            # Preparar resultado
            result = {
                'predicted_class': predicted_class,
                'predicted_class_id': int(prediction),
                'probabilities': {
                    self.target_mapping.get(i, f"Class_{i}"): float(prob) 
                    for i, prob in enumerate(probabilities)
                } if self.target_mapping else {i: float(prob) for i, prob in enumerate(probabilities)},
                'confidence': float(max(probabilities)),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Erro na predição: {e}"}
    
    def predict_with_explanation(self, patient_data):
        """
        Predição com explicação médica
        
        Parameters:
        -----------
        patient_data : dict or pd.DataFrame
            Dados do paciente
            
        Returns:
        --------
        dict
            Resultado com explicação
        """
        result = self.predict(patient_data)
        
        if "error" in result:
            return result
        
        # Adicionar explicação médica
        explanation = self._generate_medical_explanation(patient_data, result)
        result['medical_explanation'] = explanation
        
        return result
    
    def _generate_medical_explanation(self, patient_data, prediction_result):
        """
        Gera explicação médica da predição
        
        Parameters:
        -----------
        patient_data : dict or pd.DataFrame
            Dados do paciente
        prediction_result : dict
            Resultado da predição
            
        Returns:
        --------
        dict
            Explicação médica
        """
        if isinstance(patient_data, pd.DataFrame):
            data = patient_data.iloc[0].to_dict()
        else:
            data = patient_data
        
        # Calcular IMC
        bmi = data['Weight'] / (data['Height'] ** 2)
        
        # Fatores de risco
        risk_factors = []
        protective_factors = []
        
        # Histórico familiar
        if data.get('family_history_with_overweight') == 'yes':
            risk_factors.append("Histórico familiar de sobrepeso/obesidade")
        
        # Alimentação
        if data.get('FAVC') == 'yes':
            risk_factors.append("Consumo frequente de alimentos altamente calóricos")
        
        if data.get('FCVC', 0) < 2:
            risk_factors.append("Baixo consumo de vegetais")
        else:
            protective_factors.append("Bom consumo de vegetais")
        
        # Atividade física
        if data.get('FAF', 0) < 1:
            risk_factors.append("Atividade física insuficiente (sedentarismo)")
        elif data.get('FAF', 0) >= 2:
            protective_factors.append("Prática regular de atividade física")
        
        # Hidratação
        if data.get('CH2O', 0) < 1.5:
            risk_factors.append("Hidratação insuficiente")
        elif data.get('CH2O', 0) >= 2:
            protective_factors.append("Boa hidratação")
        
        # Tabagismo
        if data.get('SMOKE') == 'yes':
            risk_factors.append("Tabagismo")
        
        # Álcool
        if data.get('CALC') in ['Frequently', 'Always']:
            risk_factors.append("Consumo frequente de álcool")
        
        # Monitoramento calórico
        if data.get('SCC') == 'no':
            risk_factors.append("Não monitora ingestão calórica")
        else:
            protective_factors.append("Monitora ingestão calórica")
        
        explanation = {
            'bmi': round(bmi, 2),
            'bmi_category': self._get_bmi_category(bmi),
            'risk_factors': risk_factors,
            'protective_factors': protective_factors,
            'recommendations': self._generate_recommendations(risk_factors, bmi)
        }
        
        return explanation
    
    def _get_bmi_category(self, bmi):
        """
        Retorna categoria do IMC
        """
        if bmi < 18.5:
            return "Abaixo do peso"
        elif bmi < 25:
            return "Peso normal"
        elif bmi < 30:
            return "Sobrepeso"
        elif bmi < 35:
            return "Obesidade Grau I"
        elif bmi < 40:
            return "Obesidade Grau II"
        else:
            return "Obesidade Grau III (Mórbida)"
    
    def _generate_recommendations(self, risk_factors, bmi):
        """
        Gera recomendações médicas
        """
        recommendations = []
        
        if bmi >= 30:
            recommendations.append("Consulta com nutricionista recomendada")
            recommendations.append("Avaliação médica completa necessária")
        
        if "Atividade física insuficiente" in risk_factors:
            recommendations.append("Iniciar programa de atividade física regular (mínimo 150 min/semana)")
        
        if "Consumo frequente de alimentos altamente calóricos" in risk_factors:
            recommendations.append("Reduzir consumo de alimentos processados e altamente calóricos")
        
        if "Baixo consumo de vegetais" in risk_factors:
            recommendations.append("Aumentar consumo de frutas e vegetais (mínimo 5 porções/dia)")
        
        if "Hidratação insuficiente" in risk_factors:
            recommendations.append("Aumentar ingestão de água (mínimo 2L/dia)")
        
        if "Tabagismo" in risk_factors:
            recommendations.append("Programa de cessação do tabagismo")
        
        if "Não monitora ingestão calórica" in risk_factors:
            recommendations.append("Iniciar diário alimentar ou app de monitoramento nutricional")
        
        return recommendations
    
    def batch_predict(self, patients_data):
        """
        Predição em lote
        
        Parameters:
        -----------
        patients_data : pd.DataFrame
            Dados de múltiplos pacientes
            
        Returns:
        --------
        pd.DataFrame
            Resultados das predições
        """
        results = []
        
        for idx, row in patients_data.iterrows():
            result = self.predict(row.to_dict())
            results.append(result)
        
        return pd.DataFrame(results)
    
    def save_system(self, filepath):
        """
        Salva sistema completo
        
        Parameters:
        -----------
        filepath : str
            Caminho para salvar
        """
        system_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'feature_engineer': self.feature_engineer,
            'target_mapping': self.target_mapping
        }
        
        joblib.dump(system_data, filepath)
        print(f"✓ Sistema preditivo salvo em: {filepath}")


def create_example_patient():
    """
    Cria dados de exemplo de um paciente
    
    Returns:
    --------
    dict
        Dados do paciente
    """
    example = {
        'Gender': 'Male',
        'Age': 35,
        'Height': 1.75,
        'Weight': 90,
        'family_history_with_overweight': 'yes',
        'FAVC': 'yes',
        'FCVC': 2.0,
        'NCP': 3.0,
        'CAEC': 'Sometimes',
        'SMOKE': 'no',
        'CH2O': 2.0,
        'SCC': 'no',
        'FAF': 1.0,
        'TUE': 1.0,
        'CALC': 'Sometimes',
        'MTRANS': 'Public_Transportation'
    }
    
    return example


if __name__ == "__main__":
    # Teste do módulo
    predictor = ObesityPredictor()
    print("Módulo de predição carregado com sucesso!")
    
    # Exemplo de uso
    example_patient = create_example_patient()
    print("\nDados de exemplo criados:")
    for key, value in example_patient.items():
        print(f"  {key}: {value}")
