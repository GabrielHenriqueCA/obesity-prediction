"""
🏥 Sistema de Previsão de Obesidade
App Streamlit para predição de níveis de obesidade com Dashboard Analítico
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import dashboard_visualizations as dv

# Configuração da página
st.set_page_config(
    page_title="Previsão de Obesidade",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Mapeamento de classes em inglês para português
CLASS_TRANSLATION = {
    'Normal_Weight': 'Peso Normal',
    'Insufficient_Weight': 'Peso Insuficiente',
    'Overweight_Level_I': 'Sobrepeso Nível I',
    'Overweight_Level_II': 'Sobrepeso Nível II',
    'Obesity_Type_I': 'Obesidade Tipo I',
    'Obesity_Type_II': 'Obesidade Tipo II',
    'Obesity_Type_III': 'Obesidade Tipo III'
}

# CSS customizado
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.2rem;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def carregar_modelo():
    """Carrega o modelo treinado"""
    model_path = Path('models/obesity_prediction_model.pkl') 
    
    if not model_path.exists():
        st.error(f"""
        ❌ **Modelo não encontrado!**
        
        O sistema procurou em: `{model_path.absolute()}`
        
        Certifique-se de que o arquivo .pkl foi enviado para o GitHub
        e não está listado no .gitignore.
        """)
        st.stop()
    
    return joblib.load(model_path)


@st.cache_data
def carregar_dados_dashboard():
    """Carrega dataset para visualizações"""
    try:
        # Pega a pasta onde ESTE arquivo (app.py) está (pasta src)
        pasta_atual = Path(__file__).parent
        
        # Volta uma pasta para chegar na raiz e entra em 'data'
        # src -> raiz -> data -> arquivo
        caminho_csv = pasta_atual.parent / 'data' / 'Obesity.csv'
        
        df = pd.read_csv(caminho_csv)
        return df

    except Exception as e:
        st.error(f"❌ Erro ao carregar dataset: {e}")
        return None


def criar_features(dados):
    """Cria features derivadas para o modelo"""
    df = pd.DataFrame([dados])
    
    import sys
    sys.path.append('src')
    from feature_engineering import FeatureEngineer
    
    engineer = FeatureEngineer()
    df_features = engineer.create_all_features(df)
    
    return df_features


def fazer_predicao(dados_paciente, model_data):
    """Faz a predição usando o modelo carregado"""
    try:
        df_pac = criar_features(dados_paciente)
        
        for col in model_data['feature_names']:
            if col not in df_pac.columns:
                df_pac[col] = 0
        
        df_pac = df_pac[model_data['feature_names']]
        
        for col, le in model_data['label_encoders'].items():
            if col in df_pac.columns:
                try:
                    df_pac[col] = le.transform(df_pac[col].astype(str))
                except:
                    df_pac[col] = 0
        
        X_pac = model_data['scaler'].transform(df_pac)
        pred = model_data['model'].predict(X_pac)[0]
        proba = model_data['model'].predict_proba(X_pac)[0]
        
        classe_en = model_data['target_mapping'][pred]
        classe_pt = CLASS_TRANSLATION.get(classe_en, classe_en)
        
        probabilidades_pt = {
            CLASS_TRANSLATION.get(model_data['target_mapping'][i], model_data['target_mapping'][i]): p 
            for i, p in enumerate(proba)
        }
        
        return {
            'classe': classe_pt,
            'confianca': max(proba),
            'probabilidades': probabilidades_pt
        }
    except Exception as e:
        st.error(f"Erro na predição: {str(e)}")
        return None


def main():
    # Cabeçalho
    st.markdown('<div class="main-header">🏥 Sistema de Previsão de Obesidade</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Sistema inteligente para avaliação de risco de obesidade baseado em padrões comportamentais</div>', unsafe_allow_html=True)
    
    # Sistema de navegação com tabs
    tab1, tab2 = st.tabs(["🔮 Fazer Predição", "📊 Dashboard Analítico"])
    
    # ==================== TAB 1: PREDIÇÃO ====================
    with tab1:
        model_data = carregar_modelo()
        
        with st.expander("ℹ️ Informações sobre o Sistema"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Modelo", model_data['metrics']['model_name'])
            with col2:
                st.metric("Acurácia", f"{model_data['metrics']['accuracy']*100:.2f}%")
            with col3:
                st.metric("F1-Score", f"{model_data['metrics']['f1_score']*100:.2f}%")
            
            st.info("""
            **Como funciona:**
            - Este sistema analisa seus hábitos comportamentais e estilo de vida
            - NÃO usa peso ou altura diretamente
            - Aprende padrões reais de risco de obesidade
            - Fornece predição e recomendações personalizadas
            """)
        
        st.markdown("---")
        st.markdown("## 📋 Preencha os Dados do Paciente")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 👤 Dados Pessoais")
            gender = st.selectbox("Gênero", ["Masculino", "Feminino"], key="gender")
            age = st.number_input("Idade", min_value=10, max_value=100, value=30, key="age")
            family_history = st.selectbox("Histórico Familiar de Sobrepeso", ["Sim", "Não"], key="family")
            smoke = st.selectbox("Fumante", ["Não", "Sim"], key="smoke")
            mtrans = st.selectbox("Transporte Principal", ["Transporte Público", "Automóvel", "Caminhando", "Motocicleta", "Bicicleta"], key="mtrans")
        
        with col2:
            st.markdown("### 🍎 Alimentação")
            favc = st.selectbox("Cons Alimentos Altamente Calóricos?", ["Não", "Sim"], key="favc")
            fcvc = st.slider("Consumo de Vegetais (0-3)", min_value=0.0, max_value=3.0, value=2.0, step=0.5, key="fcvc", help="0 = Nunca, 1 = Às vezes, 2 = Geralmente, 3 = Sempre")
            ncp = st.slider("Número de Refeições por Dia", min_value=1.0, max_value=4.0, value=3.0, step=0.5, key="ncp")
            caec = st.selectbox("Consumo de Alimentos entre Refeições", ["Não", "Às Vezes", "Frequentemente", "Sempre"], key="caec")
            calc = st.selectbox("Consumo de Álcool", ["Não", "Às Vezes", "Frequentemente", "Sempre"], key="calc")
        
        with col3:
            st.markdown("### 🏃 Atividade & Hábitos")
            ch2o = st.slider("Consumo de Água (Litros/dia)", min_value=0.0, max_value=3.0, value=2.0, step=0.5, key="ch2o")
            scc = st.selectbox("Monitora Calorias?", ["Não", "Sim"], key="scc")
            faf = st.slider("Atividade Física (dias/semana)", min_value=0.0, max_value=3.0, value=1.0, step=0.5, key="faf", help="0 = Sedentário, 1-2 = Moderado, 3 = Ativo")
            tue = st.slider("Tempo de Tela (horas/dia)", min_value=0.0, max_value=3.0, value=1.0, step=0.5, key="tue")
        
        st.markdown("---")
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            predict_button = st.button("🔮 FAZER PREDIÇÃO", use_container_width=True)
        
        if predict_button:
            gender_map = {"Masculino": "Male", "Feminino": "Female"}
            yes_no_map = {"Sim": "yes", "Não": "no"}
            caec_calc_map = {"Não": "no", "Às Vezes": "Sometimes", "Frequentemente": "Frequently", "Sempre": "Always"}
            mtrans_map = {"Transporte Público": "Public_Transportation", "Automóvel": "Automobile", "Caminhando": "Walking", "Motocicleta": "Motorbike", "Bicicleta": "Bike"}
            
            dados_paciente = {
                'Gender': gender_map[gender],
                'Age': age,
                'family_history': yes_no_map[family_history],
                'FAVC': yes_no_map[favc],
                'FCVC': fcvc,
                'NCP': ncp,
                'CAEC': caec_calc_map[caec],
                'SMOKE': yes_no_map[smoke],
                'CH2O': ch2o,
                'SCC': yes_no_map[scc],
                'FAF': faf,
                'TUE': tue,
                'CALC': caec_calc_map[calc],
                'MTRANS': mtrans_map[mtrans]
            }
            
            with st.spinner('Analisando dados...'):
                resultado = fazer_predicao(dados_paciente, model_data)
            
            if resultado:
                st.markdown("---")
                st.markdown("## 📊 Resultado da Predição")
                
                col_res1, col_res2 = st.columns(2)
                
                with col_res1:
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2 style="color: #1f77b4; margin-bottom: 1rem;">Diagnóstico Previsto</h2>
                        <h1 style="color: #2ca02c; font-size: 2.5rem;">{resultado['classe']}</h1>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_res2:
                    probs_df = pd.DataFrame([{"Classe": k, "Probabilidade": v*100} for k, v in resultado['probabilidades'].items()]).sort_values("Probabilidade", ascending=True)
                    fig = px.bar(probs_df, x="Probabilidade", y="Classe", orientation='h', title="Probabilidades por Classe", color="Probabilidade", color_continuous_scale="Blues")
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### 📈 Todas as Probabilidades")
                probs_sorted = sorted(resultado['probabilidades'].items(), key=lambda x: x[1], reverse=True)
                cols = st.columns(len(probs_sorted))
                for idx, (classe, prob) in enumerate(probs_sorted):
                    with cols[idx]:
                        st.metric(classe, f"{prob*100:.1f}%")
                
                st.markdown("---")
                st.markdown("### 💡 Recomendações")
                
                recomendacoes = []
                if faf < 1:
                    recomendacoes.append("🏃 **Atividade Física:** Aumentar para pelo menos 150 minutos por semana")
                if fcvc < 2:
                    recomendacoes.append("🥗 **Vegetais:** Aumentar consumo para pelo menos 5 porções ao dia")
                if ch2o < 1.5:
                    recomendacoes.append("💧 **Hidratação:** Aumentar ingestão de água para pelo menos 2L por dia")
                if scc == "no":
                    recomendacoes.append("📱 **Monitoramento:** Considere usar app de monitoramento nutricional")
                if favc == "yes":
                    recomendacoes.append("🍔 **Alimentação:** Reduzir consumo de alimentos processados e altamente calóricos")
                if calc in ["Frequently", "Always"]:
                    recomendacoes.append("🍷 **Álcool:** Reduzir consumo de bebidas alcoólicas")
                if smoke == "yes":
                    recomendacoes.append("🚭 **Tabagismo:** Considere programa de cessação do tabagismo")
                
                if recomendacoes:
                    for rec in recomendacoes:
                        st.markdown(f"- {rec}")
                else:
                    st.success("✅ Seus hábitos estão adequados! Continue assim!")
    
    # ==================== TAB 2: DASHBOARD ANALÍTICO ====================
    with tab2:
        df_dashboard = carregar_dados_dashboard()
        
        if df_dashboard is not None:
            # Renderizar dashboard com todas as 6 visualizações
            dv.render_dashboard(df_dashboard)
        else:
            st.warning("⚠️ Não foi possível carregar os dados para visualização")


if __name__ == "__main__":
    main()
