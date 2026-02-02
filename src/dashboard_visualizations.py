"""
Dashboard Analítico - Visualizações Avançadas
Código modular para integração com aplicativo Streamlit existente
Assume que o DataFrame 'df' já existe e está carregado
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


# Dicionários de tradução
TRADUCOES = {
    # Gênero
    'Male': 'Masculino',
    'Female': 'Feminino',
    
    # Transporte
    'Public_Transportation': 'Transporte Público',
    'Automobile': 'Automóvel',
    'Walking': 'Caminhando',
    'Motorbike': 'Motocicleta',
    'Bike': 'Bicicleta',
    
    # Níveis de Obesidade
    'Insufficient_Weight': 'Peso Insuficiente',
    'Normal_Weight': 'Peso Normal',
    'Overweight_Level_I': 'Sobrepeso Nível I',
    'Overweight_Level_II': 'Sobrepeso Nível II',
    'Obesity_Type_I': 'Obesidade Tipo I',
    'Obesity_Type_II': 'Obesidade Tipo II',
    'Obesity_Type_III': 'Obesidade Tipo III',
    
    # Sim/Não
    'yes': 'Sim',
    'no': 'Não',
    
    # Frequência
    'Sometimes': 'Às vezes',
    'Frequently': 'Frequentemente',
    'Always': 'Sempre',
    'no': 'Não'
}


def traduzir_dataframe(df):
    """Traduz os valores do DataFrame de inglês para português"""
    df_trad = df.copy()
    
    # Traduzir colunas categóricas
    colunas_para_traduzir = ['Gender', 'MTRANS', 'Obesity', 'family_history', 'FAVC', 'SMOKE', 'CAEC', 'CALC', 'SCC']
    
    for col in colunas_para_traduzir:
        if col in df_trad.columns:
            df_trad[col] = df_trad[col].map(lambda x: TRADUCOES.get(x, x))
    
    return df_trad


# ==================== GRÁFICOS BÁSICOS - VISÃO GERAL ====================

def create_gender_distribution(df):
    """Gráfico de Pizza - Distribuição por Gênero"""
    gender_counts = df['Gender'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=gender_counts.index,
        values=gender_counts.values,
        hole=0.4,
        marker=dict(colors=['#00D9FF', '#FF6B9D']),
        textinfo='label+percent',
        textfont=dict(size=14)
    )])
    
    fig.update_layout(
        title={
            'text': '👥 Distribuição por Gênero',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2C3E50'}
        },
        height=400,
        showlegend=True,
        template='plotly_white'
    )
    
    return fig


def create_obesity_distribution(df):
    """Gráfico de Barras - Distribuição de Níveis de Obesidade"""
    obesity_counts = df['Obesity'].value_counts().sort_index()
    
    fig = go.Figure(data=[go.Bar(
        x=obesity_counts.index,
        y=obesity_counts.values,
        marker=dict(
            color=obesity_counts.values,
            colorscale='RdYlGn_r',
            showscale=False
        ),
        text=obesity_counts.values,
        textposition='outside',
        textfont=dict(size=12)
    )])
    
    fig.update_layout(
        title={
            'text': '⚖️ Distribuição de Níveis de Obesidade',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2C3E50'}
        },
        xaxis_title='Nível de Obesidade',
        yaxis_title='Quantidade de Pessoas',
        height=450,
        xaxis=dict(tickangle=-45),
        template='plotly_white'
    )
    
    return fig


def create_gender_obesity_comparison(df):
    """Gráfico de Barras Agrupadas - Homens vs Mulheres por Nível de Obesidade"""
    gender_obesity = df.groupby(['Obesity', 'Gender']).size().reset_index(name='count')
    
    fig = px.bar(
        gender_obesity,
        x='Obesity',
        y='count',
        color='Gender',
        barmode='group',
        title='👫 Comparação: Homens vs Mulheres por Nível de Obesidade',
        labels={'count': 'Quantidade', 'Obesity': 'Nível de Obesidade', 'Gender': 'Gênero'},
        color_discrete_map={'Masculino': '#00D9FF', 'Feminino': '#FF6B9D'},
        text='count'
    )
    
    fig.update_traces(textposition='outside', textfont=dict(size=11))
    
    fig.update_layout(
        height=500,
        xaxis=dict(tickangle=-45),
        title={
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2C3E50'}
        },
        template='plotly_white'
    )
    
    return fig


def create_age_distribution(df):
    """Histograma - Distribuição de Idade"""
    fig = px.histogram(
        df,
        x='Age',
        nbins=30,
        title='📅 Distribuição de Idade na População',
        labels={'Age': 'Idade (anos)', 'count': 'Frequência'},
        color_discrete_sequence=['#1f77b4']
    )
    
    fig.update_traces(marker=dict(line=dict(width=1, color='white')))
    
    fig.update_layout(
        height=400,
        title={
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2C3E50'}
        },
        xaxis_title='Idade (anos)',
        yaxis_title='Frequência',
        showlegend=False,
        template='plotly_white'
    )
    
    return fig


def create_bmi_distribution(df):
    """Gráfico de densidade - Distribuição de IMC por Nível de Obesidade"""
    # Calcular IMC se não existir
    if 'BMI' not in df.columns:
        df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    
    fig = px.violin(
        df,
        y='BMI',
        x='Obesity',
        color='Obesity',
        box=True,
        title='📊 Distribuição de IMC por Nível de Obesidade',
        labels={'BMI': 'IMC (kg/m²)', 'Obesity': 'Nível de Obesidade'},
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(
        height=500,
        xaxis=dict(tickangle=-45),
        title={
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2C3E50'}
        },
        showlegend=False,
        template='plotly_white'
    )
    
    return fig


# ==================== GRÁFICOS AVANÇADOS ====================




def create_radar_chart(df):
    """
    Gráfico de Radar comparando hábitos entre Peso Normal e Obesidade Tipo III
    """
    # Filtrar dados para comparação (usar nomes traduzidos)
    normal_weight = df[df['Obesity'] == 'Peso Normal']
    obesity_type3 = df[df['Obesity'] == 'Obesidade Tipo III']
    
    # Calcular médias dos hábitos
    habits = ['FCVC', 'CH2O', 'FAF', 'TUE']
    habit_labels = ['Consumo de Vegetais', 'Consumo de Água', 'Atividade Física', 'Tempo de Tela']
    
    normal_means = [normal_weight[h].mean() for h in habits]
    obesity_means = [obesity_type3[h].mean() for h in habits]
    
    # Criar gráfico de radar
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=normal_means,
        theta=habit_labels,
        fill='toself',
        name='Peso Normal',
        line=dict(color='#00D9FF', width=2),
        fillcolor='rgba(0, 217, 255, 0.3)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=obesity_means,
        theta=habit_labels,
        fill='toself',
        name='Obesidade Tipo III',
        line=dict(color='#FF6B9D', width=2),
        fillcolor='rgba(255, 107, 157, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(normal_means), max(obesity_means)) * 1.2],
                gridcolor='rgba(255, 255, 255, 0.2)'
            ),
            bgcolor='rgba(0, 0, 0, 0.05)'
        ),
        showlegend=True,
        title={
            'text': '📊 Comparação de Hábitos de Vida',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2C3E50'}
        },
        height=500,
        template='plotly_white'
    )
    
    return fig


def create_sunburst_chart(df):
    """
    Sunburst Chart: Meio de Transporte -> Gênero -> Nível de Obesidade
    """
    # Preparar dados agregados
    sunburst_data = df.groupby(['MTRANS', 'Gender', 'Obesity']).size().reset_index(name='count')
    
    fig = px.sunburst(
        sunburst_data,
        path=['MTRANS', 'Gender', 'Obesity'],
        values='count',
        color='count',
        color_continuous_scale='Viridis',
        title='🌞 Hierarquia: Transporte → Gênero → Nível de Obesidade'
    )
    
    fig.update_layout(
        height=600,
        title={
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2C3E50'}
        }
    )
    
    return fig


def create_parallel_categories(df):
    """
    Parallel Categories: Histórico Familiar -> Alto Calórico -> Fumante -> Diagnóstico
    """
    # Preparar dados categóricos
    df_cat = df.copy()
    
    # Criar dimensões para parallel categories (dados já traduzidos)
    dimensions = [
        dict(
            label="Histórico Familiar",
            values=df_cat['family_history']
        ),
        dict(
            label="Alimentos Calóricos",
            values=df_cat['FAVC']
        ),
        dict(
            label="Fumante",
            values=df_cat['SMOKE']
        ),
        dict(
            label="Diagnóstico",
            values=df_cat['Obesity']
        )
    ]
    
    # Criar colormap para diagnóstico (usar nomes traduzidos)
    color_map = {
        'Peso Insuficiente': 0,
        'Peso Normal': 1,
        'Sobrepeso Nível I': 2,
        'Sobrepeso Nível II': 3,
        'Obesidade Tipo I': 4,
        'Obesidade Tipo II': 5,
        'Obesidade Tipo III': 6
    }
    df_cat['color_val'] = df_cat['Obesity'].map(color_map)
    
    fig = go.Figure(data=[go.Parcats(
        dimensions=dimensions,
        line=dict(
            color=df_cat['color_val'],
            colorscale='Portland',
            shape='hspline'
        ),
        hoveron='color',
        hoverinfo='count+probability',
        labelfont=dict(size=14, family='Arial'),
        arrangement='freeform'
    )])
    
    fig.update_layout(
        title={
            'text': '🔄 Fluxo: Genética → Hábitos → Diagnóstico',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2C3E50'}
        },
        height=600,
        template='plotly_white'
    )
    
    return fig


def create_3d_scatter(df):
    """
    Scatter Plot 3D: Peso (X), Altura (Y), Idade (Z) coloridos por Nível de Obesidade
    """
    fig = px.scatter_3d(
        df,
        x='Weight',
        y='Height',
        z='Age',
        color='Obesity',
        title='📍 Distribuição 3D: Peso × Altura × Idade',
        labels={
            'Weight': 'Peso (kg)',
            'Height': 'Altura (m)',
            'Age': 'Idade (anos)',
            'Obesity': 'Nível de Obesidade'
        },
        color_discrete_sequence=px.colors.qualitative.Vivid,
        opacity=0.7,
        hover_data={'Weight': ':.1f', 'Height': ':.2f', 'Age': True}
    )
    
    fig.update_traces(marker=dict(size=5, line=dict(width=0.5, color='white')))
    
    fig.update_layout(
        height=700,
        title={
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2C3E50'}
        },
        scene=dict(
            xaxis=dict(backgroundcolor='rgba(230, 230, 250, 0.5)', gridcolor='white'),
            yaxis=dict(backgroundcolor='rgba(230, 250, 230, 0.5)', gridcolor='white'),
            zaxis=dict(backgroundcolor='rgba(250, 230, 230, 0.5)', gridcolor='white')
        )
    )
    
    return fig


def create_violin_plot(df):
    """
    Violin Plot: Distribuição da Idade por Nível de Obesidade
    """
    fig = px.violin(
        df,
        y='Age',
        x='Obesity',
        color='Obesity',
        box=True,
        points='outliers',
        title='🎻 Distribuição de Idade por Nível de Obesidade',
        labels={
            'Age': 'Idade (anos)',
            'Obesity': 'Nível de Obesidade'
        },
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    fig.update_traces(
        meanline_visible=True,
        width=0.8
    )
    
    fig.update_layout(
        height=600,
        title={
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2C3E50'}
        },
        xaxis=dict(tickangle=-45),
        showlegend=False,
        template='plotly_white'
    )
    
    return fig


def create_correlation_heatmap(df):
    """
    Mapa de Calor de Correlação - Apenas variáveis numéricas
    """
    # Selecionar apenas colunas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Calcular matriz de correlação
    corr_matrix = df[numeric_cols].corr()
    
    # Dicionário de tradução para nomes de colunas
    col_traducoes = {
        'Age': 'Idade',
        'Height': 'Altura',
        'Weight': 'Peso',
        'FCVC': 'Consumo Vegetais',
        'NCP': 'Nº Refeições',
        'CH2O': 'Consumo Água',
        'FAF': 'Atividade Física',
        'TUE': 'Tempo Tela'
    }
    
    # Renomear colunas e índices
    corr_matrix = corr_matrix.rename(columns=col_traducoes, index=col_traducoes)
    
    # Criar heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlação")
    ))
    
    fig.update_layout(
        title={
            'text': '🔥 Mapa de Calor - Correlação entre Variáveis Numéricas',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2C3E50'}
        },
        height=700,
        xaxis=dict(tickangle=-45),
        template='plotly_white'
    )
    
    return fig


def render_dashboard(df):
    """
    Função principal para renderizar o Dashboard Analítico
    Integre esta função no seu app Streamlit existente
    """
    # Traduzir dados para português
    df = traduzir_dataframe(df)
    
    st.title("📊 Dashboard Analítico - Análise de Risco de Obesidade")
    st.markdown("---")
    
    # Criar abas para organizar visualizações (do mais simples ao mais complexo)
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Estatísticas Básicas", 
        "🎯 Distribuições", 
        "💪 Hábitos & Comportamento", 
        "🧬 Análise Multidimensional"
    ])
    
    # ==================== TAB 1: ESTATÍSTICAS BÁSICAS ====================
    with tab1:
        st.subheader("Visão Geral da População")
        
        # KPIs principais
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("👥 Total de Pessoas", len(df))
        with col2:
            idade_media = df['Age'].mean()
            st.metric("📅 Idade Média", f"{idade_media:.1f} anos")
        with col3:
            masculino_pct = (df['Gender'] == 'Masculino').sum() / len(df) * 100
            st.metric("👨 Homens", f"{masculino_pct:.1f}%")
        with col4:
            obesos = df[df['Obesity'].str.contains('Obesidade', na=False)].shape[0]
            obesos_pct = obesos / len(df) * 100
            st.metric("⚠️ Obesidade", f"{obesos_pct:.1f}%")
        
        st.markdown("---")
        
        # Gráficos básicos
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribuição por Gênero
            fig_gender = create_gender_distribution(df)
            st.plotly_chart(fig_gender, use_container_width=True)
            
            # Distribuição de Idade
            fig_age = create_age_distribution(df)
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            # Distribuição de Níveis de Obesidade
            fig_obesity = create_obesity_distribution(df)
            st.plotly_chart(fig_obesity, use_container_width=True)
        
        # Comparação Homens vs Mulheres (largura completa)
        st.markdown("### 👫 Análise por Gênero")
        fig_gender_obesity = create_gender_obesity_comparison(df)
        st.plotly_chart(fig_gender_obesity, use_container_width=True)
    
    # ==================== TAB 2: DISTRIBUIÇÕES ====================
    with tab2:
        st.subheader("Distribuições e Correlações")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Violin Plot
            fig_violin = create_violin_plot(df)
            st.plotly_chart(fig_violin, use_container_width=True)
        
        with col2:
            # 3D Scatter
            fig_3d = create_3d_scatter(df)
            st.plotly_chart(fig_3d, use_container_width=True)
        
        # IMC Distribution
        st.markdown("### 📊 Análise de IMC")
        fig_bmi = create_bmi_distribution(df)
        st.plotly_chart(fig_bmi, use_container_width=True)
        
        # Correlation Heatmap (largura completa)
        st.markdown("### 🔍 Análise de Correlação")
        fig_corr = create_correlation_heatmap(df)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # ==================== TAB 3: HÁBITOS & COMPORTAMENTO ====================
    with tab3:
        st.subheader("Perfis de Hábitos e Fluxos Comportamentais")
        
        # Radar Chart
        fig_radar = create_radar_chart(df)
        st.plotly_chart(fig_radar, use_container_width=True)
        
        st.markdown("---")
        
        # Parallel Categories
        fig_parallel = create_parallel_categories(df)
        st.plotly_chart(fig_parallel, use_container_width=True)
    
    # ==================== TAB 4: ANÁLISE MULTIDIMENSIONAL ====================
    with tab4:
        st.subheader("Análise Hierárquica e Segmentação")
        
        # Sunburst Chart
        fig_sunburst = create_sunburst_chart(df)
        st.plotly_chart(fig_sunburst, use_container_width=True)
        
        # Insights
        st.markdown("---")
        st.markdown("""
        ### 💡 Insights Principais
        
        - **Estatísticas Básicas**: Visão rápida dos principais indicadores da população
        - **Distribuições**: Entenda como idade, IMC e obesidade se distribuem nos dados
        - **Radar Chart**: Compare diretamente os padrões de hábitos saudáveis entre grupos extremos
        - **Sunburst**: Identifique nichos de risco (ex: sedentários + gênero específico)
        - **Parallel Categories**: Visualize o impacto cascata de fatores genéticos e comportamentais
        - **3D Scatter**: Explore relações espaciais entre medidas físicas e diagnóstico
        - **Heatmap**: Identifique quais variáveis tem maior correlação com o peso
        """)
