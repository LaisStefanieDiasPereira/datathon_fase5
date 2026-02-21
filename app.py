
import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Risco de Defasagem - Passos Mágicos", layout="wide")

model = joblib.load("models/modelo_risco_defasagem_mlp.joblib")
scaler = joblib.load("models/scaler.joblib")

st.title("🎓 Modelo Preditivo de Risco de Defasagem")
st.markdown("Aplicação desenvolvida para a Associação Passos Mágicos")

st.sidebar.header("Inserir Indicadores do Aluno")

ida = st.sidebar.slider("IDA (Ano anterior)", 0.0, 10.0, 6.0)
ieg = st.sidebar.slider("IEG (Ano anterior)", 0.0, 10.0, 6.0)
ips = st.sidebar.slider("IPS (Ano anterior)", 0.0, 10.0, 6.0)
ipp = st.sidebar.slider("IPP (Ano anterior)", 0.0, 10.0, 6.0)
iaa = st.sidebar.slider("IAA (Ano anterior)", 0.0, 10.0, 6.0)

input_data = np.array([[ida, ieg, ips, ipp, iaa]])
input_scaled = scaler.transform(input_data)

prob = model.predict_proba(input_scaled)[0][1]

st.subheader("📊 Resultado")
st.metric("Probabilidade de Risco", f"{prob:.2%}")

if prob >= 0.7:
    st.error("🔴 Alto Risco – Intervenção Imediata Recomendada")
elif prob >= 0.4:
    st.warning("🟡 Risco Moderado – Monitoramento Ativo")
else:
    st.success("🟢 Baixo Risco – Acompanhamento Regular")

st.markdown("""
### Interpretação

O modelo utiliza indicadores históricos para prever risco futuro.
Use como ferramenta de apoio à decisão pedagógica.
""")
