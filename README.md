# Detecção de Furto de Energia com Redes Neurais Híbridas (CNN-VQC)

Este repositório contém a implementação oficial da pesquisa comparativa entre Redes Neurais Convolucionais Clássicas e uma abordagem Híbrida Quântica para detecção de anomalias em dados desbalanceados de consumo de energia (Dataset SGCC).

## 🎯 Objetivo
Avaliar se a utilização de **Quantum Machine Learning (VQC)** como classificador final permite manter alta performance (AUC) reduzindo a necessidade de técnicas agressivas de balanceamento de dados (como ROS/SMOTE), economizando memória computacional.

## 🏗️ Arquitetura
O projeto compara dois cenários:
1. **Baseline:** CNN Clássica (Baseada em Pereira & Saraiva, 2021) com e sem Oversampling.
2. **Proposta:** CNN (Extrator de Features) + VQC (Classificador Variacional Quântico).

## 🚀 Como Rodar

1. Clone o repositório:
   ```bash
   git clone [https://github.com/alanveloso/electricity-theft-hybrid-qml.git](https://github.com/alanveloso/electricity-theft-hybrid-qml.git)