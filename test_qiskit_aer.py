#!/usr/bin/env python3
"""
Teste local do simulador qiskit.aer (Qiskit Aer / stack IBM).

Roda 1 época do modelo híbrido com dados aleatórios (não precisa do dataset SGCC).
Use: pip install pennylane-qiskit  (além dos requirements do projeto)
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

def main():
    print("Configurando device qiskit.aer...")
    try:
        from src.models.hybrid import set_quantum_device, build_hybrid_model
        set_quantum_device(device="qiskit.aer")
    except Exception as e:
        print("ERRO:", e)
        print("Instale o plugin: pip install pennylane-qiskit")
        return 1

    print("Construindo modelo híbrido e rodando forward pass...")
    import numpy as np

    model = build_hybrid_model(input_shape=(148, 7, 1))
    X = np.random.randn(4, 148, 7, 1).astype(np.float32) * 0.1
    pred = model.predict(X, verbose=0)
    assert pred.shape == (4, 1), pred.shape
    print("OK — simulador qiskit.aer funcionando neste computador (forward pass).")
    return 0

if __name__ == "__main__":
    sys.exit(main())
