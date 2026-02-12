# 🔬 Classificação Quântica Variacional com Data Re-uploading

> **Explorando o poder expressivo dos circuitos quânticos através da técnica de Data Re-uploading**

Este projeto investiga o impacto da técnica de **Data Re-uploading** em classificadores quânticos variacionais (VQC) aplicados a diferentes níveis de complexidade de dados: desde problemas linearmente separáveis até fronteiras de decisão altamente não-lineares.

---

## 📋 Sumário

- [Visão Geral](#-visão-geral)
- [O que é Data Re-uploading?](#-o-que-é-data-re-uploading)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Datasets e Resultados](#-datasets-e-resultados)
- [Arquitetura dos Circuitos](#-arquitetura-dos-circuitos)
- [Instalação e Execução](#-instalação-e-execução)
- [Conclusões](#-conclusões)
- [Referências](#-referências)

---

## 🎯 Visão Geral

O objetivo deste projeto é demonstrar empiricamente como a técnica de **Data Re-uploading** aumenta a expressividade de circuitos quânticos variacionais, permitindo que eles aprendam fronteiras de decisão mais complexas.

### Por que isso importa?

Circuitos quânticos com encoding tradicional (dados inseridos uma única vez) têm limitações na representação de funções não-lineares. O Data Re-uploading supera essa limitação ao re-encodar os dados clássicos em múltiplas camadas do circuito, funcionando de forma análoga às camadas ocultas de uma rede neural clássica.

---

## 🔄 O que é Data Re-uploading?

### Conceito Fundamental

O **Data Re-uploading** é uma técnica proposta por Pérez-Salinas et al. (2020) que permite que um circuito quântico atue como um aproximador universal de funções. A ideia central é simples, mas poderosa:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENCODING TRADICIONAL                         │
│                                                                 │
│   |0⟩ ──[Encoding(x)]──[Layer 1]──[Layer 2]──...──[Medição]    │
│                 ↑                                               │
│           Dados entram                                          │
│           APENAS AQUI                                           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    DATA RE-UPLOADING                            │
│                                                                 │
│   |0⟩ ──[Enc(x)]──[L1]──[Enc(x)]──[L2]──[Enc(x)]──...──[Med]   │
│            ↑              ↑              ↑                      │
│         Dados          Dados          Dados                     │
│      re-encodados   re-encodados   re-encodados                 │
│      em CADA camada!                                            │
└─────────────────────────────────────────────────────────────────┘
```

### Por que funciona?

1. **Maior Expressividade**: Cada re-encoding cria uma nova "camada" de não-linearidade
2. **Analogia com Redes Neurais**: Similar a ter múltiplas camadas ocultas
3. **Aproximador Universal**: Com camadas suficientes, pode aproximar qualquer função contínua

### Implementação no Código

**Com Re-uploading** (dados dentro do loop):
```python
@qml.qnode(dev)
def circuito(weights, x):
    for i, layer_w in enumerate(weights):
        qml.AngleEmbedding(features=x, wires=range(n_qubits), rotation='Z')  # ← RE-ENCODING
        layer(layer_w)
    return qml.expval(qml.PauliZ(0))
```

**Sem Re-uploading** (dados fora do loop):
```python
@qml.qnode(dev)
def circuito(weights, x):
    qml.AngleEmbedding(features=x, wires=range(n_qubits), rotation='Z')  # ← ENCODING ÚNICO
    for i, layer_w in enumerate(weights):
        layer(layer_w)
    return qml.expval(qml.PauliZ(0))
```

---

## 📁 Estrutura do Projeto

```
Laciq_PS/
│
├── 📂 Baseline - Blobs/
│   └── Blobs.ipynb              # Baseline: problema linearmente separável
│
├── 📂 Moons/
│   ├── moons_CRecupload.ipynb   # Moons COM Data Re-uploading
│   └── moonS_SReupload.ipynb    # Moons SEM Data Re-uploading
│
├── 📂 Iris/
│   ├── iris_CReupload.ipynb     # Iris COM Data Re-uploading
│   └── iris_SReupload.ipynb     # Iris SEM Data Re-uploading
│
└── README.md
```

---

## 📊 Datasets e Resultados

### 1️⃣ Blobs (Baseline)

| Característica | Valor |
|----------------|-------|
| **Complexidade** | Linear |
| **Amostras** | 500 |
| **Classes** | 2 |
| **Qubits** | 1 |
| **Camadas** | 1 |
![Blobs](Result\blobs.png)
> 💡 **Insight**: Por ser linearmente separável, não há necessidade de circuitos complexos nem Data Re-uploading. Serve como baseline para validar a implementação.

---

### 2️⃣ Moons (Não-Linear)

| Métrica | Sem Re-uploading | Com Re-uploading |
|---------|------------------|------------------|
| **Qubits** | 2 | 2 |
| **Camadas** | 6 | 6 |
| **Convergência** | Mais lenta | Mais rápida |
| **Acurácia Final** | ~80-90% | **~100%** |
![Moons](Result\moons.png)

> 🔥 **Resultado chave**: O Data Re-uploading permite que o modelo alcance **100% de acurácia** em problemas com fronteiras não-lineares como o Moons!

---

### 3️⃣ Iris (Multiclasse)

| Métrica | Sem Re-uploading | Com Re-uploading |
|---------|------------------|------------------|
| **Qubits** | 4 | 4 |
| **Camadas** | 8 | 8 |
| **Classes** | 3 | 3 |
| **Learning Rate** | 0.04 | 0.0004 |
| **Framework** | PyTorch | PyTorch |

> 🧠 **Estratégia Multiclasse**: Utilizamos 3 "sub-classificadores" quânticos, cada um com pesos específicos para uma classe. A predição final é o argmax das 3 saídas.

```python
# Estrutura de pesos: (n_classes, n_layers, n_qubits, 3)
shape_weights = (3, 8, 4, 3)  # 3 classificadores × 8 camadas × 4 qubits × 3 parâmetros
```

---

## 🏗️ Arquitetura dos Circuitos

### Componentes Principais

1. **Encoding**: `AngleEmbedding` com rotação Z
2. **Ansatz**: `StronglyEntanglingLayers` (rotações + CNOTs)
3. **Medição**: Valor esperado de PauliZ

### Visualização do Circuito (Moons com Re-uploading)

```
     ┌────────┐ ┌────────────────────┐ ┌────────┐ ┌────────────────────┐
q0: ─┤ RZ(x₀) ├─┤                    ├─┤ RZ(x₀) ├─┤                    ├─ ... ─┤ ⟨Z⊗Z⟩
     └────────┘ │  StronglyEntangling│ └────────┘ │  StronglyEntangling│
     ┌────────┐ │       Layer        │ ┌────────┐ │       Layer        │
q1: ─┤ RZ(x₁) ├─┤                    ├─┤ RZ(x₁) ├─┤                    ├─ ... ─┤
     └────────┘ └────────────────────┘ └────────┘ └────────────────────┘
     
     ╰──────── Camada 1 ─────────────╯ ╰──────── Camada 2 ─────────────╯
```

---

## 💻 Instalação e Execução

### Requisitos

```bash
pip install pennylane pennylane-numpy torch scikit-learn matplotlib seaborn tqdm
```

### Execução

1. Clone ou baixe o repositório
2. Abra os notebooks no Jupyter ou VS Code
3. Execute as células sequencialmente

### Ordem Recomendada

1. **Blobs.ipynb** - Entender o baseline
2. **moons_CRecupload.ipynb** - Ver o Re-uploading em ação
3. **moonS_SReupload.ipynb** - Comparar sem Re-uploading
4. **iris_CReupload.ipynb** - Problema multiclasse com Re-uploading
5. **iris_SReupload.ipynb** - Comparar sem Re-uploading

---

## 🎓 Conclusões

### Principais Descobertas

1. **O Data Re-uploading é essencial para problemas não-lineares**
   - Sem ele, o modelo fica limitado a fronteiras de decisão simples
   - Com ele, conseguimos separar datasets como Moons com 100% de acurácia

2. **Trade-off: Expressividade vs Complexidade**
   - Mais re-encodings = mais expressivo, mas mais custoso computacionalmente
   - É preciso balancear o número de camadas com o tempo de treinamento

3. **Normalização é crítica**
   - Para `AngleEmbedding`, escalar os dados para `[0, π]` melhorou significativamente a convergência
   - Dados fora dessa faixa limitam as rotações do circuito

4. **PyTorch facilita problemas multiclasse**
   - A integração PennyLane + PyTorch permite usar `CrossEntropyLoss` e otimizadores sofisticados

### Comparação Visual

```
                    SEM RE-UPLOADING              COM RE-UPLOADING
                    
Expressividade:     ████░░░░░░ (40%)             ██████████ (100%)

Fronteiras:         Lineares/Simples             Altamente não-lineares

Moons Accuracy:     ~80-90%                      ~100%

Convergência:       Instável                     Estável e rápida
```

## 👨‍💻 Autor

Desenvolvido como parte do processo seletivo LACIQ.

---