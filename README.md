# API Previsão de preços

Este é um projeto de desenvolvimento de API com Flask para previsão de preços de ativos listados na bolsa.

## 📖 Descrição
A **API** é uma aplicação Python que possui um endpoint para acesso a um modelo de previsão de valores.

## 🚀 Funcionalidades

- Recebe dados de entrada via requisição HTTP (`POST`) com valores históricos.
- Carrega o modelo e o scaler correspondentes ao ticker solicitado.
- Preprocessa os dados de entrada e realiza a previsão.
- Retorna a previsão em formato JSON.
- Tratamento de erros para entradas inválidas ou tickers não encontrados.


## 📁 Estrutura do Projeto

```bash
API_Previsao/
├── models/
│   ├── ABEV3.SA
│   ├── ITUB4.SA
│   └── PETR4.SA
├── app.py
├── Dockerfile
├── PostAPI.txt                       
├── requirements.txt              
└── README.md                     # Este arquivo
```

## 🛠️ Como Executar o Projeto

## Tecnologias

- **Python 3.10+**
- **Flask** - Framework web
- **TensorFlow / Keras** - Treinamento e inferência dos modelos
- **Joblib** - Para carregar os scalers
- **Pandas / NumPy** - Manipulação de dados

---

## Instalação

```bash
git clone https://github.com/seu-usuario/TC4_project.git
cd TC4_project

python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

## Uso

```bash
python app.py
```

```json
{
  "ticker": "TICKER1",
  "values": [1.2, 1.3, 1.5, ...]  # quantidade deve ser igual a TIME_STEPS definido
}
```

## Retorno

```json
{
  "prediction": [1.45]
}
```
