# Projeto

!!! abstract "Enunciados"

    [Projects](https://insper.github.io/ann-dl/){:target='_blank'}

O projeto é **um só**, feito em equipe sobre **o mesmo dataset**, e entregue em três partes
ao longo do semestre, cada uma com data e peso próprios.

## Equipe

| Nome completo | E-mail | GitHub |
|---------------|--------|--------|
| Tiago Liberman | tiago@liberman.com.br | [tiliberman](https://github.com/tiliberman) |
| Júlia Campos | juliacamposmagalhaes@yahoo.com.br | [julia-campos-ma](https://github.com/julia-campos-ma) |

Os nomes se repetem no cabeçalho de cada entrega, para que cada página seja atribuível
sozinha.

## As três entregas

| # | Entrega | Página |
|---|---------|--------|
| 1 | EDA | [EDA](eda/index.md) |
| 2 | Classificação | [Classificação](classification/index.md) |
| 3 | Generativo | [Generativo](generative/index.md) |

Datas e pesos são da edição — veja o
[overview](https://insper.github.io/ann-dl/){:target='_blank'}.

!!! danger "A nota do projeto costuma ser limitada por uma prova sobre o próprio projeto"

    Os relatórios foram escritos para serem defendidos: cada decisão aponta para o número ou
    a figura que a motivou, e o [registro de decisões](#registro-de-decisoes) abaixo guarda o
    raciocínio que atravessa as entregas.

## Dataset

O mesmo dataset atravessa as três entregas — escolhê-lo bem no EDA é o que torna as outras
duas viáveis.

| | |
|---|---|
| **Nome** | Hotel Booking Demand (Antonio, de Almeida & Nunes, 2019) |
| **Fonte (URL)** | <https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand> · artigo: <https://doi.org/10.1016/j.dib.2018.11.126> |
| **Licença / termos de uso** | CC BY 4.0 (artigo em *Data in Brief*); cópia pública idêntica no TidyTuesday 2020-02-11 |
| **Amostras** | 119.390 reservas no arquivo bruto; 119.210 após remover reservas sem hóspedes |
| **Features** | 32 colunas no bruto; 26 usadas (14 numéricas, 12 categóricas) após descartar o alvo, 4 colunas de vazamento e o ano |
| **Variável alvo** | `is_canceled` (1 = reserva cancelada ou *no-show*), 37,1% das reservas |
| **Tarefa escolhida** | Classificação (binária) |

Reservas reais de dois hotéis portugueses (um *resort* no Algarve, um hotel urbano em Lisboa)
entre julho de 2015 e agosto de 2017, documentadas variável a variável num artigo revisado
por pares. Cancelamentos são um problema de negócio concreto: obrigam o hotel a fazer
*overbooking* e custam receita. A tarefa não é trivial porque o sinal está espalhado entre
features numéricas (antecedência, diária, histórico do cliente) e categóricas (país, segmento,
tipo de depósito, agência), com 27% de linhas duplicadas, na maioria por reservas de grupo, colunas que
vazam o desfecho e deriva temporal na taxa de cancelamento — tudo documentado no
[EDA](eda/index.md).

## Status

- [x] **1. EDA**
- [ ] **2. Classificação**
- [ ] **3. Generativo**

## Registro de decisões

| Data | Decisão | Motivo |
|------|---------|--------|
| 2026-09-17 | Proposta com um único dataset: Hotel Booking Demand | Como alunos de Administração, buscamos um problema de negócio concreto; entre os candidatos testados com apoio do Claude, foi o único com impacto financeiro claro (cancelamentos e *overbooking*), material para todos os itens do roteiro de EDA, dicionário publicado e tamanho compatível com o repositório |
| 2026-10-06 | Tarefa: **classificação** binária de `is_canceled`; pasta `regression/` abandonada | O alvo natural do dataset é binário; a alternativa (regredir `adr`) teria um alvo com 1,7% de zeros e um erro de digitação em 5.400 |
| 2026-10-06 | Split **temporal** (os últimos 20% das chegadas = teste, corte em 2017-04-22), definido uma vez em `eda/code/data.py` e reutilizado nas entregas seguintes | 26,8% de linhas duplicadas por grupos: um split aleatório poria cópias idênticas nos dois lados (33% do teste); e o uso real é prever o futuro |
| 2026-10-06 | Duplicatas **mantidas** | São reservas de grupo canceladas em bloco (58% de cancelamento dentro delas); removê-las mudaria o alvo de 37% para 27% |
| 2026-10-06 | Descartadas `reservation_status`, `reservation_status_date`, `assigned_room_type`, `required_car_parking_spaces` (vazamento) e `arrival_date_year` (proxy do split) | Evidências na seção 1B do EDA: as duas primeiras são o desfecho; as outras duas se comportam como preenchidas só para quem chegou (5,4% e 0,0% de cancelamento) |
| 2026-10-06 | `booking_changes` mantida, com ressalva | Acumula alterações até o check-in ou cancelamento; a entrega de classificação mede o modelo com e sem ela |
| 2026-10-06 | `previous_cancellations` mantida, mas tratada como o principal risco | 85,7% das linhas com valor > 0 não são de hóspede repetido e a taxa de cancelamento delas cai de 99% (2015) para 17% (2017): parece contar os quartos cancelados de um mesmo bloco de grupo. A classificação treina com e sem ela |
| 2026-10-06 | `country`: `CN` fundido em `CHN`; `adr` recortada mas **não** logaritmizada | Mesmo país em dois códigos ISO; o log de `adr` transformava as 1,7% de diárias zero num pico a seis desvios da média |
