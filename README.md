# Purchase Propensity MLOps

Entrega da Fase 2 do Tech Challenge: um pipeline de MLOps que estima a propensao de compra de visitantes de um e-commerce a partir de seu comportamento de navegacao.

O projeto foi desenhado para que uma nova execucao reproduza o mesmo fluxo de dados, treinamento, avaliacao e registro do modelo. O objetivo de negocio e permitir que marketing e vendas priorizem visitantes com maior probabilidade de concluir uma compra.

## Visao da solucao

```text
Dados de navegacao
       |
       v
DVC: download -> preprocess -> train
       |                         |
       |                         +--> MLflow: experimentos, metricas e Model Registry
       v
Modelo selecionado (ROC-AUC) --> models/best_model.joblib
```

- **Dados:** Online Shoppers Purchasing Intention Dataset, da UCI Machine Learning Repository. Quando a fonte externa estiver indisponivel, a etapa de download cria uma base sintetica deterministica com o mesmo esquema, para manter a demonstracao executavel. O comportamento fica explicito no console.
- **Preparacao:** remove duplicidades, normaliza o alvo `Revenue` para `0/1` e preserva as variaveis de navegacao.
- **Modelagem:** compara regressao logistica e Random Forest; o campeao e selecionado por ROC-AUC de validacao.
- **Rastreabilidade:** MLflow registra parametros, metricas, artefatos e a versao do melhor modelo no Model Registry, com alias `champion`.
- **Reprodutibilidade:** DVC descreve e versiona as etapas do pipeline; `params.yaml` centraliza sementes e hiperparametros.

## Estrutura

```text
src/purchase_propensity/  Codigo de download, preparacao, features e treinamento
tests/                    Testes unitarios
data/                     Dados gerados pelo DVC (nao versionados pelo Git)
models/                   Artefato do modelo selecionado (nao versionado pelo Git)
reports/                  Metricas do treinamento (nao versionadas pelo Git)
params.yaml               Parametros versionados
dvc.yaml                  Pipeline DVC
Dockerfile                Execucao isolada do pipeline
```

## Como executar localmente

Prerequisitos: Python 3.11 ou 3.12, [uv](https://docs.astral.sh/uv/), Git e DVC (instalado pelo ambiente do projeto).

```bash
git clone <URL_DO_REPOSITORIO>
cd tech-challenge-fase-2
uv sync --all-groups
cp .env.example .env
uv run dvc repro
```

Ao final, os principais resultados estarao em:

- `models/best_model.joblib`: pipeline Scikit-Learn vencedor;
- `reports/metrics.json`: ROC-AUC, accuracy, precision, recall e F1;
- `mlruns.db` e `mlruns/`: historico local do MLflow.

Para ver os experimentos e o registro de modelos:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlruns.db
```

Abra `http://127.0.0.1:5000` no navegador e consulte o experimento `purchase-propensity` e o modelo `purchase-propensity-classifier`.

## Testes e qualidade

```bash
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
```

## Reexecucao controlada

Para alterar uma semente, hiperparametro ou a divisao de teste, modifique `params.yaml` e execute novamente:

```bash
uv run dvc repro
```

O DVC identifica apenas as etapas afetadas. Para inspecionar o grafo:

```bash
uv run dvc dag
```

## Container Docker

O container executa o mesmo pipeline DVC e nao inclui dados, modelos ou experimentos locais na imagem. Como o historico Git tambem fica fora da imagem, o DVC e inicializado internamente no modo isolado apenas durante a execucao do container.

```bash
docker build -t purchase-propensity-mlops .
docker run --rm -v "$(pwd)/artifacts:/app/artifacts" purchase-propensity-mlops
```

Para manter os artefatos fora do container, prefira executar o pipeline localmente ou monte os diretorios `data`, `models`, `reports` e `mlruns` como volumes conforme a necessidade do ambiente.

## Tela de demonstracao comercial

Depois do treinamento, uma tela web permite testar a propensao de compra com sinais simplificados de um visitante e receber uma prioridade comercial e acao sugerida.

```bash
podman run --rm -p 18080:8080 \\
  -v "$(pwd)/local-artifacts/models:/app/models" \\
  purchase-propensity-mlops:local \\
  uv run --no-sync serve-demo
```

Abra `http://127.0.0.1:18080`. O mesmo comando pode ser executado com `docker run` em um ambiente com Docker ativo.

## Configuracao

Copie `.env.example` para `.env` antes de executar. Para este projeto, o MLflow usa SQLite local por padrao. Em ambiente corporativo, `MLFLOW_TRACKING_URI` pode apontar para um servidor MLflow centralizado.

## Limites da demonstracao

Esta entrega trata propensao de compra como classificacao binaria e usa dados publicos (ou o fallback sintetico deterministico). Antes de uso comercial real, e necessario validar qualidade, vieses, consentimento de dados e metricas de negocio com dados representativos.
