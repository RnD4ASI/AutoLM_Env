# AutoLM

## Project Structure

```
AutoLM/
├── README.md
├── requirement.txt
├── .env
├── .gitignore
├── config/
├── db/
│   ├── graph/
│   ├── preprocess/
│   ├── raw/
│   └── vector/
├── doc/
│   ├── design/
│   └── method/
│       └── architecture.md
├── interface/
│   ├── kb.sh
│   └── llm.sh
├── model/
├── src/
│   ├── pipeline/
│   │   ├── processing/
│   │   │   ├── evaluator.py
│   │   │   ├── executor.py
│   │   │   ├── generator.py
│   │   │   ├── retriever.py
│   │   │   └── topologist.py
│   │   ├── orchestration/
│   │   │   ├── plan_service.py
│   │   │   ├── execution_service.py
│   │   │   └── knowledge_service.py
│   │   └── utility/
│   │       ├── logging.py
│   │       └── utility.py
│   └── optimisation/
│       ├── generic/
│       └── reinforcement/
└── test/
```
