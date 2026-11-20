# Maria Kodzaeva — HW2 / Session 2: history-weight Encoder

Индивидуальная часть Sparse Signals для ДЗ 8: реальная `Qwen/Qwen2.5-3B-Instruct` меняет относительные веса прошлых Buy-активов, после чего неизменённый CBF ранжирует допустимые активы. Вмешательство применяется к заранее выбранной группе с 4+ различными историческими активами; остальные пользователи сохраняют `buy_count`. Модель получает полную историю, counts и пять категориальных признаков CBF с временными ключами. Прямые ID, названия, анкета и будущие покупки в промпт не входят. Невалидный ответ даёт зарегистрированный fallback к частотам.

Это отдельный компонент; совместная реализация с retrieval/reranking коллег здесь не заявляется. Индивидуальный PDF и подлинный экспорт AI-сессии сдаются отдельно. Публичная папка содержит код и агрегаты; CSV, пользовательские протоколы, контексты, промпты, ответы модели, индивидуальные рекомендации и AI-журналы сюда не включены.

## Фактический результат

Учебный протокол: cutoff `2021-06-01`, окно `[2021-06-01, 2021-07-01)`, одни и те же 100 warm-пользователей, 11 с будущими покупками; повторные покупки разрешены. Empty truth даёт 0 в общей средней. Группа 4+: 37 пользователей, 10 с непустой истиной. Это не финальный benchmark проекта.

| Метод | nDCG@10: все 100 | nDCG@10: непустые 11 |
|---|---:|---:|
| Popularity | 0.0286 | 0.2599 |
| CBF | 0.0201 | 0.1827 |
| Latest buy | 0.0277 | 0.2520 |
| Latest buy or sell | 0.0271 | 0.2464 |
| Qwen Encoder + CBF | 0.0219 | 0.1988 |
| Равные веса, post-hoc | 0.0192 | 0.1747 |
| `sqrt(buy_count)`, post-hoc | 0.0224 | 0.2040 |

LLM улучшила nDCG лишь у одного пользователя. Дешёвый sqrt-контроль оказался выше LLM, а три предоставленных baseline — выше обоих. Устойчивое преимущество LLM этот пилот не подтверждает. Равные веса и sqrt-контроль выбраны **после основного результата** и обозначены post-hoc; параметры по будущим меткам не подбирались. Точные числа, counts, версия модели, настройки, времена и хеши исходных CSV находятся в [results.json](results.json).

Основной прогон выполнил 37 уникальных вызовов, включая 3 пилотных, без invalid/fallback. Генерация заняла 303.62 с; два inference-процесса вместе — 352.01 с, применение ответов и оценка пяти методов — ещё 2.88 с. Четыре baseline вместе заняли 2.66 с; отдельных времён методов нет. Скачивание весов в inference не включено. Это один greedy-проход, а не три стохастических повтора плана HW7.

## Состав и происхождение

- `src/` — точные копии исполненного Encoder, загрузки модели, двух контролей, логгера и синтетических тестов.
- `source/` — неизменённый starter преподавателя из `dryjins/RecSys-LLMs`, commit `799e54c0493cf7e2b014152548ae5c18ee24a8f5`; авторство baseline не приписывается студенту.
- `checks/` — независимый пересчёт метрики/временного протокола и проверка влияния весов. Эти проверки выполняли AI-агенты; это не заявление о личной проверке студентом каждого вычисления.
- `requirements.lock.txt` — точные зависимости исполненного inference-окружения.
- [source_provenance.json](source_provenance.json) — происхождение и SHA-256 точных исходников; `manifest.json` — хеши публичных файлов.

Данные: [официальная FAR-Trans release](https://researchdata.gla.ac.uk/1658/), CC BY 4.0; J. Sanz-Cruzado, N. Droukas, R. McCreadie, *FAR-Trans: An Investment Dataset for Financial Asset Recommendation*, 2024. Модель: [закреплённая revision](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/tree/aa8e72537993ba99e69dfaafa59ed015b17504d1), лицензия `qwen-research`. Покупательская релевантность не является оценкой доходности или пригодности инвестиций.

## Повторение

Команды выполняются **из этой папки `hw2/`**. Предоставленный inference использует Apple Silicon/MPS float16 и явно запрещает CPU fallback. Исполненная среда: Apple M4, 16 ГБ, Python 3.12.14, torch 2.11.0, transformers 4.57.6. Для другой платформы нужен отдельно проверенный перенос inference; classical baseline и проверки используют стандартную библиотеку Python 3.10+.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
mkdir -p data evidence logs
curl --fail --location https://researchdata.gla.ac.uk/1658/1/FAR-Trans.zip --output data/FAR-Trans.zip
.venv/bin/python -m zipfile -e data/FAR-Trans.zip data

PYTHONHASHSEED=0 .venv/bin/python source/starter.py run --data-dir data/FAR-Trans --output-dir outputs/baseline --cutoff 2021-06-01 --horizon-days 30 --max-users 100 --cohort warm --candidate-count 50

# Только публичные веса; скрипт не читает датасет.
.venv/bin/python src/download_model.py
MODEL_DIR=.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1

# Пилот: первый, второй и последний пользователь группы4+ в фиксированном порядке.
.venv/bin/python src/run_logged.py --label llm_pilot -- .venv/bin/python src/history_encoder.py --baseline-dir outputs/baseline --output-dir outputs/llm_pilot --model-path "$MODEL_DIR" --pilot-only
.venv/bin/python src/run_logged.py --label llm_full -- .venv/bin/python src/history_encoder.py --baseline-dir outputs/baseline --output-dir outputs/llm_full --model-path "$MODEL_DIR" --resume-pilot-dir outputs/llm_pilot

# Оценка реальных ответов наряду с теми же четырьмя baseline.
PYTHONHASHSEED=0 .venv/bin/python source/starter.py run --data-dir data/FAR-Trans --output-dir outputs/change --cutoff 2021-06-01 --horizon-days 30 --max-users 100 --cohort warm --candidate-count 50 --responses outputs/llm_full/responses.jsonl --stage encoder --on-invalid fallback

# Дополнительные post-hoc контроли; вызовов LLM нет.
PYTHONHASHSEED=0 .venv/bin/python source/starter.py run --data-dir data/FAR-Trans --output-dir outputs/uniform --plugin src/uniform_encoder.py --on-invalid fallback
PYTHONHASHSEED=0 .venv/bin/python source/starter.py run --data-dir data/FAR-Trans --output-dir outputs/sqrt --plugin src/sqrt_encoder.py --on-invalid fallback

.venv/bin/python -m unittest discover -s source -p test_starter.py
.venv/bin/python -m unittest discover -s src -p test_history_encoder.py
PYTHONHASHSEED=0 .venv/bin/python checks/verify_outputs_independent.py
PYTHONHASHSEED=0 .venv/bin/python checks/check_encoder_effect.py --responses outputs/llm_full/responses.jsonl --change-dir outputs/change
```

Для повторного запуска используйте новые папки и label: код отказывается перезаписывать существующие результаты/receipts. Скачивание требует сети; `run_logged.py` включает offline-настройки для локального inference. Его собственные промпты и ответы остаются в игнорируемых `outputs/`. Классические методы можно воспроизвести без скачивания модели; для воспроизведения LLM из этого публичного пакета нужен новый фактический вызов — частные сохранённые ответы здесь не опубликованы. Небольшие различия при почти равных float-оценках между окружениями возможны.
