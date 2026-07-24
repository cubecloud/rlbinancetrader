# ЗАДАЧА от юзера: закрепить prod-predictions за одной картой 3060 Ti на jarvis

От: rlbinancetrader-сессия, 2026-07-22, ~23:40. Задача поставлена юзером
(Telegram): «сделать lock карты 3060 Ti для контейнера, который занимается
predictions, чтобы остальные GPU были свободны для работы по другим
направлениям». Контур ваш — исполнение за вами; ниже вся разведка, чтобы
не собирать заново.

## Разведка (сделана только что по ssh, read-only)

- Контейнер: `sunday_main` (sunday-main-tf212:latest, `bash ./start.sh`,
  tough_pipe_runner.py). Запущен через `docker run` (НЕ compose),
  `--gpus all` (DeviceRequests Count=-1), NVIDIA_VISIBLE_DEVICES=all,
  restart=always, network host, volumes /data/projects:/projects и
  /data/westmere/wheels:/wheels, workdir /sunday.
- Карты jarvis: GPU0 = RTX 3060 12GB (uuid GPU-6d81ac73-...),
  GPU1 = 3060 Ti 8GB (GPU-a53eeed2-c3af-d78c-f6b7-4999b50c41eb),
  GPU2 = 3060 Ti 8GB (GPU-255eefde-...), GPU3 = 3060 Ti 8GB (GPU-b329bb0a-...).
- Фактическое потребление процесса predictions: GPU1 — 1182 MiB (реальная
  нагрузка), GPU0/2/3 — 104–141 MiB (пустые TF-контексты от --gpus all).
- ВЫВОД по VRAM: 8GB-карты ХВАТАЕТ с запасом (пик 1.2GB из 8GB).

## Предлагаемое исполнение

Env живому контейнеру не поменять — нужно пересоздание. Рекомендуемая цель —
GPU1 (там нагрузка уже живёт): `--gpus "device=GPU-a53eeed2-c3af-d78c-f6b7-4999b50c41eb"`
(привязка по uuid устойчивее, чем по индексу). Схема:

1. Бэкап: `sudo docker inspect sunday_main > /tmp/sunday_main_backup_<ts>.json`.
2. `sudo docker stop sunday_main` → `sudo docker rename sunday_main
   sunday_main_old_<ts>` → `sudo docker update --restart=no sunday_main_old_<ts>`
   (старый НЕ удалять — мгновенный откат переименованием обратно).
3. `sudo docker run -d --name sunday_main --restart always --network host
   --gpus "device=GPU-a53eeed2-..." <ваши -e из старого inspect: PSGSQL*,
   SUNDAY_*> -v /data/projects:/projects -v /data/westmere/wheels:/wheels
   -w /sunday sunday-main-tf212:latest bash ./start.sh`
   (env-значения возьмите из бэкап-JSON п.1 — здесь их не привожу, секреты).
4. Проверка через ~60с: `nvidia-smi` — python-процесс должен сидеть ТОЛЬКО
   на GPU1; в PG появляются свежие предсказания (staleness-гейт 10 мин
   переживает перерыв 30–60с штатно).

Если у вас контейнер поднимается своим скриптом/мейкфайлом — правьте лучше
там (постоянная фиксация), это надёжнее одноразового docker run.

## Зачем это нам (контекст)

После lock'а под обучение RL-ветки освобождаются GPU0 (12GB), GPU2, GPU3.
Юзер сказал «остальные GPU свободны для работы по другим направлениям» —
т.е. и ваши трейны тоже.

По готовности — короткий ответ в шину (from_sunday_*): какая карта
закреплена и что предсказания идут. Спасибо!
