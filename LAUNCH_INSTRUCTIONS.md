# Инструкция по запуску генерации музыки

## 1. Открыть терминал и перейти в папку проекта

```bash
export PATH="$HOME/.local/bin:$PATH"
cd "/Users/lev/Documents/AI Antigravity/models/music/ACE-Step/ACE-Step-V15"
```

## 2. Запустить сервер генерации

```bash
uv run acestep --init_service true --offload_to_cpu true --lm_model "ACE-Step-V15"
```

Дождитесь сообщения `Running on local URL: http://127.0.0.1:7860` и откройте эту ссылку в браузере.

---

## 3. Рекомендуемые настройки в Web UI

На основе последних успешных генераций:

| Параметр | Значение | Описание |
| :--- | :--- | :--- |
| **Inference Steps** | 8 | Количество шагов диффузии |
| **Guidance Scale** | 7.0 | Сила следования промпту |
| **Inference Method** | ode | Стабильный ритм и качество |
| **Shift** | 3 | Фокус на семантической структуре |
| **LM Temperature** | 0.85 | Баланс креативности |

### Chain-of-Thought (CoT) — Включить:
- ✅ CoT Metas
- ✅ CoT Caption  
- ✅ CoT Language
- ✅ Constrained Decoding
- ❌ CoT Lyrics (выключено)

---

## 4. Формат промпта

### Caption (описание)
Опишите жанр, инструменты, атмосферу и темп:
> "Russian Urban Pop / Hip-Hop Fusion. Structure: Male rap verses, female chorus. Atmosphere: Cinematic, triumphant. Instruments: Synth-bass, trap drums, brass. Tempo: 92 BPM."

### Lyrics (текст)
Используйте теги для структуры:
```
[verse]
Текст куплета...

[chorus]
Текст припева...

[bridge]
Текст бриджа...

[outro]
Финал...
[end]
```

---

*Основано на успешной генерации batch_1770321400*
