---
title: Cosine Similarity in Recommender Systems
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Косинусна подібність у рекомендаційних системах

Інтерактивний навчальний застосунок на [Shiny for Python](https://shiny.posit.co/py/).

**Три вкладки:**
- 📊 **Матриця** — редагована матриця вподобань, косинусна подібність, таблиця сусідів.
- 🔥 **Теплокарта** — хітмап з рядками, відсортованими за подібністю до цільового користувача.
- ⚖️ **Вага сигналів** — симуляція Engagement Score (TikTok / YouTube-логіка).

## Запуск локально

```bash
uv run shiny run --reload app.py
```
