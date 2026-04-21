# Використовуємо офіційний образ uv для копіювання бінарника,
# потім мінімальний Python-образ для запуску.
FROM python:3.11-slim

# Копіюємо uv з офіційного образу (швидше, ніж pip install uv)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Копіюємо lock-файли першими — Docker кешує цей шар,
# перевстановлює залежності лише коли pyproject.toml / uv.lock змінились.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Копіюємо вихідний код
COPY app.py ./

# Hugging Face Spaces очікує порт 7860
EXPOSE 7860

CMD ["uv", "run", "shiny", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]
