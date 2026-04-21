"""
Інтерактивний навчальний застосунок: Косинусна подібність у рекомендаційних системах.

Запуск:
    uv run shiny run --reload app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------------ Константи

DEFAULT_USERS = [
    "Макс", "Аня", "Олег", "Іра", "Дмитро",
    "Софія", "Тарас", "Катя", "Богдан", "Оля",
]
DEFAULT_FEATURES = [
    "Футбол", "Кіберспорт", "Музика", "Кулінарія", "Подорожі",
    "Ігри", "Наука", "Мистецтво", "Кіно", "Книги",
]
MIN_N, MAX_N = 2, 10
MIN_M, MAX_M = 2, 10
DEFAULT_N = 5
DEFAULT_M = 5
RATING_MIN, RATING_MAX = 0, 5

TARGET_COLOR = "#e74c3c"        # червоний — цільовий
NEIGHBOR_COLOR = "#27ae60"      # зелений — найближчий сусід
OTHER_COLOR = "#95a5a6"         # сірий — решта


def default_user_name(i: int) -> str:
    return DEFAULT_USERS[i] if i < len(DEFAULT_USERS) else f"Користувач {i + 1}"


def default_feature_name(j: int) -> str:
    return DEFAULT_FEATURES[j] if j < len(DEFAULT_FEATURES) else f"Категорія {j + 1}"


def make_random_matrix(n: int, m: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(RATING_MIN, RATING_MAX + 1, size=(n, m)).astype(int)


# ------------------------------------------------------------------ UI helpers

def _dynamic_name_inputs(prefix: str, count_input_id: str,
                         max_count: int, default_fn) -> ui.Tag:
    """Статично створює max_count текстових інпутів, показує тільки перші N через CSS.

    Це зберігає стан користувацьких імен навіть коли N/M змінюється —
    ми не перестворюємо інпути, лише ховаємо/показуємо.
    """
    return ui.div(
        *[
            ui.panel_conditional(
                f"input.{count_input_id} > {i}",
                ui.input_text(
                    f"{prefix}_{i}",
                    label=f"#{i + 1}",
                    value=default_fn(i),
                    width="100%",
                ),
            )
            for i in range(max_count)
        ],
        class_="name-input-stack",
    )


# ------------------------------------------------------------------ Tab 1: Матриця

matrix_tab = ui.nav_panel(
    "📊 Матриця",
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_numeric(
                "n_users", "Користувачів (N)",
                value=DEFAULT_N, min=MIN_N, max=MAX_N, step=1,
            ),
            ui.input_numeric(
                "m_features", "Категорій (M)",
                value=DEFAULT_M, min=MIN_M, max=MAX_M, step=1,
            ),
            ui.input_select(
                "target_user", "🎯 Цільовий користувач",
                choices=DEFAULT_USERS[:DEFAULT_N],
                selected=DEFAULT_USERS[0],
            ),
            ui.input_action_button(
                "randomize", "🎲 Перемішати",
                class_="btn-primary btn-sm w-100 mb-1",
            ),
            ui.input_action_button(
                "zeros", "🗑 Очистити",
                class_="btn-outline-secondary btn-sm w-100 mb-1",
            ),
            ui.accordion(
                ui.accordion_panel(
                    "👤 Імена користувачів",
                    _dynamic_name_inputs(
                        "user_name", "n_users", MAX_N, default_user_name,
                    ),
                ),
                ui.accordion_panel(
                    "🎯 Категорії інтересів",
                    _dynamic_name_inputs(
                        "feature_name", "m_features", MAX_M, default_feature_name,
                    ),
                ),
                ui.accordion_panel(
                    "📚 Формула",
                    ui.markdown(r"""
$$\text{sim}(A, B) = \frac{A \cdot B}{\|A\| \cdot \|B\|}$$

- **1.0** — ідентичні вподобання (кут 0°)
- **0.0** — нічого спільного (кут 90°)

Чисельник — скалярний добуток, знаменник — добуток довжин векторів.
"""),
                ),
                id="matrix_acc",
                open=False,
            ),
            width=260,
            gap="0.3rem",
        ),
        ui.layout_columns(
            ui.value_box(
                "🎯 Ціль",
                ui.output_text("target_user_display"),
                theme="primary",
            ),
            ui.value_box(
                "📈 Найвища подібність",
                ui.output_text("top_similarity_display"),
                theme="bg-gradient-indigo-blue",
            ),
            ui.value_box(
                "🤝 Рекомендований сусід",
                ui.output_text("top_neighbor_display"),
                theme="success",
            ),
            col_widths=[4, 4, 4],
            fill=False,
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header(
                    "Матриця переваг (клац на клітинку, щоб редагувати; 0–5)"
                ),
                ui.output_data_frame("matrix_display"),
                full_screen=True,
            ),
            ui.card(
                ui.card_header("🤝 Найближчі сусіди"),
                ui.output_data_frame("neighbors_table"),
                full_screen=True,
            ),
            col_widths=[7, 5],
            fill=True,
        ),
    ),
)


# ------------------------------------------------------------------ Tab 2: Теплокарта

HEATMAP_COLORSCALES = {
    "Viridis": "Viridis",
    "Blues": "Blues",
    "Plasma": "Plasma",
    "YlGnBu": "YlGnBu",
    "RdYlGn": "RdYlGn",
    "Cividis": "Cividis",
}

heatmap_tab = ui.nav_panel(
    "🔥 Теплокарта",
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_select(
                "heatmap_colorscale", "🎨 Колірна палітра",
                choices=HEATMAP_COLORSCALES,
                selected="Viridis",
            ),
            ui.input_switch(
                "heatmap_show_values", "Значення у клітинках",
                value=True,
            ),
            ui.accordion(
                ui.accordion_panel(
                    "🧠 Ідея",
                    ui.markdown("""
Кожен **рядок** — користувач, кожна **клітинка** — оцінка категорії.
Колір кодує силу вподобання.

**Сортування рядків:**

- 🎯 ціль — завжди вгорі;
- 🤝 найближчий сусід — одразу під нею;
- 🔍 решта — за спаданням косинусної подібності.
"""),
                ),
                ui.accordion_panel(
                    "💡 Як це читати?",
                    ui.markdown("""
- **Схожі смуги** у сусідніх рядках → висока подібність.
- **Контраст між рядками** → мало спільного.
- Якщо у цілі клітинка темна, а у сусіда **яскрава** — саме цю
  категорію алгоритм і порекомендує 🎯.

Це і є **Collaborative Filtering**: «людям, схожим на тебе,
сподобалось *оце* — можливо, сподобається і тобі».
"""),
                ),
                id="heatmap_acc",
                open=False,
            ),
            width=260,
            gap="0.3rem",
        ),
        ui.card(
            ui.card_header(
                "🔥 Теплокарта вподобань "
                "(рядки відсортовано за косинусною подібністю)"
            ),
            output_widget("heatmap_plot", fill=True),
            full_screen=True,
        ),
    ),
)


# ------------------------------------------------------------------ Tab 3: Вага сигналів

weights_tab = ui.nav_panel(
    "⚖️ Вага сигналів",
    ui.layout_sidebar(
        ui.sidebar(
            ui.markdown(r"""
### 💡 Сенс

У реальних системах (**TikTok**, **YouTube**, **Instagram**) клітинки матриці
переваг **не бінарні**. Замість 1/0 алгоритм агрегує десятки сигналів у єдиний
**Engagement Score**:

$$\text{Score} = \sum_{k} w_k \cdot \text{сигнал}_k$$

**Чим "дорожча" дія — тим більша вага:**

- ⚡ Лайк — мала вага.
- 💬 Коментар — середня.
- 🔁 Репост — найбільша.

Цей композитний бал і потрапляє у матрицю з 1-ї вкладки.
"""),
            width=280,
            gap="0.3rem",
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("👁 Дії користувача"),
                ui.input_slider(
                    "watch_time", "Час перегляду",
                    min=0, max=100, value=50, step=5, post=" %",
                ),
                ui.input_checkbox("liked", "❤️ Лайк"),
                ui.input_checkbox("commented", "💬 Коментар"),
                ui.input_checkbox("reposted", "🔁 Репост"),
                fill=True,
            ),
            ui.card(
                ui.card_header("⚙️ Ваги алгоритму"),
                ui.input_slider(
                    "w_watch", "W_watch (перегляд)",
                    min=0, max=10, value=5, step=0.5,
                ),
                ui.input_slider(
                    "w_like", "W_like (лайк)",
                    min=0, max=10, value=3, step=0.5,
                ),
                ui.input_slider(
                    "w_comment", "W_comment (коментар)",
                    min=0, max=20, value=10, step=0.5,
                ),
                ui.input_slider(
                    "w_repost", "W_repost (репост)",
                    min=0, max=30, value=20, step=0.5,
                ),
                fill=True,
            ),
            ui.div(
                ui.value_box(
                    "⭐ Engagement Score",
                    ui.output_text("final_score"),
                    ui.output_text("score_interpretation"),
                    theme="bg-gradient-orange-red",
                ),
                ui.card(
                    ui.card_header("🧮 Розрахунок"),
                    ui.output_ui("score_formula"),
                    fill=True,
                ),
                class_="d-flex flex-column gap-2 h-100",
            ),
            col_widths=[3, 4, 5],
            fill=True,
        ),
    ),
)


# ------------------------------------------------------------------ Page

app_ui = ui.page_navbar(
    ui.head_content(
        ui.tags.script("""
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
  },
  svg: { fontCache: 'global' }
};
"""),
        ui.tags.script(
            src="https://polyfill.io/v3/polyfill.min.js?features=es6",
        ),
        ui.tags.script(
            src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js",
            id="MathJax-script",
            **{"async": "async"},
        ),
        ui.tags.style("""
            /* Увесь додаток — на один екран, без глобальних скролів. */
            html, body { height: 100%; }
            body.bslib-page-navbar { overflow: hidden; }
            .navbar { min-height: 44px; padding-top: 0.25rem; padding-bottom: 0.25rem; }
            .navbar-brand { font-size: 1rem; }

            /* Компактні value-box'и. */
            .bslib-value-box { min-height: 72px; }
            .bslib-value-box .value-box-area { padding: 0.5rem 0.75rem; }
            .bslib-value-box .value-box-title { font-size: 0.8rem; margin: 0; }
            .bslib-value-box .value-box-value { font-size: 1.25rem; line-height: 1.2; margin: 0; }
            .bslib-value-box .value-box-showcase { padding: 0 0.5rem; }

            /* Компактні картки. */
            .card-header { padding: 0.35rem 0.6rem; font-size: 0.9rem; font-weight: 600; }
            .card-body { padding: 0.5rem 0.6rem; }

            /* Компактні інпути. */
            .shiny-input-container { margin-bottom: 0.4rem; }
            .form-label, .control-label, label { font-size: 0.82rem; margin-bottom: 0.15rem; }
            .form-control, .form-select { padding: 0.2rem 0.4rem; font-size: 0.85rem; min-height: 0; }
            .irs--shiny { height: 40px; }
            .irs--shiny .irs-bar { top: 25px; }
            .irs--shiny .irs-line { top: 25px; }
            .irs--shiny .irs-handle { top: 18px; }
            .irs--shiny .irs-min, .irs--shiny .irs-max,
            .irs--shiny .irs-from, .irs--shiny .irs-to, .irs--shiny .irs-single {
                font-size: 0.7rem; padding: 1px 3px;
            }

            /* Кнопки щільніше. */
            .btn-sm, .btn { padding: 0.25rem 0.5rem; font-size: 0.85rem; }

            /* Акордеон компактний. */
            .accordion-button { padding: 0.4rem 0.6rem; font-size: 0.85rem; }
            .accordion-body { padding: 0.45rem 0.6rem; font-size: 0.82rem; }

            /* Приховати системні імена-інпути в акордеоні без відступів. */
            .name-input-stack .shiny-input-container { margin-bottom: 0.2rem; }
            .name-input-stack label { font-size: 0.75rem; margin-bottom: 0.05rem; }

            /* Таблиці. */
            .shiny-data-frame td, .shiny-data-frame th { text-align: center; font-size: 0.85rem; padding: 2px 4px; }

            /* Сайдбар трохи щільніше. */
            .bslib-sidebar-layout > .sidebar { padding: 0.5rem; font-size: 0.85rem; }
            .bslib-sidebar-layout hr { margin: 0.4rem 0; }

            /* Маленька формула в Tab 3. */
            .mini-formula code { font-size: 0.72rem; word-break: break-word; }
            .mini-formula .badge { font-size: 0.7rem; }
        """),
    ),
    matrix_tab,
    heatmap_tab,
    weights_tab,
    title="🎯 Косинусна подібність",
    id="main_nav",
    fillable=True,
    padding="0.4rem",
)


# ------------------------------------------------------------------ Server

def server(input, output, session):
    # Сирі значення матриці (numpy int array). Імена керуються окремо.
    matrix_values = reactive.value(make_random_matrix(DEFAULT_N, DEFAULT_M, seed=42))

    # ----- Розміри (з обмеженнями) -----
    @reactive.calc
    def n_users() -> int:
        v = input.n_users()
        if v is None:
            return DEFAULT_N
        return max(MIN_N, min(MAX_N, int(v)))

    @reactive.calc
    def m_features() -> int:
        v = input.m_features()
        if v is None:
            return DEFAULT_M
        return max(MIN_M, min(MAX_M, int(v)))

    # ----- Імена (з дедуплікацією для валідних pandas-індексів) -----
    def _collect_names(prefix: str, count: int, default_fn) -> list[str]:
        names: list[str] = []
        seen: dict[str, int] = {}
        for i in range(count):
            try:
                raw = input[f"{prefix}_{i}"]()
            except Exception:
                raw = None
            name = (raw or "").strip() or default_fn(i)
            if name in seen:
                seen[name] += 1
                name = f"{name} ({seen[name]})"
            else:
                seen[name] = 1
            names.append(name)
        return names

    @reactive.calc
    def user_names() -> list[str]:
        return _collect_names("user_name", n_users(), default_user_name)

    @reactive.calc
    def feature_names() -> list[str]:
        return _collect_names("feature_name", m_features(), default_feature_name)

    # ----- Ресайз матриці при зміні N або M (зі збереженням існуючих значень) -----
    @reactive.effect
    @reactive.event(input.n_users, input.m_features)
    def _resize_matrix():
        n, m = n_users(), m_features()
        cur = matrix_values.get()
        cn, cm = cur.shape
        if cn == n and cm == m:
            return
        new = np.zeros((n, m), dtype=int)
        r, c = min(cn, n), min(cm, m)
        new[:r, :c] = cur[:r, :c]
        rng = np.random.default_rng(42 + 1000 * n + m)
        if n > cn:
            new[cn:, :c] = rng.integers(RATING_MIN, RATING_MAX + 1, size=(n - cn, c))
        if m > cm:
            new[:, cm:] = rng.integers(RATING_MIN, RATING_MAX + 1, size=(n, m - cm))
        matrix_values.set(new)

    # ----- Кнопки -----
    @reactive.effect
    @reactive.event(input.randomize, ignore_init=True)
    def _randomize():
        seed = int(input.randomize()) * 17 + 3
        matrix_values.set(make_random_matrix(n_users(), m_features(), seed=seed))

    @reactive.effect
    @reactive.event(input.zeros, ignore_init=True)
    def _zero():
        matrix_values.set(np.zeros((n_users(), m_features()), dtype=int))

    # ----- DataFrame з іменами -----
    @reactive.calc
    def matrix_df() -> pd.DataFrame:
        data = matrix_values.get()
        n, m = data.shape
        users = user_names()[:n]
        feats = feature_names()[:m]
        while len(users) < n:
            users.append(default_user_name(len(users)))
        while len(feats) < m:
            feats.append(default_feature_name(len(feats)))
        return pd.DataFrame(
            data,
            index=pd.Index(users, name="👤 Користувач"),
            columns=feats,
        )

    # ----- Синхронізація списку цільового користувача -----
    @reactive.effect
    def _sync_target_user_choices():
        users = user_names()
        if not users:
            return
        current = input.target_user()
        selected = current if current in users else users[0]
        ui.update_select("target_user", choices=users, selected=selected)

    # ----- Відображення матриці (редаговане) -----
    @render.data_frame
    def matrix_display():
        df = matrix_df().reset_index()
        return render.DataGrid(df, editable=True, width="100%", height="auto")

    @matrix_display.set_patch_fn
    async def _patch_matrix(*, patch):
        df = matrix_df().reset_index()
        row_idx = patch["row_index"]
        col_idx = patch["column_index"]
        # Перший стовпець — імена користувачів, редагувати не можна
        if col_idx == 0:
            return df.iloc[row_idx, 0]
        try:
            value = int(float(str(patch["value"]).replace(",", ".").strip()))
        except (ValueError, TypeError):
            # Залишаємо поточне значення
            current = matrix_values.get()
            return int(current[row_idx, col_idx - 1])
        value = max(RATING_MIN, min(RATING_MAX, value))
        current = matrix_values.get().copy()
        current[row_idx, col_idx - 1] = value
        matrix_values.set(current)
        return value

    # ----- Косинусна подібність -----
    @reactive.calc
    def similarity_results() -> pd.DataFrame | None:
        df = matrix_df()
        target = input.target_user()
        if target is None or target not in df.index:
            return None
        target_vec = df.loc[target].values.astype(float).reshape(1, -1)
        others = df.drop(index=target)
        if len(others) == 0:
            return pd.DataFrame({"Користувач": [], "Подібність": []})
        others_vals = others.values.astype(float)
        sims = cosine_similarity(target_vec, others_vals)[0]
        sims = np.nan_to_num(sims, nan=0.0)
        return (
            pd.DataFrame({
                "Користувач": others.index.tolist(),
                "Подібність": sims,
            })
            .sort_values("Подібність", ascending=False)
            .reset_index(drop=True)
        )

    @render.data_frame
    def neighbors_table():
        res = similarity_results()
        if res is None or len(res) == 0:
            empty = pd.DataFrame({"#": [], "Користувач": [], "Подібність": []})
            return render.DataGrid(empty, width="100%")
        display = res.copy()
        display.insert(0, "#", range(1, len(display) + 1))
        display["Подібність"] = display["Подібність"].map(lambda x: f"{x:.4f}")
        return render.DataGrid(display, width="100%", height="auto")

    # ----- Value boxes -----
    @render.text
    def target_user_display():
        return input.target_user() or "—"

    @render.text
    def top_similarity_display():
        res = similarity_results()
        if res is None or len(res) == 0:
            return "—"
        return f"{res.iloc[0]['Подібність']:.4f}"

    @render.text
    def top_neighbor_display():
        res = similarity_results()
        if res is None or len(res) == 0:
            return "—"
        return str(res.iloc[0]["Користувач"])

    # ----- Теплокарта -----
    @render_widget
    def heatmap_plot():
        df = matrix_df()
        target = input.target_user()
        if target is None or target not in df.index or len(df) == 0:
            return go.Figure()

        # Сортуємо рядки: цільовий → найближчий → решта за спаданням подібності
        res = similarity_results()
        ordered: list[str] = [target]
        if res is not None and len(res) > 0:
            ordered.extend(res["Користувач"].tolist())
        ordered = [u for u in ordered if u in df.index]
        sorted_df = df.loc[ordered]

        nearest = (
            res.iloc[0]["Користувач"]
            if res is not None and len(res) > 0
            else None
        )

        # Префіксуємо імена емодзі, щоб виділити ключові ролі прямо на осі Y.
        def decorate(u: str) -> str:
            if u == target:
                return f"🎯 {u}"
            if u == nearest:
                return f"🤝 {u}"
            return u

        y_labels = [decorate(u) for u in ordered]

        colorscale = input.heatmap_colorscale() or "Viridis"
        show_values = bool(input.heatmap_show_values())

        fig = px.imshow(
            sorted_df.values,
            x=sorted_df.columns.tolist(),
            y=y_labels,
            color_continuous_scale=colorscale,
            aspect="auto",
            text_auto=".0f" if show_values else False,
            zmin=RATING_MIN,
            zmax=RATING_MAX,
            labels=dict(x="Категорія", y="Користувач", color="Оцінка"),
        )
        fig.update_traces(
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Категорія: %{x}<br>"
                "Оцінка: %{z}<extra></extra>"
            ),
            xgap=2,
            ygap=2,
        )
        fig.update_layout(
            height=600,
            margin=dict(l=40, r=20, t=20, b=60),
            coloraxis_colorbar=dict(
                title=dict(text="Оцінка"),
                tickvals=list(range(RATING_MIN, RATING_MAX + 1)),
            ),
            xaxis=dict(side="top", tickangle=-30),
            yaxis=dict(autorange="reversed"),
            plot_bgcolor="white",
        )

        # Підсвітимо прямокутником два верхні рядки (ціль + сусід).
        n_rows = len(ordered)
        if n_rows >= 1:
            fig.add_shape(
                type="rect",
                xref="paper", yref="y",
                x0=0, x1=1,
                y0=-0.5, y1=0.5,
                line=dict(color=TARGET_COLOR, width=3),
                fillcolor="rgba(0,0,0,0)",
                layer="above",
            )
        if nearest is not None and n_rows >= 2:
            fig.add_shape(
                type="rect",
                xref="paper", yref="y",
                x0=0, x1=1,
                y0=0.5, y1=1.5,
                line=dict(color=NEIGHBOR_COLOR, width=3, dash="dash"),
                fillcolor="rgba(0,0,0,0)",
                layer="above",
            )
        return fig

    # ----- Engagement Score -----
    @reactive.calc
    def engagement() -> dict:
        w_pct = input.watch_time() or 0
        ww = input.w_watch() or 0
        wl = input.w_like() or 0
        wc = input.w_comment() or 0
        wr = input.w_repost() or 0
        lk = 1 if input.liked() else 0
        cm = 1 if input.commented() else 0
        rp = 1 if input.reposted() else 0

        watch_score = (w_pct / 100.0) * ww * 10
        like_score = lk * wl
        comment_score = cm * wc
        repost_score = rp * wr
        total = watch_score + like_score + comment_score + repost_score

        return {
            "watch_pct": w_pct,
            "liked": lk, "commented": cm, "reposted": rp,
            "w_watch": ww, "w_like": wl, "w_comment": wc, "w_repost": wr,
            "watch_score": watch_score,
            "like_score": like_score,
            "comment_score": comment_score,
            "repost_score": repost_score,
            "total": total,
        }

    @render.ui
    def score_formula():
        e = engagement()
        return ui.HTML(f"""
<div class="mini-formula">
  <div class="mb-1"><strong>Формула:</strong>
    <code class="text-primary">Watch%/100·W_watch·10 + Like·W_like + Comment·W_comment + Repost·W_repost</code>
  </div>
  <div class="mb-1"><strong>Ваші значення:</strong>
    <code>{e['watch_pct']}/100·{e['w_watch']}·10 + {e['liked']}·{e['w_like']} + {e['commented']}·{e['w_comment']} + {e['reposted']}·{e['w_repost']}</code>
  </div>
  <div class="mb-1">
    <span class="badge bg-info text-dark">👁 {e['watch_score']:.1f}</span>
    <span class="badge bg-success">❤️ {e['like_score']:.1f}</span>
    <span class="badge bg-warning text-dark">💬 {e['comment_score']:.1f}</span>
    <span class="badge bg-danger">🔁 {e['repost_score']:.1f}</span>
  </div>
  <div><strong>Σ:</strong>
    <code>{e['watch_score']:.1f} + {e['like_score']:.1f} + {e['comment_score']:.1f} + {e['repost_score']:.1f} =
    <span class="text-danger fw-bold">{e['total']:.2f}</span></code>
  </div>
</div>
""")

    @render.text
    def final_score():
        return f"{engagement()['total']:.2f}"

    @render.text
    def score_interpretation():
        total = engagement()["total"]
        if total < 10:
            return "Користувач майже не зацікавлений."
        if total < 30:
            return "Слабка зацікавленість."
        if total < 70:
            return "Середня зацікавленість — варто показувати схожий контент."
        if total < 120:
            return "Висока зацікавленість — сильний сигнал для рекомендацій."
        return "Дуже висока зацікавленість — ідеальний матеріал для алгоритму."


# ------------------------------------------------------------------ App

app = App(app_ui, server)

