# -*- coding: utf-8 -*-
"""
이미지 해밍거리 계산 (Streamlit Web App)
파이썬 코드 개발: 대전대신고 하진수
웹 구현: 서라벌고 윤진석
"""
import numpy as np
import matplotlib.pyplot as plt
import collections
import streamlit as st

# =====================================================================
# 페이지 설정
# =====================================================================
st.set_page_config(
    page_title="이미지 해밍거리 계산",
    page_icon="🔢",
    layout="wide",
)

st.title("🔢 이미지 해밍거리 계산")
st.caption(
    "파이썬 코드 개발: 대전대신고 하진수 · 웹 구현: 서라벌고 윤진석"
)

# =====================================================================
# 0~9 참조 비트맵 (6x6) — 1단계/2단계용
# =====================================================================
REF_6 = {
    0: np.array([
        [0,1,1,1,1,0],[1,0,0,0,0,1],[1,0,0,0,0,1],
        [1,0,0,0,0,1],[1,0,0,0,0,1],[0,1,1,1,1,0],
    ]),
    1: np.array([
        [0,0,1,1,0,0],[0,0,0,1,0,0],[0,0,0,1,0,0],
        [0,0,0,1,0,0],[0,0,0,1,0,0],[0,0,1,1,1,0],
    ]),
    2: np.array([
        [0,1,1,1,1,0],[1,0,0,0,0,1],[0,0,0,0,1,0],
        [0,0,0,1,0,0],[0,0,1,0,0,0],[1,1,1,1,1,1],
    ]),
    3: np.array([
        [0,1,1,1,1,0],[0,0,0,0,0,1],[0,0,1,1,1,0],
        [0,0,0,0,0,1],[0,0,0,0,0,1],[0,1,1,1,1,0],
    ]),
    4: np.array([
        [0,0,0,1,0,0],[0,0,1,1,0,0],[0,1,0,1,0,0],
        [1,1,1,1,1,1],[0,0,0,1,0,0],[0,0,0,1,0,0],
    ]),
    5: np.array([
        [1,1,1,1,1,1],[1,0,0,0,0,0],[1,1,1,1,1,0],
        [0,0,0,0,0,1],[1,0,0,0,0,1],[0,1,1,1,1,0],
    ]),
    6: np.array([
        [0,1,1,1,1,0],[1,0,0,0,0,0],[1,1,1,1,1,0],
        [1,0,0,0,0,1],[1,0,0,0,0,1],[0,1,1,1,1,0],
    ]),
    7: np.array([
        [1,1,1,1,1,1],[0,0,0,0,0,1],[0,0,0,0,1,0],
        [0,0,0,1,0,0],[0,0,1,0,0,0],[0,1,0,0,0,0],
    ]),
    8: np.array([
        [0,1,1,1,1,0],[1,0,0,0,0,1],[0,1,1,1,1,0],
        [1,0,0,0,0,1],[1,0,0,0,0,1],[0,1,1,1,1,0],
    ]),
    9: np.array([
        [0,1,1,1,1,0],[1,0,0,0,0,1],[1,0,0,0,0,1],
        [0,1,1,1,1,1],[0,0,0,0,0,1],[0,1,1,1,1,0],
    ]),
}

# =====================================================================
# 0~9 참조 비트맵 (12x12) — 3단계용
# =====================================================================
GRID12 = 12
_BASE_8x12 = {
    0: np.array([[0,0,1,1,1,1,0,0],[0,1,1,0,0,1,1,0],[1,1,0,0,0,0,1,1],[1,1,0,0,0,0,1,1],[1,1,0,0,0,0,1,1],[1,1,0,0,0,0,1,1],[1,1,0,0,0,0,1,1],[1,1,0,0,0,0,1,1],[1,1,0,0,0,0,1,1],[0,1,1,0,0,1,1,0],[0,0,1,1,1,1,0,0],[0,0,0,0,0,0,0,0]]),
    1: np.array([[0,0,0,1,1,0,0,0],[0,0,1,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,1,1,1,1,1,1,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]),
    2: np.array([[0,0,1,1,1,1,0,0],[0,1,1,0,0,1,1,0],[1,1,0,0,0,0,1,1],[0,0,0,0,0,0,1,1],[0,0,0,0,0,1,1,0],[0,0,0,0,1,1,0,0],[0,0,0,1,1,0,0,0],[0,0,1,1,0,0,0,0],[0,1,1,0,0,0,0,0],[1,1,1,1,1,1,1,1],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]),
    3: np.array([[0,0,1,1,1,1,0,0],[0,1,1,0,0,1,1,0],[0,0,0,0,0,0,1,1],[0,0,0,0,1,1,1,0],[0,0,0,0,0,0,1,1],[0,0,0,0,0,0,1,1],[0,0,0,0,0,0,1,1],[1,1,0,0,0,0,1,1],[0,1,1,0,0,1,1,0],[0,0,1,1,1,1,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]),
    4: np.array([[0,0,0,0,1,1,0,0],[0,0,0,1,1,1,0,0],[0,0,1,1,0,1,1,0],[0,1,1,0,0,1,1,0],[1,1,1,1,1,1,1,1],[0,0,0,0,0,1,1,0],[0,0,0,0,0,1,1,0],[0,0,0,0,0,1,1,0],[0,0,0,0,0,1,1,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]),
    5: np.array([[1,1,1,1,1,1,1,0],[1,1,0,0,0,0,0,0],[1,1,1,1,1,1,0,0],[0,0,0,0,0,1,1,0],[0,0,0,0,0,1,1,0],[0,0,0,0,0,1,1,0],[0,0,0,0,0,1,1,0],[1,1,0,0,0,0,1,1],[0,1,1,0,0,1,1,0],[0,0,1,1,1,1,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]),
    6: np.array([[0,0,1,1,1,1,0,0],[0,1,1,0,0,0,0,0],[1,1,0,0,0,0,0,0],[1,1,1,1,1,1,0,0],[1,1,0,0,0,1,1,0],[1,1,0,0,0,1,1,0],[1,1,0,0,0,1,1,0],[1,1,0,0,0,1,1,0],[0,1,1,0,0,1,1,0],[0,0,1,1,1,1,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]),
    7: np.array([[1,1,1,1,1,1,1,1],[0,0,0,0,0,1,1,0],[0,0,0,0,1,1,0,0],[0,0,0,0,1,1,0,0],[0,0,0,1,1,0,0,0],[0,0,0,1,1,0,0,0],[0,0,1,1,0,0,0,0],[0,0,1,1,0,0,0,0],[0,0,1,1,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]),
    8: np.array([[0,0,1,1,1,1,0,0],[0,1,1,0,0,1,1,0],[1,1,0,0,0,0,1,1],[0,1,1,0,0,1,1,0],[0,0,1,1,1,1,0,0],[0,1,1,0,0,1,1,0],[1,1,0,0,0,0,1,1],[1,1,0,0,0,0,1,1],[0,1,1,0,0,1,1,0],[0,0,1,1,1,1,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]),
    9: np.array([[0,0,1,1,1,1,0,0],[0,1,1,0,0,1,1,0],[1,1,0,0,0,0,1,1],[1,1,0,0,0,0,1,1],[0,1,1,0,0,1,1,1],[0,0,1,1,1,1,1,1],[0,0,0,0,0,1,1,0],[0,0,0,0,0,1,1,0],[0,1,1,0,0,1,1,0],[0,0,1,1,1,1,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]),
}

def _pad_12(img_8x12: np.ndarray) -> np.ndarray:
    padded = np.zeros((GRID12, GRID12), dtype=int)
    padded[:, 2:10] = img_8x12
    return padded

REF_12 = {k: _pad_12(v) for k, v in _BASE_8x12.items()}


def _shift_image(img: np.ndarray, sr: int, sc: int) -> np.ndarray:
    out = np.zeros_like(img)
    h, w = img.shape
    rs, re = max(0, sr), min(h, h + sr)
    cs, ce = max(0, sc), min(w, w + sc)
    irs, ire = max(0, -sr), min(h, h - sr)
    ics, ice = max(0, -sc), min(w, w - sc)
    out[rs:re, cs:ce] = img[irs:ire, ics:ice]
    return out


@st.cache_data(show_spinner=False)
def build_train_data():
    data = []
    for num, base in REF_12.items():
        for sr, sc in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
            shifted = _shift_image(base, sr, sc)
            data.append({
                "label": num,
                "vector": shifted.flatten(),
                "image": shifted,
            })
    return data


# =====================================================================
# 시각화 헬퍼
# =====================================================================
def render_reference_grid(refs: dict, title: str = ""):
    fig, axes = plt.subplots(2, 5, figsize=(11, 4.5))
    axes = axes.flatten()
    for n, img in refs.items():
        ax = axes[n]
        ax.imshow(1 - img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Digit {n}")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xticks(np.arange(-0.5, img.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, img.shape[0], 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def render_diff_panels(user: np.ndarray, refs: dict, results: list, best_num: int,
                       title_fmt):
    """results: [{'num','hamming',...}], title_fmt(res) -> str"""
    n_grid = user.shape[0]
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    axes = axes.flatten()
    for res in results:
        num = res["num"]
        ref = refs[num]
        ax = axes[num]
        vis = np.ones((n_grid, n_grid, 3))
        diff_idx = np.where(user != ref)
        vis[diff_idx] = [1.0, 0.55, 0.55]  # 차이: 연한 빨강
        match_idx = np.where((user == 1) & (ref == 1))
        vis[match_idx] = [0.2, 0.2, 0.2]   # 둘 다 검정: 진한 회색
        ax.imshow(vis)

        if n_grid <= 6:
            for r, c in zip(*diff_idx):
                ax.text(c, r, "1", ha="center", va="center",
                        color="darkred", fontweight="bold", fontsize=9)

        is_best = (num == best_num)
        ax.set_title(title_fmt(res),
                     color="blue" if is_best else "black",
                     fontweight="bold" if is_best else "normal",
                     fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xticks(np.arange(-0.5, n_grid, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_grid, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    fig.tight_layout()
    return fig


# =====================================================================
# 클릭 가능한 그리드 (Streamlit 버튼 그리드)
# =====================================================================
def clickable_grid(state_key: str, n: int, btn_size: str = "small"):
    """st.session_state[state_key]: (n,n) ndarray. 버튼 클릭으로 0/1 토글."""
    if state_key not in st.session_state:
        st.session_state[state_key] = np.zeros((n, n), dtype=int)
    grid = st.session_state[state_key]

    # 버튼 크기 조정용 CSS
    st.markdown(
        f"""
        <style>
        div[data-testid="stHorizontalBlock"] button {{
            min-height: 0 !important;
            height: { '34px' if n<=6 else '26px' };
            padding: 0 !important;
            font-size: { '14px' if n<=6 else '11px' };
            line-height: 1;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    for i in range(n):
        cols = st.columns(n, gap="small")
        for j in range(n):
            on = bool(grid[i, j])
            label = "■" if on else " "
            btn_type = "primary" if on else "secondary"
            if cols[j].button(label, key=f"{state_key}_{i}_{j}", type=btn_type,
                              use_container_width=True):
                st.session_state[state_key][i, j] = 0 if on else 1
                st.rerun()
    return st.session_state[state_key]


# =====================================================================
# 사이드바: 단계 선택
# =====================================================================
st.sidebar.header("📚 단계 선택")
step = st.sidebar.radio(
    "진행할 단계를 선택하세요",
    [
        "1단계 · 6×6 참조 숫자 보기",
        "2단계 · 6×6 해밍거리 분석",
        "3단계 · 12×12 + KNN 분류",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "💡 **사용 방법**\n\n"
    "- 격자를 **클릭**하여 점을 켜고/끄세요.\n"
    "- 각 단계는 독립적으로 동작합니다.\n"
    "- 단계 간 그림은 **유지**됩니다."
)

# =====================================================================
# 1단계
# =====================================================================
if step.startswith("1단계"):
    st.header("1️⃣ 6×6 행렬 숫자 이미지 만들기")
    st.write("0부터 9까지의 참조 숫자 비트맵을 미리 보여줍니다. "
             "이 모양들이 다음 단계에서 ‘정답지’ 역할을 합니다.")

    fig = render_reference_grid(REF_6, title="참조 숫자 (0~9)")
    st.pyplot(fig)

    with st.expander("🔬 참조 숫자끼리의 해밍거리 표 (0이면 동일, 클수록 다름)"):
        nums = list(range(10))
        mat = np.zeros((10, 10), dtype=int)
        for a in nums:
            for b in nums:
                mat[a, b] = int(np.sum(REF_6[a] != REF_6[b]))
        import pandas as pd
        df = pd.DataFrame(mat, index=[f"{i}" for i in nums],
                          columns=[f"{i}" for i in nums])
        st.dataframe(df, use_container_width=True)
        st.caption("참고: 디자인상 6 vs 8 = 2, 5 vs 6 = 3 처럼 매우 가까운 쌍이 있어 "
                   "사용자 입력에 따라 오분류가 일어날 수 있습니다.")

# =====================================================================
# 2단계
# =====================================================================
elif step.startswith("2단계"):
    st.header("2️⃣ 6×6 그리드로 숫자 그리고 해밍거리 분석")

    left, right = st.columns([1, 1.3])

    with left:
        st.subheader("🖱 입력 그리드 (클릭=토글)")
        user = clickable_grid("grid6", 6)

        c1, c2 = st.columns(2)
        run = c1.button("🔍 해밍거리 분석 실행", type="primary",
                        use_container_width=True, key="run6")
        if c2.button("🧹 초기화", use_container_width=True, key="clear6"):
            st.session_state["grid6"] = np.zeros((6, 6), dtype=int)
            st.rerun()

    with right:
        st.subheader("📊 결과")
        if not run:
            st.info("왼쪽 그리드에 숫자를 그린 뒤 **분석 실행**을 누르세요.")
        else:
            results = []
            for n, ref in REF_6.items():
                d = int(np.sum(user != ref))
                results.append({"num": n, "hamming": d, "ref": ref})
            results_sorted = sorted(results, key=lambda x: x["hamming"])
            best = results_sorted[0]

            st.success(f"가장 유사한 숫자: **{best['num']}**  "
                       f"(해밍거리 = {best['hamming']})")

            import pandas as pd
            table = pd.DataFrame([
                {"순위": i + 1, "숫자": r["num"], "해밍거리": r["hamming"],
                 "유사도(%)": round((1 - r["hamming"] / 36) * 100, 1)}
                for i, r in enumerate(results_sorted)
            ])
            st.dataframe(table, use_container_width=True, hide_index=True)

            fig = render_diff_panels(
                user, REF_6,
                sorted(results, key=lambda x: x["num"]),
                best_num=best["num"],
                title_fmt=lambda r: f"Number {r['num']}\nDist: {r['hamming']}",
            )
            st.pyplot(fig)
            st.caption("🟥 연한 빨강 = 사용자/참조가 서로 다른 픽셀(해밍거리에 +1) · "
                       "⬛ 진한 회색 = 둘 다 검정으로 일치하는 픽셀")

# =====================================================================
# 3단계 — 해밍거리만 사용 (유클리드 거리 제거)
# =====================================================================
else:
    st.header("3️⃣ 12×12 그리드 + KNN(K=5) 숫자 분류 — 해밍거리만 사용")
    st.write("기준 숫자에 상하좌우로 ±1픽셀씩 이동시킨 데이터를 학습 샘플로 사용합니다 "
             "(총 50개). 사용자가 그린 그림과 학습 샘플 사이의 **해밍거리**를 모두 계산해, "
             "가장 가까운 5개의 다수결 라벨로 숫자를 분류합니다.")

    train_data = build_train_data()

    left, right = st.columns([1, 1.3])

    with left:
        st.subheader("🖱 12×12 입력 그리드")
        user = clickable_grid("grid12", GRID12)

        c1, c2 = st.columns(2)
        run = c1.button("🧠 분석 실행", type="primary",
                        use_container_width=True, key="run12")
        if c2.button("🧹 초기화", use_container_width=True, key="clear12"):
            st.session_state["grid12"] = np.zeros((GRID12, GRID12), dtype=int)
            st.rerun()

    with right:
        st.subheader("📊 결과")
        if not run:
            st.info("왼쪽 12×12 그리드에 숫자를 그린 뒤 **분석 실행**을 누르세요.")
        else:
            user_vec = user.flatten()

            # ---------- KNN 거리 계산 (해밍거리) ----------
            knn = []
            for s in train_data:
                # bool 비교 후 합 == 다른 픽셀 개수 == 해밍거리
                d = int(np.sum(user_vec != s["vector"]))
                knn.append({"label": s["label"], "dist": d})
            knn.sort(key=lambda x: x["dist"])
            top5 = knn[:5]
            votes = [x["label"] for x in top5]
            prediction = collections.Counter(votes).most_common(1)[0][0]

            # ---------- 각 base reference와의 해밍거리 ----------
            metrics = []
            for n in range(10):
                ref = REF_12[n]
                ham = int(np.sum(user != ref))
                metrics.append({
                    "num": n,
                    "hamming": ham,
                    "similarity": (1 - ham / (GRID12 * GRID12)) * 100,
                })
            metrics_sorted = sorted(metrics, key=lambda x: x["hamming"])

            st.success(f"🧠 **KNN(K=5) 예측 결과: {prediction}**  "
                       f"(투표: {votes})")

            import pandas as pd
            table = pd.DataFrame([
                {"순위": i + 1,
                 "숫자": m["num"],
                 "해밍거리(픽셀차)": m["hamming"],
                 "유사도(%)": round(m["similarity"], 1)}
                for i, m in enumerate(metrics_sorted[:5])
            ])
            st.markdown("**기본 참조 이미지와의 거리 (상위 5)**")
            st.dataframe(table, use_container_width=True, hide_index=True)

            fig = render_diff_panels(
                user, REF_12,
                sorted(metrics, key=lambda x: x["num"]),
                best_num=prediction,
                title_fmt=lambda r: (
                    f"Num {r['num']}\nHamming: {r['hamming']}"
                ),
            )
            st.pyplot(fig)
            st.caption(
                "🟥 연한 빨강 = 다른 픽셀 · ⬛ 진한 회색 = 둘 다 검정으로 일치하는 픽셀 · "
                "파란 제목 = KNN 예측 라벨"
            )

            with st.expander("🔍 KNN 상세: 가장 가까운 5개 학습 샘플"):
                knn_table = pd.DataFrame([
                    {"순위": i + 1, "라벨": x["label"], "해밍거리": x["dist"]}
                    for i, x in enumerate(top5)
                ])
                st.dataframe(knn_table, use_container_width=True, hide_index=True)

# =====================================================================
# 푸터
# =====================================================================
st.markdown("---")
st.caption(
    "© 이미지 해밍거리 계산  ·  파이썬 코드 개발: 대전대신고 하진수  ·  웹 구현: 서라벌고 윤진석"
)
