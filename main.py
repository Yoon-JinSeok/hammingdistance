# -*- coding: utf-8 -*-
"""
이미지 해밍거리 계산 (Streamlit 웹 버전)
- 파이썬 코드 개발: 대전대신고 하진수
- 웹 구현      : 서라벌고 윤진석
"""
import collections
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# =========================================================
# 페이지 설정 / 공통 스타일
# =========================================================
st.set_page_config(page_title="이미지 해밍거리 계산", layout="wide")

st.markdown(
    """
    <style>
    /* 그리드용 버튼을 정사각형 + 작게 */
    div[data-testid="column"] .stButton > button {
        padding: 0px 0px;
        min-height: 28px;
        height: 28px;
        width: 100%;
        font-size: 14px;
        line-height: 1;
        border-radius: 4px;
    }
    /* 큰 셀(6x6)은 조금 더 크게 */
    .big-grid div[data-testid="column"] .stButton > button {
        min-height: 38px;
        height: 38px;
        font-size: 18px;
    }
    .small-note { color: #666; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔢 이미지 해밍거리 계산")
st.caption(
    "파이썬 코드 개발: 대전대신고 하진수 / 웹 구현: 서라벌고 윤진석"
)

# =========================================================
# 데이터: 6x6 참조 숫자 (2단계용)
# =========================================================
REF_6 = {
    0: np.array([[0,1,1,1,1,0],[1,0,0,0,0,1],[1,0,0,0,0,1],[1,0,0,0,0,1],[1,0,0,0,0,1],[0,1,1,1,1,0]]),
    1: np.array([[0,0,1,1,0,0],[0,0,0,1,0,0],[0,0,0,1,0,0],[0,0,0,1,0,0],[0,0,0,1,0,0],[0,0,1,1,1,0]]),
    2: np.array([[0,1,1,1,1,0],[1,0,0,0,0,1],[0,0,0,0,1,0],[0,0,0,1,0,0],[0,0,1,0,0,0],[1,1,1,1,1,1]]),
    3: np.array([[0,1,1,1,1,0],[0,0,0,0,0,1],[0,0,1,1,1,0],[0,0,0,0,0,1],[0,0,0,0,0,1],[0,1,1,1,1,0]]),
    4: np.array([[0,0,0,1,0,0],[0,0,1,1,0,0],[0,1,0,1,0,0],[1,1,1,1,1,1],[0,0,0,1,0,0],[0,0,0,1,0,0]]),
    5: np.array([[1,1,1,1,1,1],[1,0,0,0,0,0],[1,1,1,1,1,0],[0,0,0,0,0,1],[1,0,0,0,0,1],[0,1,1,1,1,0]]),
    6: np.array([[0,1,1,1,1,0],[1,0,0,0,0,0],[1,1,1,1,1,0],[1,0,0,0,0,1],[1,0,0,0,0,1],[0,1,1,1,1,0]]),
    7: np.array([[1,1,1,1,1,1],[0,0,0,0,0,1],[0,0,0,0,1,0],[0,0,0,1,0,0],[0,0,1,0,0,0],[0,1,0,0,0,0]]),
    8: np.array([[0,1,1,1,1,0],[1,0,0,0,0,1],[0,1,1,1,1,0],[1,0,0,0,0,1],[1,0,0,0,0,1],[0,1,1,1,1,0]]),
    9: np.array([[0,1,1,1,1,0],[1,0,0,0,0,1],[1,0,0,0,0,1],[0,1,1,1,1,1],[0,0,0,0,0,1],[0,1,1,1,1,0]]),
}

# =========================================================
# 데이터: 12x12 참조 숫자 (3단계용) - 12x8 → 12x12 패딩
# =========================================================
_BASE_RAW_12 = {
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

GRID_SIZE_12 = 12

def _pad_to_12x12(img_12x8: np.ndarray) -> np.ndarray:
    padded = np.zeros((12, 12), dtype=int)
    padded[:, 2:10] = img_12x8
    return padded

REF_12 = {k: _pad_to_12x12(v) for k, v in _BASE_RAW_12.items()}

def shift_image(img: np.ndarray, sr: int, sc: int) -> np.ndarray:
    H, W = img.shape
    out = np.zeros_like(img)
    r0, r1 = max(0, sr), min(H, H + sr)
    c0, c1 = max(0, sc), min(W, W + sc)
    ir0, ir1 = max(0, -sr), min(H, H - sr)
    ic0, ic1 = max(0, -sc), min(W, W - sc)
    out[r0:r1, c0:c1] = img[ir0:ir1, ic0:ic1]
    return out

@st.cache_data(show_spinner=False)
def build_train_data():
    """KNN 학습 데이터(상하좌우 1픽셀 이동 증강) 캐싱."""
    data = []
    for num, base_img in REF_12.items():
        for sr, sc in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
            shifted = shift_image(base_img, sr, sc)
            data.append({"label": num, "vector": shifted.flatten()})
    return data

# =========================================================
# 시각화 헬퍼
# =========================================================
def plot_reference_grid(refs: dict, title: str = ""):
    fig, axes = plt.subplots(2, 5, figsize=(11, 5))
    axes = axes.flatten()
    n = next(iter(refs.values())).shape[0]
    for k in range(10):
        ax = axes[k]
        ax.imshow(1 - refs[k], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Digit {k}")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_diff_panels(user_grid: np.ndarray, refs: dict, results: list, best_num: int,
                     show_diff_text: bool = True):
    """각 숫자별 사용자 그림과 참조의 차이를 시각화."""
    n = user_grid.shape[0]
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    axes = axes.flatten()
    for r in results:
        num = r["num"]
        ax = axes[num]
        ref_img = refs[num]
        diff_mask = (user_grid != ref_img)
        visual = np.ones((n, n, 3))
        # 차이: 연한 빨강
        dr_idx, dc_idx = np.where(diff_mask)
        visual[dr_idx, dc_idx] = [1, 0.5, 0.5]
        # 둘 다 켜진 픽셀: 진한 회색
        match_idx = np.where((user_grid == 1) & (ref_img == 1))
        visual[match_idx] = [0.2, 0.2, 0.2]
        ax.imshow(visual)
        if show_diff_text:
            for dr, dc in zip(dr_idx, dc_idx):
                ax.text(dc, dr, "1", ha="center", va="center",
                        color="darkred", fontweight="bold", fontsize=8)
        is_best = (num == best_num)
        info = [f"Number {num}", f"Hamming: {r['hamming']}"]
        if "euclidean" in r:
            info.append(f"E-Dist: {r['euclidean']:.1f}")
        ax.set_title("\n".join(info),
                     color="blue" if is_best else "black",
                     fontweight="bold" if is_best else "normal",
                     fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    plt.tight_layout()
    return fig

# =========================================================
# 그리드 입력 UI (버튼 토글)
# =========================================================
def init_grid(state_key: str, size: int):
    if state_key not in st.session_state:
        st.session_state[state_key] = np.zeros((size, size), dtype=int)

def render_clickable_grid(state_key: str, size: int, css_class: str = ""):
    grid = st.session_state[state_key]
    if css_class:
        st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)
    for r in range(size):
        cols = st.columns(size, gap="small")
        for c in range(size):
            label = "■" if grid[r, c] == 1 else " "
            btn_type = "primary" if grid[r, c] == 1 else "secondary"
            if cols[c].button(label, key=f"{state_key}_{r}_{c}", type=btn_type):
                grid[r, c] = 1 - grid[r, c]
                st.session_state[state_key] = grid
                st.rerun()
    if css_class:
        st.markdown("</div>", unsafe_allow_html=True)

def reset_grid(state_key: str, size: int):
    st.session_state[state_key] = np.zeros((size, size), dtype=int)

# =========================================================
# 사이드바: 단계 선택
# =========================================================
st.sidebar.header("📚 단계 선택")
step = st.sidebar.radio(
    "현재 단계",
    [
        "1단계 · 6x6 참조 숫자 보기",
        "2단계 · 6x6 해밍거리 계산",
        "3단계 · 12x12 해밍거리 + KNN 분류",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div class='small-note'>각 단계는 사이드바에서 선택해 순서대로 진행하세요. "
    "단계별 그리드 상태는 독립적으로 유지됩니다.</div>",
    unsafe_allow_html=True,
)

# =========================================================
# 1단계
# =========================================================
if step.startswith("1단계"):
    st.header("1단계 · 6×6 참조 숫자(0~9) 살펴보기")
    st.write(
        "각 숫자는 6×6 비트맵(검정=1, 흰색=0)으로 정의되어 있습니다. "
        "이 모양들이 다음 단계에서 비교 대상이 됩니다."
    )
    fig = plot_reference_grid(REF_6, title="6x6 Reference Digits")
    st.pyplot(fig)

    with st.expander("ℹ️ 참고: 참조 숫자들 사이의 해밍거리 (가까울수록 비슷)"):
        M = np.array([[int(np.sum(REF_6[i] != REF_6[j])) for j in range(10)] for i in range(10)])
        st.dataframe(
            {"숫자": list(range(10)), **{str(j): M[:, j].tolist() for j in range(10)}},
            use_container_width=True,
        )
        st.caption(
            "예) 6 vs 8 의 해밍거리는 매우 작아 사용자가 그린 6이 8로 분류될 수 있습니다. "
            "이는 코드 버그가 아니라 6×6 디자인의 한계입니다."
        )

# =========================================================
# 2단계
# =========================================================
elif step.startswith("2단계"):
    st.header("2단계 · 6×6 그리드에 직접 그리고 해밍거리 계산")
    st.write("아래 6×6 격자의 셀을 **클릭**하면 검정/흰색이 토글됩니다. "
             "다 그렸으면 [해밍거리 분석 실행]을 누르세요.")

    init_grid("grid6", 6)

    left, right = st.columns([1, 1.2])
    with left:
        st.subheader("✏️ 입력 그리드 (6×6)")
        render_clickable_grid("grid6", 6, css_class="big-grid")
        c1, c2 = st.columns(2)
        run = c1.button("🚀 해밍거리 분석 실행", type="primary", use_container_width=True)
        if c2.button("🔄 그리드 초기화", use_container_width=True):
            reset_grid("grid6", 6)
            st.rerun()

    with right:
        st.subheader("📐 참조 숫자 미리보기")
        st.pyplot(plot_reference_grid(REF_6))

    if run:
        user_grid = st.session_state["grid6"]
        if user_grid.sum() == 0:
            st.warning("⚠️ 빈 그리드입니다. 먼저 셀을 클릭해 숫자를 그려주세요.")
        else:
            results = []
            for num, ref_img in REF_6.items():
                h = int(np.sum(user_grid != ref_img))
                results.append({"num": num, "hamming": h})
            results.sort(key=lambda x: x["hamming"])
            best = results[0]
            st.success(
                f"✅ 가장 유사한 숫자는 **{best['num']}** 입니다 "
                f"(해밍거리 {best['hamming']})."
            )
            # 순위표
            table = [
                {"순위": i + 1, "숫자": r["num"], "해밍거리": r["hamming"],
                 "유사도(%)": round((1 - r["hamming"] / 36) * 100, 1)}
                for i, r in enumerate(results)
            ]
            st.dataframe(table, use_container_width=True, hide_index=True)
            # 시각화
            fig = plot_diff_panels(user_grid, REF_6, results, best["num"],
                                   show_diff_text=True)
            st.pyplot(fig)

# =========================================================
# 3단계
# =========================================================
else:
    st.header("3단계 · 12×12 해밍거리 + KNN(K=5) 분류")
    st.write(
        "12×12 격자에 숫자를 그리고 [분석 및 수치 확인]을 누르세요. "
        "내부적으로 학습 데이터(원본 숫자를 상하좌우로 1칸씩 이동시킨 50개)를 만들어 "
        "**유클리드 거리** 기준 5-최근접이웃 투표로 분류합니다."
    )

    init_grid("grid12", GRID_SIZE_12)
    train_data = build_train_data()

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("✏️ 입력 그리드 (12×12)")
        render_clickable_grid("grid12", GRID_SIZE_12)
        c1, c2 = st.columns(2)
        run = c1.button("🚀 분석 및 수치 확인", type="primary", use_container_width=True)
        if c2.button("🔄 그리드 초기화", use_container_width=True):
            reset_grid("grid12", GRID_SIZE_12)
            st.rerun()

    with right:
        st.subheader("📐 참조 숫자 (12×12)")
        st.pyplot(plot_reference_grid(REF_12))

    if run:
        user_grid = st.session_state["grid12"]
        if user_grid.sum() == 0:
            st.warning("⚠️ 빈 그리드입니다. 먼저 셀을 클릭해 숫자를 그려주세요.")
        else:
            uvec = user_grid.flatten()

            # KNN 예측
            dists = []
            for s in train_data:
                d = float(np.sqrt(np.sum((uvec - s["vector"]) ** 2)))
                dists.append({"label": s["label"], "dist": d})
            dists.sort(key=lambda x: x["dist"])
            votes = [d["label"] for d in dists[:5]]
            prediction = collections.Counter(votes).most_common(1)[0][0]

            # base reference 거리표
            metric_results = []
            for num in range(10):
                ref_img = REF_12[num]
                ref_vec = ref_img.flatten()
                hamming = int(np.sum(user_grid != ref_img))
                euclidean = float(np.sqrt(np.sum((uvec - ref_vec) ** 2)))
                similarity = round((1 - hamming / 144) * 100, 1)
                metric_results.append({
                    "num": num, "hamming": hamming,
                    "euclidean": round(euclidean, 2),
                    "similarity": similarity,
                })

            sorted_metrics = sorted(metric_results, key=lambda x: x["euclidean"])

            st.success(f"🧠 KNN(K=5) 예측 결과: **{prediction}**")

            # 표
            table = [
                {"순위": i + 1, "숫자": m["num"],
                 "유클리드 거리": m["euclidean"],
                 "해밍 거리(틀린 픽셀)": m["hamming"],
                 "유사도(%)": m["similarity"],
                 "K=5 투표 수": votes.count(m["num"])}
                for i, m in enumerate(sorted_metrics)
            ]
            st.dataframe(table, use_container_width=True, hide_index=True)

            # 시각화
            fig = plot_diff_panels(
                user_grid, REF_12, metric_results, prediction,
                show_diff_text=False,  # 12x12는 너무 빽빽하므로 생략
            )
            st.pyplot(fig)

            with st.expander("🔍 KNN 상세: 가장 가까운 5개 학습 샘플"):
                top5 = [{"순위": i + 1, "라벨": d["label"],
                         "유클리드 거리": round(d["dist"], 3)}
                        for i, d in enumerate(dists[:5])]
                st.dataframe(top5, use_container_width=True, hide_index=True)
