# -*- coding: utf-8 -*-
"""
이미지 해밍거리 계산 (Streamlit 버전)
- 파이썬 코드 개발: 대전대신고 하진수
- 웹 구현: 서라벌고 윤진석
- 표기 규칙(교과서): 흰색 = 1, 검은색 = 0
"""

import collections
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# =============================================================================
# 페이지 설정
# =============================================================================
st.set_page_config(
    page_title="이미지 해밍거리 계산",
    page_icon="🔢",
    layout="wide",
)

st.title("🔢 이미지 해밍거리 계산")
st.caption("파이썬 코드 개발: 대전대신고 하진수 · 웹 구현: 서라벌고 윤진석")
st.caption("표기 규칙(교과서) : **흰색 = 1, 검은색 = 0**")

# =============================================================================
# 공용 데이터 (흰=1, 검=0)
# =============================================================================
# 6x6 참조 숫자 (교과서 표기: 흰=1, 검=0)
reference_digits_6 = {
    0: np.array([
        [1,0,0,0,0,1], [0,1,1,1,1,0], [0,1,1,1,1,0],
        [0,1,1,1,1,0], [0,1,1,1,1,0], [1,0,0,0,0,1]
    ]),
    1: np.array([
        [1,1,0,0,1,1], [1,1,1,0,1,1], [1,1,1,0,1,1],
        [1,1,1,0,1,1], [1,1,1,0,1,1], [1,1,0,0,0,1]
    ]),
    2: np.array([
        [1,0,0,0,0,1], [0,1,1,1,1,0], [1,1,1,1,0,1],
        [1,1,1,0,1,1], [1,1,0,1,1,1], [0,0,0,0,0,0]
    ]),
    3: np.array([
        [1,0,0,0,0,1], [1,1,1,1,1,0], [1,1,0,0,0,1],
        [1,1,1,1,1,0], [1,1,1,1,1,0], [1,0,0,0,0,1]
    ]),
    4: np.array([
        [1,1,1,0,1,1], [1,1,0,0,1,1], [1,0,1,0,1,1],
        [0,0,0,0,0,0], [1,1,1,0,1,1], [1,1,1,0,1,1]
    ]),
    5: np.array([
        [0,0,0,0,0,0], [0,1,1,1,1,1], [0,0,0,0,0,1],
        [1,1,1,1,1,0], [0,1,1,1,1,0], [1,0,0,0,0,1]
    ]),
    6: np.array([
        [1,0,0,0,0,1], [0,1,1,1,1,1], [0,0,0,0,0,1],
        [0,1,1,1,1,0], [0,1,1,1,1,0], [1,0,0,0,0,1]
    ]),
    7: np.array([
        [0,0,0,0,0,0], [1,1,1,1,1,0], [1,1,1,1,0,1],
        [1,1,1,0,1,1], [1,1,0,1,1,1], [1,0,1,1,1,1]
    ]),
    8: np.array([
        [1,0,0,0,0,1], [0,1,1,1,1,0], [1,0,0,0,0,1],
        [0,1,1,1,1,0], [0,1,1,1,1,0], [1,0,0,0,0,1]
    ]),
    9: np.array([
        [1,0,0,0,0,1], [0,1,1,1,1,0], [0,1,1,1,1,0],
        [1,0,0,0,0,0], [1,1,1,1,1,0], [1,0,0,0,0,1]
    ]),
}

# 12x12 base reference - 8x12 원본을 좌우 2칸 패딩하여 만든다
GRID_SIZE = 12

_base_8x12 = {
    0: np.array([[1,1,0,0,0,0,1,1],[1,0,0,1,1,0,0,1],[0,0,1,1,1,1,0,0],[0,0,1,1,1,1,0,0],[0,0,1,1,1,1,0,0],[0,0,1,1,1,1,0,0],[0,0,1,1,1,1,0,0],[0,0,1,1,1,1,0,0],[0,0,1,1,1,1,0,0],[1,0,0,1,1,0,0,1],[1,1,0,0,0,0,1,1],[1,1,1,1,1,1,1,1]]),
    1: np.array([[1,1,1,0,0,1,1,1],[1,1,0,0,0,1,1,1],[1,1,1,0,0,1,1,1],[1,1,1,0,0,1,1,1],[1,1,1,0,0,1,1,1],[1,1,1,0,0,1,1,1],[1,1,1,0,0,1,1,1],[1,1,1,0,0,1,1,1],[1,1,1,0,0,1,1,1],[1,0,0,0,0,0,0,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1]]),
    2: np.array([[1,1,0,0,0,0,1,1],[1,0,0,1,1,0,0,1],[0,0,1,1,1,1,0,0],[1,1,1,1,1,1,0,0],[1,1,1,1,1,0,0,1],[1,1,1,1,0,0,1,1],[1,1,1,0,0,1,1,1],[1,1,0,0,1,1,1,1],[1,0,0,1,1,1,1,1],[0,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1]]),
    3: np.array([[1,1,0,0,0,0,1,1],[1,0,0,1,1,0,0,1],[1,1,1,1,1,1,0,0],[1,1,1,1,0,0,0,1],[1,1,1,1,1,1,0,0],[1,1,1,1,1,1,0,0],[1,1,1,1,1,1,0,0],[0,0,1,1,1,1,0,0],[1,0,0,1,1,0,0,1],[1,1,0,0,0,0,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1]]),
    4: np.array([[1,1,1,1,0,0,1,1],[1,1,1,0,0,0,1,1],[1,1,0,0,1,0,0,1],[1,0,0,1,1,0,0,1],[0,0,0,0,0,0,0,0],[1,1,1,1,1,0,0,1],[1,1,1,1,1,0,0,1],[1,1,1,1,1,0,0,1],[1,1,1,1,1,0,0,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1]]),
    5: np.array([[0,0,0,0,0,0,0,1],[0,0,1,1,1,1,1,1],[0,0,0,0,0,0,1,1],[1,1,1,1,1,0,0,1],[1,1,1,1,1,0,0,1],[1,1,1,1,1,0,0,1],[1,1,1,1,1,0,0,1],[0,0,1,1,1,1,0,0],[1,0,0,1,1,0,0,1],[1,1,0,0,0,0,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1]]),
    6: np.array([[1,1,0,0,0,0,1,1],[1,0,0,1,1,1,1,1],[0,0,1,1,1,1,1,1],[0,0,0,0,0,0,1,1],[0,0,1,1,1,0,0,1],[0,0,1,1,1,0,0,1],[0,0,1,1,1,0,0,1],[0,0,1,1,1,0,0,1],[1,0,0,1,1,0,0,1],[1,1,0,0,0,0,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1]]),
    7: np.array([[0,0,0,0,0,0,0,0],[1,1,1,1,1,0,0,1],[1,1,1,1,0,0,1,1],[1,1,1,1,0,0,1,1],[1,1,1,0,0,1,1,1],[1,1,1,0,0,1,1,1],[1,1,0,0,1,1,1,1],[1,1,0,0,1,1,1,1],[1,1,0,0,1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1]]),
    8: np.array([[1,1,0,0,0,0,1,1],[1,0,0,1,1,0,0,1],[0,0,1,1,1,1,0,0],[1,0,0,1,1,0,0,1],[1,1,0,0,0,0,1,1],[1,0,0,1,1,0,0,1],[0,0,1,1,1,1,0,0],[0,0,1,1,1,1,0,0],[1,0,0,1,1,0,0,1],[1,1,0,0,0,0,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1]]),
    9: np.array([[1,1,0,0,0,0,1,1],[1,0,0,1,1,0,0,1],[0,0,1,1,1,1,0,0],[0,0,1,1,1,1,0,0],[1,0,0,1,1,0,0,0],[1,1,0,0,0,0,0,0],[1,1,1,1,1,0,0,1],[1,1,1,1,1,0,0,1],[1,0,0,1,1,0,0,1],[1,1,0,0,0,0,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1]]),
}

def pad_to_12x12(img_8x12: np.ndarray) -> np.ndarray:
    """좌우 2칸씩 흰색(1) 패딩하여 12x12 로 만든다."""
    padded = np.ones((12, 12), dtype=int)
    padded[:, 2:10] = img_8x12
    return padded

base_references_12 = {k: pad_to_12x12(v) for k, v in _base_8x12.items()}

# =============================================================================
# 데이터 증강 (12x12 KNN 용)
# =============================================================================
def shift_image(img: np.ndarray, shift_r: int, shift_c: int) -> np.ndarray:
    """빈 공간을 흰색(1)로 채우며 시프트."""
    shifted = np.ones_like(img)
    r_start = max(0, shift_r); r_end = min(GRID_SIZE, GRID_SIZE + shift_r)
    c_start = max(0, shift_c); c_end = min(GRID_SIZE, GRID_SIZE + shift_c)
    img_r_start = max(0, -shift_r); img_r_end = min(GRID_SIZE, GRID_SIZE - shift_r)
    img_c_start = max(0, -shift_c); img_c_end = min(GRID_SIZE, GRID_SIZE - shift_c)
    shifted[r_start:r_end, c_start:c_end] = img[img_r_start:img_r_end, img_c_start:img_c_end]
    return shifted

@st.cache_data(show_spinner=False)
def build_train_data():
    data = []
    for num, base_img in base_references_12.items():
        for sr, sc in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
            shifted = shift_image(base_img, sr, sc)
            data.append({
                "label": num,
                "vector": shifted.flatten(),
                "image": shifted,
            })
    return data

# =============================================================================
# 시각화 도우미
# =============================================================================
def draw_grid_lines(ax, n):
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    ax.set_xticks([]); ax.set_yticks([])

def render_reference_panel(ref_dict, n, title="참조 숫자 모양 (0~9)"):
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    for num, img in ref_dict.items():
        ax = axes[num]
        # 흰=1, 검=0 → cmap='gray' + vmin/vmax 그대로 사용
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Digit {num}")
        draw_grid_lines(ax, n)
    fig.suptitle(title)
    fig.tight_layout()
    return fig

def render_diff_panel(user_grid, ref_dict, results, best_num, n):
    """
    각 숫자에 대해
      - 둘 다 검정(0)인 칸 -> 진한 회색
      - 차이가 있는 칸 -> 연한 빨강
      - 그 외 -> 흰색
    """
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    axes = axes.flatten()

    for res in results:
        num = res["num"]
        ref_img = ref_dict[num]
        ax = axes[num]

        visual_img = np.ones((n, n, 3))  # 흰색 배경

        diff_mask = (user_grid != ref_img)
        diff_rows, diff_cols = np.where(diff_mask)
        visual_img[diff_rows, diff_cols] = [1, 0.5, 0.5]   # 차이: 연한 빨강

        # 일치 + 둘 다 '검정(0)'
        match_rows, match_cols = np.where((user_grid == 0) & (ref_img == 0))
        visual_img[match_rows, match_cols] = [0.2, 0.2, 0.2]

        ax.imshow(visual_img)

        # 차이 칸 위에 '1' 텍스트
        for dr, dc in zip(diff_rows, diff_cols):
            ax.text(dc, dr, "1", ha="center", va="center",
                    color="darkred", fontweight="bold", fontsize=8)

        is_best = (num == best_num)
        title_color = "blue" if is_best else "black"
        fw = "bold" if is_best else "normal"
        ax.set_title(f"Number {num}\nDist: {res['hamming']}",
                     color=title_color, fontweight=fw)
        draw_grid_lines(ax, n)

    fig.tight_layout()
    return fig

def render_diff_panel_with_metrics(user_grid, ref_dict, metrics, best_num, n):
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))
    axes = axes.flatten()

    for m in metrics:
        num = m["num"]
        ref_img = ref_dict[num]
        ax = axes[num]

        visual_img = np.ones((n, n, 3))
        diff_mask = (user_grid != ref_img)
        diff_rows, diff_cols = np.where(diff_mask)
        visual_img[diff_rows, diff_cols] = [1, 0.6, 0.6]

        match_rows, match_cols = np.where((user_grid == 0) & (ref_img == 0))
        visual_img[match_rows, match_cols] = [0.2, 0.2, 0.2]

        ax.imshow(visual_img)

        is_best = (num == best_num)
        title_color = "blue" if is_best else "black"
        fw = "bold" if is_best else "normal"

        info_text = f"Num {num}\nHamming: {m['hamming']}"
        ax.set_title(info_text, color=title_color, fontweight=fw, fontsize=10)
        draw_grid_lines(ax, n)

    fig.tight_layout()
    return fig

# =============================================================================
# 그리드 입력 컴포넌트
# =============================================================================
def render_clickable_grid(state_key: str, n: int):
    """
    n x n 클릭 토글 그리드.
    상태값: 1 = 흰색(기본), 0 = 검정(클릭됨)
    버튼 라벨은 검정칸일 때만 '■' 로 표시.
    """
    if state_key not in st.session_state:
        st.session_state[state_key] = np.ones((n, n), dtype=int)

    grid = st.session_state[state_key]

    # 상단 도구 버튼
    c1, c2, _ = st.columns([1, 1, 6])
    with c1:
        if st.button("🧹 초기화", key=f"clear_{state_key}"):
            st.session_state[state_key] = np.ones((n, n), dtype=int)
            st.rerun()
    with c2:
        if st.button("⬛ 전체 검정", key=f"fill_{state_key}"):
            st.session_state[state_key] = np.zeros((n, n), dtype=int)
            st.rerun()

    # n x n 버튼 그리드
    for i in range(n):
        cols = st.columns(n, gap="small")
        for j in range(n):
            is_black = (grid[i, j] == 0)
            label = "■" if is_black else " "
            btn_type = "primary" if is_black else "secondary"
            if cols[j].button(label, key=f"{state_key}_{i}_{j}", type=btn_type):
                # 1 <-> 0 토글
                st.session_state[state_key][i, j] = 0 if grid[i, j] == 1 else 1
                st.rerun()

    return st.session_state[state_key]

# =============================================================================
# 사이드바: 단계 선택
# =============================================================================
with st.sidebar:
    st.header("📚 단계 선택")
    step = st.radio(
        "단계를 골라 진행하세요.",
        [
            "1단계 · 6×6 숫자 이미지 미리보기",
            "2단계 · 6×6 해밍거리 구하기",
            "3단계 · 12×12 해밍거리 + KNN 분류",
        ],
        index=0,
    )
    st.markdown("---")
    st.markdown("**표기 규칙**\n- 흰색 = `1`\n- 검은색 = `0`")
    st.caption("그리드를 클릭하면 흰칸이 검정칸(=0)으로 토글됩니다.")

# =============================================================================
# 1단계
# =============================================================================
if step.startswith("1단계"):
    st.header("1단계 · 6×6 숫자 이미지 미리보기")
    st.write("아래는 0~9에 해당하는 **6×6 비트맵 정답지**입니다. (흰=1, 검=0)")
    fig = render_reference_panel(reference_digits_6, 6)
    st.pyplot(fig)

    with st.expander("🧮 참조 숫자끼리의 해밍거리 표 보기 (디자인 한계 확인용)"):
        nums = list(range(10))
        mat = np.zeros((10, 10), dtype=int)
        for a in nums:
            for b in nums:
                mat[a, b] = int(np.sum(reference_digits_6[a] != reference_digits_6[b]))
        import pandas as pd
        df = pd.DataFrame(mat, index=[f"{i}" for i in nums], columns=[f"{i}" for i in nums])
        st.dataframe(df, use_container_width=True)
        st.info(
            "참조 디자인 자체가 매우 비슷한 쌍이 있어(예: 6과 8, 5와 6 등) "
            "사용자가 약간만 다르게 그려도 다른 숫자로 분류될 수 있습니다. "
            "이는 코드 결함이 아니라 6×6 비트맵의 표현력 한계 때문입니다."
        )

# =============================================================================
# 2단계
# =============================================================================
elif step.startswith("2단계"):
    st.header("2단계 · 6×6 해밍거리 구하기")
    st.write("아래 6×6 격자를 클릭해 숫자를 그린 뒤 **분석 실행** 버튼을 누르세요.")
    st.caption("기본은 흰색(1). 클릭하면 검정(0)으로 바뀌고, 다시 누르면 흰색으로 돌아옵니다.")

    user_grid = render_clickable_grid("grid6", 6)

    if st.button("🚀 해밍 거리 분석 실행", type="primary"):
        results = []
        for num, ref_img in reference_digits_6.items():
            diff_matrix = (user_grid != ref_img)
            hamming_dist = int(np.sum(diff_matrix))
            results.append({"num": num, "hamming": hamming_dist})

        results_sorted = sorted(results, key=lambda x: x["hamming"])
        best = results_sorted[0]

        st.success(f"✅ 가장 유사한 숫자는 **'{best['num']}'** 입니다. (해밍거리: {best['hamming']})")

        st.subheader("📊 숫자별 해밍거리 (오름차순)")
        import pandas as pd
        df = pd.DataFrame(results_sorted)
        df.index = range(1, len(df) + 1)
        df.index.name = "순위"
        df = df.rename(columns={"num": "숫자", "hamming": "해밍거리(틀린 픽셀 수)"})
        st.dataframe(df, use_container_width=True)

        st.subheader("🖼 차이 시각화 (빨강=차이, 회색=둘 다 검정)")
        # 0~9 순서로 그리기
        results_by_num = sorted(results, key=lambda x: x["num"])
        fig = render_diff_panel(user_grid, reference_digits_6, results_by_num, best["num"], 6)
        st.pyplot(fig)
    else:
        st.info("그리드를 그린 뒤 위 버튼을 누르면 분석 결과가 표시됩니다.")

# =============================================================================
# 3단계
# =============================================================================
else:
    st.header("3단계 · 12×12 해밍거리 + KNN(K-최근접이웃) 분류")
    st.write(
        "12×12 격자에 숫자를 그리면 **해밍거리 기반 KNN(K=5)** 으로 가장 가까운 숫자를 추정합니다. "
        "학습 데이터는 0~9 각 숫자를 상하좌우로 한 칸씩 시프트한 5개 변형(총 50개)을 사용합니다."
    )
    st.caption("기본은 흰색(1). 클릭으로 검정(0) 토글.")

    user_grid = render_clickable_grid("grid12", GRID_SIZE)

    if st.button("🧠 분석 및 수치 확인", type="primary"):
        train_data = build_train_data()
        input_vec = user_grid.flatten()

        # KNN: 해밍거리 기반
        distances = []
        for sample in train_data:
            d_h = int(np.sum(input_vec != sample["vector"]))
            distances.append({"label": sample["label"], "hamming": d_h})
        distances.sort(key=lambda x: x["hamming"])

        votes = [d["label"] for d in distances[:5]]
        prediction = collections.Counter(votes).most_common(1)[0][0]

        # base_reference 와의 거리 (해밍, 유사도)
        metrics = []
        for num in range(10):
            ref_img = base_references_12[num]
            hamming = int(np.sum(user_grid != ref_img))
            similarity = (1 - hamming / (GRID_SIZE * GRID_SIZE)) * 100
            metrics.append({
                "num": num,
                "hamming": hamming,
                "similarity": similarity,
            })

        sorted_metrics = sorted(metrics, key=lambda x: x["hamming"])

        st.success(f"🧠 가장 유사한 숫자는 **[{prediction}]** 입니다. (KNN K=5, 해밍거리 기반)")

        st.subheader("📊 숫자별 해밍거리 순위 (상위 5개)")
        import pandas as pd
        df = pd.DataFrame(sorted_metrics[:5])
        df.index = range(1, len(df) + 1)
        df.index.name = "순위"
        df = df.rename(columns={
            "num": "숫자",
            "hamming": "해밍거리(픽셀차)",
            "similarity": "유사도(%)",
        })
        df["유사도(%)"] = df["유사도(%)"].map(lambda v: f"{v:.1f}%")
        st.dataframe(df, use_container_width=True)

        with st.expander("🔍 KNN 상세: 가장 가까운 5개 학습 샘플"):
            knn_df = pd.DataFrame(distances[:5])
            knn_df.index = range(1, len(knn_df) + 1)
            knn_df.index.name = "순위"
            knn_df = knn_df.rename(columns={"label": "라벨", "hamming": "해밍거리"})
            st.dataframe(knn_df, use_container_width=True)

        st.subheader("🖼 base reference 와의 차이 시각화")
        fig = render_diff_panel_with_metrics(
            user_grid, base_references_12, metrics, prediction, GRID_SIZE
        )
        st.pyplot(fig)
    else:
        st.info("12×12 그리드에 숫자를 그린 뒤 위 버튼을 누르세요.")

st.markdown("---")
st.caption("© 대전대신고 하진수(코드) · 서라벌고 윤진석(웹). Streamlit으로 구동됩니다.")
