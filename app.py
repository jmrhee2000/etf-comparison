"""ETF 비교 분석 대시보드 — Streamlit"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta

from collector import load_all_data, collect_all, update
from analyzer import (
    get_portfolio_data, get_dates, track_entries_exits,
    get_holding_periods, classify_winners_losers,
    compare_holdings, compare_entry_exit_timing,
    get_weight_timeseries, get_all_weight_timeseries,
)
from signals import (
    detect_signals, build_ticker_map, fetch_all_prices,
    backtest_signals, summarize_backtest, get_current_signals,
    find_consensus_signals, find_divergence_signals,
    generate_trading_notes, run_full_analysis,
    build_ticker_map, fetch_all_prices,
)
import config

st.set_page_config(page_title="ETF 비교 분석", layout="wide")
st.title("타임폴리오 vs Koact ETF 비교 분석")


@st.cache_data(ttl=3600)
def load_data():
    tf = get_portfolio_data("timefolio")
    ko = get_portfolio_data("koact")
    return tf, ko


@st.cache_data(ttl=3600, show_spinner="시그널 분석 및 백테스트 중...")
def load_signal_analysis():
    tf_raw = load_all_data("timefolio")
    ko_raw = load_all_data("koact")
    tf = get_portfolio_data("timefolio")
    ko = get_portfolio_data("koact")

    tf_sigs = detect_signals(tf, "Timefolio") if not tf.empty else pd.DataFrame()
    ko_sigs = detect_signals(ko, "Koact") if not ko.empty else pd.DataFrame()
    all_sigs = pd.concat([tf_sigs, ko_sigs], ignore_index=True)

    ticker_map = build_ticker_map(tf_raw, ko_raw)

    dates_all = []
    for df in [tf, ko]:
        if not df.empty:
            dates_all.extend(get_dates(df))
    start_date = min(dates_all).date() if hasattr(min(dates_all), 'date') else min(dates_all)
    end_date = max(dates_all).date() if hasattr(max(dates_all), 'date') else max(dates_all)

    prices = fetch_all_prices(ticker_map, start_date, end_date)
    bt = backtest_signals(all_sigs, prices)
    consensus = find_consensus_signals(tf_sigs, ko_sigs)

    return {
        "tf_sigs": tf_sigs, "ko_sigs": ko_sigs, "all_sigs": all_sigs,
        "bt": bt, "consensus": consensus, "prices": prices,
        "ticker_map": ticker_map,
    }


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("데이터 관리")
    if st.button("최신 데이터 수집 (업데이트)"):
        with st.spinner("수집 중..."):
            update()
            st.cache_data.clear()
        st.success("완료!")

    if st.button("전체 6개월 재수집"):
        with st.spinner("6개월 데이터 수집 중... (약 10분 소요)"):
            collect_all(force=True)
            st.cache_data.clear()
        st.success("완료!")

    st.divider()
    page = st.radio("페이지 선택", [
        "오늘의 브리핑",
        "트레이딩 인사이트",
        "시그널 백테스트",
        "워크포워드 검증",
        "개요",
        "위너/루저 분석",
        "종목별 상세",
        "편입/편출 타이밍",
        "비중 히트맵",
    ])

signal_labels = {
    "NEW_ENTRY": "신규 편입", "EXIT": "완전 편출",
    "BIG_INCREASE": "대량 매수", "BIG_DECREASE": "대량 매도",
    "CONVICTION_BUY": "확신 매수",
}

tf_df, ko_df = load_data()

if tf_df.empty and ko_df.empty:
    st.warning("데이터가 없습니다. 자동으로 최근 6개월 데이터를 수집합니다...")
    with st.spinner("데이터 수집 중... (첫 실행 시 약 5~10분 소요)"):
        collect_all()
        st.cache_data.clear()
    tf_df, ko_df = load_data()
    if tf_df.empty and ko_df.empty:
        st.error("데이터 수집 실패. 네트워크 연결을 확인하세요.")
        st.stop()
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Page: 오늘의 브리핑
# ══════════════════════════════════════════════════════════════════════════════

if page == "오늘의 브리핑":
    sa = load_signal_analysis()
    bt = sa["bt"]
    all_sigs = sa["all_sigs"]
    tf_sigs = sa["tf_sigs"]
    ko_sigs = sa["ko_sigs"]
    consensus = sa["consensus"]
    prices = sa["prices"]

    today_str = date.today().strftime("%Y년 %m월 %d일")
    st.header(f"{today_str} 트레이딩 브리핑")

    # ── Helper: signal stats ─────────────────────────────────────────────
    def _sig_stats(sig_type, ret_col="return_10d"):
        if bt.empty or ret_col not in bt.columns:
            return None
        s = bt[(bt["signal"] == sig_type)].dropna(subset=[ret_col])
        if len(s) < 3:
            return None
        return {
            "n": len(s),
            "wr": (s[ret_col] > 0).mean() * 100,
            "avg": s[ret_col].mean(),
        }

    # ── 1. 매수 고려 종목 ────────────────────────────────────────────────
    st.subheader("매수 고려 종목")

    recent_buy = all_sigs[
        (all_sigs["date"] >= pd.Timestamp(date.today()) - pd.Timedelta(days=7)) &
        (all_sigs["signal"].isin(["BIG_INCREASE", "CONVICTION_BUY", "NEW_ENTRY"]))
    ].sort_values("date", ascending=False)

    if recent_buy.empty:
        st.info("최근 7일 내 매수 시그널이 없습니다.")
    else:
        # De-duplicate by stock, keep the most recent / strongest signal
        signal_priority = {"BIG_INCREASE": 0, "CONVICTION_BUY": 1, "NEW_ENTRY": 2}
        recent_buy = recent_buy.copy()
        recent_buy["_priority"] = recent_buy["signal"].map(signal_priority).fillna(9)
        deduped = recent_buy.sort_values(["norm_name", "_priority", "date"],
                                          ascending=[True, True, False])
        deduped = deduped.drop_duplicates(subset=["norm_name"], keep="first")

        for _, sig in deduped.iterrows():
            stats = _sig_stats(sig["signal"])
            sig_date = pd.Timestamp(sig["date"]).strftime("%m/%d")
            etf_label = "타임폴리오" if sig["etf"] == "Timefolio" else "Koact"

            # Check if both ETFs hold this stock
            both_hold = (
                not tf_df.empty and not ko_df.empty and
                sig["norm_name"] in set(tf_df[tf_df["date"] == get_dates(tf_df)[-1]]["norm_name"]) and
                sig["norm_name"] in set(ko_df[ko_df["date"] == get_dates(ko_df)[-1]]["norm_name"])
            )
            consensus_tag = " | 양 펀드 공통 보유" if both_hold else ""

            sig_name_kr = signal_labels.get(sig["signal"], sig["signal"])

            body = f"**{sig['name']}**\n\n"
            body += f"- **시그널**: {etf_label}가 {sig_date}에 {sig_name_kr} ({sig['detail']})\n"
            if stats:
                body += (f"- **과거 실적**: 이 유형의 시그널 {stats['n']}건 중 "
                         f"10일 후 승률 **{stats['wr']:.0f}%**, "
                         f"평균 수익률 **{stats['avg']:+.1f}%**\n")
            if both_hold:
                # Get weights from both
                tf_latest = tf_df[tf_df["date"] == get_dates(tf_df)[-1]]
                ko_latest = ko_df[ko_df["date"] == get_dates(ko_df)[-1]]
                tf_w = tf_latest[tf_latest["norm_name"] == sig["norm_name"]]
                ko_w = ko_latest[ko_latest["norm_name"] == sig["norm_name"]]
                if not tf_w.empty and not ko_w.empty:
                    body += (f"- **양 펀드 비중**: 타임폴리오 {tf_w.iloc[0]['weight']:.1f}% / "
                             f"Koact {ko_w.iloc[0]['weight']:.1f}%\n")
            body += f"- **판단 근거**: 펀드매니저가 의도적으로 비중을 확대한 종목. "
            if sig["signal"] == "BIG_INCREASE":
                body += ("수량과 비중이 동시에 급증하여 '적극 매수' 판단. "
                         "워크포워드 백테스트 기준 가장 높은 승률을 보이는 시그널.")
            elif sig["signal"] == "CONVICTION_BUY":
                body += "5거래일 연속 비중 증가로 펀드매니저의 확신이 반영된 매수."
            elif sig["signal"] == "NEW_ENTRY":
                body += "신규 편입 종목으로 펀드매니저가 새롭게 주목하는 종목."
            body += consensus_tag

            if stats and stats["wr"] >= 60:
                st.success(body)
            elif stats and stats["wr"] >= 50:
                st.info(body)
            else:
                st.warning(body)

    st.divider()

    # ── 2. 매도/회피 고려 종목 ───────────────────────────────────────────
    st.subheader("매도 / 회피 고려 종목")

    recent_sell = all_sigs[
        (all_sigs["date"] >= pd.Timestamp(date.today()) - pd.Timedelta(days=7)) &
        (all_sigs["signal"].isin(["EXIT", "BIG_DECREASE"]))
    ].sort_values("date", ascending=False)

    if recent_sell.empty:
        st.info("최근 7일 내 매도 시그널이 없습니다.")
    else:
        deduped_sell = recent_sell.drop_duplicates(subset=["norm_name"], keep="first")
        for _, sig in deduped_sell.iterrows():
            stats = _sig_stats(sig["signal"])
            sig_date = pd.Timestamp(sig["date"]).strftime("%m/%d")
            etf_label = "타임폴리오" if sig["etf"] == "Timefolio" else "Koact"
            sig_name_kr = signal_labels.get(sig["signal"], sig["signal"])

            body = f"**{sig['name']}**\n\n"
            body += f"- **시그널**: {etf_label}가 {sig_date}에 {sig_name_kr} ({sig['detail']})\n"
            if stats:
                body += (f"- **과거 실적**: 편출 후 10일간 해당 종목 평균 "
                         f"**{stats['avg']:+.1f}%** 변동 (편출 판단 정확도 {stats['wr']:.0f}%)\n")
            if sig["signal"] == "EXIT":
                body += ("- **판단 근거**: 펀드매니저가 포지션을 완전히 청산한 종목. "
                         "전문 운용사의 편출 판단은 해당 종목의 단기 하락을 "
                         "예고하는 경향이 있음.")
            else:
                body += ("- **판단 근거**: 수량과 비중이 동시에 급감하여 "
                         "펀드매니저가 적극적으로 비중을 축소한 종목.")
            st.error(body)

    st.divider()

    # ── 3. 포트폴리오 변화 요약 ──────────────────────────────────────────
    st.subheader("이번 주 포트폴리오 변화 요약")

    for df, etf_name in [(tf_df, "타임폴리오"), (ko_df, "Koact")]:
        if df.empty:
            continue
        dates = get_dates(df)
        if len(dates) < 2:
            continue

        latest = df[df["date"] == dates[-1]].sort_values("weight", ascending=False)
        prev_5d_idx = max(0, len(dates) - 6)
        prev = df[df["date"] == dates[prev_5d_idx]]
        prev_date_str = pd.Timestamp(dates[prev_5d_idx]).strftime("%m/%d")
        latest_date_str = pd.Timestamp(dates[-1]).strftime("%m/%d")

        prev_stocks = set(prev["norm_name"])
        curr_stocks = set(latest["norm_name"])
        new_in = curr_stocks - prev_stocks
        gone = prev_stocks - curr_stocks

        # Weight changes for continuing stocks
        common = curr_stocks & prev_stocks
        weight_changes = []
        for norm in common:
            c = latest[latest["norm_name"] == norm]
            p = prev[prev["norm_name"] == norm]
            if not c.empty and not p.empty:
                wc = float(c.iloc[0]["weight"]) - float(p.iloc[0]["weight"])
                if abs(wc) > 0.5:
                    weight_changes.append((c.iloc[0]["name"], wc))

        weight_changes.sort(key=lambda x: -abs(x[1]))

        summary = f"**{etf_name}** ({prev_date_str} → {latest_date_str})\n\n"
        summary += f"- 현재 보유: **{len(curr_stocks)}종목**\n"
        if new_in:
            names = [latest[latest["norm_name"]==n].iloc[0]["name"] for n in new_in
                     if not latest[latest["norm_name"]==n].empty]
            summary += f"- 신규 편입: {', '.join(names)}\n"
        if gone:
            names = [prev[prev["norm_name"]==n].iloc[0]["name"] for n in gone
                     if not prev[prev["norm_name"]==n].empty]
            summary += f"- 편출: {', '.join(names)}\n"
        if weight_changes:
            top_up = [f"{n} (+{w:.1f}%p)" for n, w in weight_changes[:3] if w > 0]
            top_down = [f"{n} ({w:.1f}%p)" for n, w in weight_changes if w < 0][:3]
            if top_up:
                summary += f"- 비중 확대: {', '.join(top_up)}\n"
            if top_down:
                summary += f"- 비중 축소: {', '.join(top_down)}\n"

        st.markdown(summary)

    st.divider()

    # ── 4. 전략 성과 현황 ────────────────────────────────────────────────
    st.subheader("시그널 전략 누적 성과")

    if not bt.empty and "return_10d" in bt.columns:
        from signals import walk_forward_backtest, compute_equity_curve

        @st.cache_data(ttl=3600, show_spinner=False)
        def _wf_for_briefing():
            tf = get_portfolio_data("timefolio")
            ko = get_portfolio_data("koact")
            tf_raw = load_all_data("timefolio")
            ko_raw = load_all_data("koact")
            tm = build_ticker_map(tf_raw, ko_raw)
            pr = fetch_all_prices(tm, date(2025, 10, 1), date.today())
            return walk_forward_backtest(tf, ko, pr, tm, hold_days=10)

        wf = _wf_for_briefing()
        if not wf.empty:
            eq = compute_equity_curve(wf)
            total_trades = len(wf)
            wr = (wf["return_pct"] > 0).mean() * 100
            avg_ret = wf["return_pct"].mean()
            cumul = eq["cumulative"].iloc[-1] if not eq.empty else 0

            perf_text = (
                f"워크포워드 백테스트 기준 (10일 보유), "
                f"총 **{total_trades}건** 거래에서 "
                f"승률 **{wr:.0f}%**, 평균 수익률 **{avg_ret:+.2f}%**, "
                f"누적 수익률 **{cumul:+.1f}%**입니다.\n\n"
            )

            # Best signal type
            best_sig = None
            best_wr = 0
            for sig_type in ["BIG_INCREASE", "CONVICTION_BUY", "NEW_ENTRY", "EXIT"]:
                s = wf[wf["signal"] == sig_type]
                if len(s) >= 5:
                    sig_wr = (s["return_pct"] > 0).mean() * 100
                    sig_avg = s["return_pct"].mean()
                    if sig_wr > best_wr:
                        best_wr = sig_wr
                        best_sig = (sig_type, sig_wr, sig_avg, len(s))

            if best_sig:
                sig_label = signal_labels.get(best_sig[0], best_sig[0])
                perf_text += (
                    f"가장 유효한 시그널은 **'{sig_label}'**로, "
                    f"{best_sig[3]}건 중 승률 **{best_sig[1]:.0f}%**, "
                    f"평균 **{best_sig[2]:+.1f}%**입니다."
                )

            st.markdown(perf_text)

    st.divider()

    # ── 5. 주의사항 ──────────────────────────────────────────────────────
    st.caption(
        "이 브리핑은 타임폴리오/Koact ETF 포트폴리오 변화 데이터와 "
        "과거 패턴 분석을 기반으로 생성됩니다. "
        "투자 판단의 참고자료이며, 투자 결과에 대한 책임은 투자자 본인에게 있습니다. "
        "데이터는 1~2 거래일 지연될 수 있습니다."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Page: 트레이딩 인사이트
# ══════════════════════════════════════════════════════════════════════════════

elif page == "트레이딩 인사이트":
    sa = load_signal_analysis()
    bt = sa["bt"]
    all_sigs = sa["all_sigs"]
    tf_sigs = sa["tf_sigs"]
    ko_sigs = sa["ko_sigs"]
    consensus = sa["consensus"]
    prices = sa["prices"]

    # ── Section 1: 최근 액션 시그널 ──────────────────────────────────────────
    st.header("최근 펀드매니저 액션 (최근 7일)")

    recent = get_current_signals(all_sigs, lookback_days=7)
    if recent.empty:
        st.info("최근 7일 내 시그널 없음")
    else:
        # Color-code by signal type
        signal_colors = {
            "NEW_ENTRY": "🟢", "EXIT": "🔴",
            "BIG_INCREASE": "🔵", "BIG_DECREASE": "🟠",
            "CONVICTION_BUY": "🟣",
        }
        signal_labels = {
            "NEW_ENTRY": "신규 편입", "EXIT": "완전 편출",
            "BIG_INCREASE": "대량 매수", "BIG_DECREASE": "대량 매도",
            "CONVICTION_BUY": "확신 매수",
        }

        # Cross-reference with backtest performance
        for _, sig in recent.iterrows():
            sig_bt = bt[bt["signal"] == sig["signal"]]
            if not sig_bt.empty and "return_5d" in sig_bt.columns:
                valid = sig_bt.dropna(subset=["return_5d"])
                if len(valid) >= 3:
                    wr = (valid["return_5d"] > 0).mean() * 100
                    avg_ret = valid["return_5d"].mean()
                    track = f"과거 승률 {wr:.0f}% / 평균 {avg_ret:+.1f}%"
                else:
                    track = "데이터 부족"
            else:
                track = ""

            icon = signal_colors.get(sig["signal"], "⚪")
            label = signal_labels.get(sig["signal"], sig["signal"])
            sig_date = pd.Timestamp(sig["date"]).strftime("%m/%d")

            col1, col2, col3, col4 = st.columns([1, 2, 3, 2])
            col1.write(f"{icon} **{label}**")
            col2.write(f"**{sig['name']}**")
            col3.write(f"{sig['etf']} | {sig_date} | {sig['detail']}")
            col4.caption(track)

    st.divider()

    # ── Section 2: 핵심 발견 / 액셔너블 인사이트 ────────────────────────────
    st.header("백테스트 기반 핵심 발견")

    if bt.empty:
        st.warning("백테스트 데이터 없음")
    else:
        # Compute summaries for different horizons
        insights = []
        for days in [5, 10, 20]:
            col = f"return_{days}d"
            if col not in bt.columns:
                continue
            for sig_type in bt["signal"].unique():
                sig_data = bt[bt["signal"] == sig_type].dropna(subset=[col])
                if len(sig_data) < 5:
                    continue
                wr = (sig_data[col] > 0).mean() * 100
                avg = sig_data[col].mean()
                med = sig_data[col].median()
                n = len(sig_data)
                insights.append({
                    "signal": sig_type,
                    "horizon": f"{days}일",
                    "count": n,
                    "win_rate": wr,
                    "avg_return": avg,
                    "median_return": med,
                })

        if insights:
            ins_df = pd.DataFrame(insights)

            # Show top findings
            st.subheader("유효한 매매 시그널 (승률 55%+, 평균수익률 1%+)")
            good = ins_df[(ins_df["win_rate"] >= 55) & (ins_df["avg_return"] > 1)].sort_values(
                "avg_return", ascending=False)
            if not good.empty:
                for _, row in good.iterrows():
                    label = signal_labels.get(row["signal"], row["signal"])
                    st.success(
                        f"**{label}** 시그널 → {row['horizon']} 후: "
                        f"승률 **{row['win_rate']:.0f}%** | "
                        f"평균수익률 **{row['avg_return']:+.1f}%** | "
                        f"중간값 **{row['median_return']:+.1f}%** "
                        f"(표본 {row['count']}건)"
                    )
            else:
                st.info("아직 통계적으로 유의미한 매수 시그널이 없습니다.")

            st.subheader("역행 패턴 (승률 45% 이하)")
            bad = ins_df[(ins_df["win_rate"] <= 45) & (ins_df["avg_return"] < -0.5)].sort_values(
                "avg_return")
            if not bad.empty:
                for _, row in bad.iterrows():
                    label = signal_labels.get(row["signal"], row["signal"])
                    st.error(
                        f"**{label}** 시그널 → {row['horizon']} 후: "
                        f"승률 **{row['win_rate']:.0f}%** | "
                        f"평균수익률 **{row['avg_return']:+.1f}%** "
                        f"(표본 {row['count']}건)"
                    )

    st.divider()

    # ── Section 3: ETF별 시그널 성과 비교 ────────────────────────────────────
    st.header("ETF별 시그널 정확도")

    if not bt.empty and "return_10d" in bt.columns:
        for etf in ["Timefolio", "Koact"]:
            etf_bt = bt[bt["etf"] == etf]
            if etf_bt.empty:
                continue
            etf_label = "타임폴리오" if etf == "Timefolio" else "Koact"
            st.subheader(f"{etf_label}")

            summary = summarize_backtest(etf_bt, "signal", "return_10d")
            if not summary.empty:
                summary_display = summary.rename(columns={
                    "count": "건수", "avg_return": "평균수익률(%)",
                    "median_return": "중간값(%)", "win_rate": "승률(%)",
                    "avg_win": "평균이익(%)", "avg_loss": "평균손실(%)",
                    "best": "최고(%)", "worst": "최저(%)",
                }).rename(index=signal_labels)
                st.dataframe(summary_display, use_container_width=True)

                # Visual
                fig = go.Figure()
                for sig in summary.index:
                    fig.add_trace(go.Bar(
                        x=[signal_labels.get(sig, sig)],
                        y=[summary.loc[sig, "avg_return"]],
                        marker_color="#2ecc71" if summary.loc[sig, "avg_return"] > 0 else "#e74c3c",
                        text=f"{summary.loc[sig, 'win_rate']:.0f}%",
                        textposition="outside",
                        showlegend=False,
                    ))
                fig.update_layout(
                    title=f"{etf_label} 시그널별 10일 평균 수익률",
                    yaxis_title="평균 수익률 (%)",
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Section 4: 컨센서스 시그널 ───────────────────────────────────────────
    st.header("양 펀드 동시 매매 (컨센서스)")
    st.caption("두 펀드가 5일 이내에 같은 종목을 같은 방향으로 매매한 경우 — 강한 시그널")

    if not consensus.empty:
        cons_display = consensus.copy()
        cons_display["signal"] = cons_display["signal"].map(signal_labels).fillna(cons_display["signal"])
        cons_display["tf_date"] = cons_display["tf_date"].dt.strftime("%Y-%m-%d")
        cons_display["ko_date"] = cons_display["ko_date"].dt.strftime("%Y-%m-%d")
        st.dataframe(
            cons_display[["name", "signal", "tf_date", "ko_date", "tf_weight", "ko_weight"]].rename(
                columns={"name": "종목", "signal": "시그널", "tf_date": "타임폴리오 날짜",
                          "ko_date": "Koact 날짜", "tf_weight": "TF 비중", "ko_weight": "KO 비중"}
            ),
            use_container_width=True,
        )

        # Backtest consensus signals specifically
        if not bt.empty and "return_10d" in bt.columns:
            consensus_norms = set(consensus["norm_name"])
            cons_bt = bt[
                bt["norm_name"].isin(consensus_norms) &
                bt["signal"].isin(consensus["signal"].unique())
            ]
            if not cons_bt.empty:
                valid = cons_bt.dropna(subset=["return_10d"])
                if len(valid) >= 3:
                    wr = (valid["return_10d"] > 0).mean() * 100
                    avg = valid["return_10d"].mean()
                    st.metric("컨센서스 시그널 10일 승률", f"{wr:.0f}%",
                              delta=f"평균 {avg:+.1f}%")
    else:
        st.info("양 펀드 동시 매매 사례 없음")

    st.divider()

    # ── Section 5: 종목별 주가 + 비중 오버레이 ───────────────────────────────
    st.header("비중 변화 vs 실제 주가")
    st.caption("펀드매니저의 비중 변화와 실제 주가를 겹쳐서 타이밍을 확인")

    # Build stock list with prices available
    stocks_with_prices = []
    stock_name_map = {}
    for df in [tf_df, ko_df]:
        if df.empty:
            continue
        for _, row in df[["name", "norm_name"]].drop_duplicates().iterrows():
            if row["norm_name"] in prices:
                stocks_with_prices.append(row["norm_name"])
                stock_name_map[row["norm_name"]] = row["name"]

    stocks_with_prices = sorted(set(stocks_with_prices),
                                 key=lambda n: stock_name_map.get(n, n))
    if stocks_with_prices:
        selected = st.selectbox(
            "종목 선택", stocks_with_prices,
            format_func=lambda n: stock_name_map.get(n, n),
            key="price_overlay_stock"
        )

        price_df = prices[selected].copy()
        price_df["Date"] = pd.to_datetime(price_df["Date"])

        fig = go.Figure()

        # Price line
        fig.add_trace(go.Scatter(
            x=price_df["Date"], y=price_df["Close"],
            name="주가 (USD)", line=dict(color="#333", width=2),
            yaxis="y"
        ))

        # Weight lines
        for df, name, color in [
            (tf_df, "타임폴리오 비중", "#636EFA"),
            (ko_df, "Koact 비중", "#EF553B")
        ]:
            if df.empty:
                continue
            ts = get_weight_timeseries(df, selected)
            if not ts.empty:
                fig.add_trace(go.Scatter(
                    x=ts["date"], y=ts["weight"],
                    name=name, line=dict(color=color, dash="dot"),
                    yaxis="y2"
                ))

        # Signal markers
        stock_sigs = all_sigs[all_sigs["norm_name"] == selected]
        if not stock_sigs.empty:
            buy_sigs = stock_sigs[stock_sigs["signal"].isin(
                ["NEW_ENTRY", "BIG_INCREASE", "CONVICTION_BUY"])]
            sell_sigs = stock_sigs[stock_sigs["signal"].isin(
                ["EXIT", "BIG_DECREASE"])]

            for sigs, color, symbol, label in [
                (buy_sigs, "green", "triangle-up", "매수 시그널"),
                (sell_sigs, "red", "triangle-down", "매도 시그널"),
            ]:
                if sigs.empty:
                    continue
                # Match signal dates to price
                sig_prices = []
                for _, s in sigs.iterrows():
                    sig_dt = pd.Timestamp(s["date"])
                    close_prices = price_df[price_df["Date"] >= sig_dt].head(1)
                    if not close_prices.empty:
                        sig_prices.append({
                            "date": sig_dt,
                            "price": float(close_prices.iloc[0]["Close"]),
                            "detail": f"{s['etf']}: {signal_labels.get(s['signal'], s['signal'])}",
                        })
                if sig_prices:
                    sp_df = pd.DataFrame(sig_prices)
                    fig.add_trace(go.Scatter(
                        x=sp_df["date"], y=sp_df["price"],
                        mode="markers", name=label,
                        marker=dict(color=color, size=12, symbol=symbol),
                        text=sp_df["detail"], hovertemplate="%{text}<br>$%{y:.2f}",
                        yaxis="y"
                    ))

        fig.update_layout(
            height=500,
            yaxis=dict(title="주가 (USD)", side="left"),
            yaxis2=dict(title="비중 (%)", side="right", overlaying="y"),
            legend=dict(orientation="h", y=-0.15),
            title=f"{stock_name_map.get(selected, selected)} — 주가 vs 펀드 비중",
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Page: 시그널 백테스트
# ══════════════════════════════════════════════════════════════════════════════

elif page == "시그널 백테스트":
    sa = load_signal_analysis()
    bt = sa["bt"]

    if bt.empty:
        st.warning("백테스트 데이터 없음. 먼저 데이터를 수집하세요.")
        st.stop()

    st.header("시그널 유형별 수익률 분포")

    horizon = st.select_slider("분석 기간", options=[1, 3, 5, 10, 20],
                                value=10, format_func=lambda x: f"{x}일")
    ret_col = f"return_{horizon}d"

    if ret_col not in bt.columns:
        st.error(f"{ret_col} 데이터 없음")
        st.stop()

    valid_bt = bt.dropna(subset=[ret_col])

    # Box plot by signal type
    fig = px.box(
        valid_bt, x="signal", y=ret_col, color="signal",
        color_discrete_map={
            "NEW_ENTRY": "#3498db", "EXIT": "#e74c3c",
            "BIG_INCREASE": "#2ecc71", "BIG_DECREASE": "#f39c12",
            "CONVICTION_BUY": "#9b59b6",
        },
        title=f"시그널 유형별 {horizon}일 수익률 분포",
        labels={ret_col: f"{horizon}일 수익률 (%)", "signal": "시그널"},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    # Cumulative returns if you followed each signal type
    st.subheader("시그널 추종 시 누적 수익률")
    fig2 = go.Figure()
    for sig_type in valid_bt["signal"].unique():
        sig_data = valid_bt[valid_bt["signal"] == sig_type].sort_values("date")
        if len(sig_data) < 3:
            continue
        cumul = (1 + sig_data[ret_col] / 100).cumprod() * 100 - 100
        fig2.add_trace(go.Scatter(
            x=sig_data["date"], y=cumul,
            name=signal_labels.get(sig_type, sig_type),
            mode="lines+markers",
        ))
    fig2.add_hline(y=0, line_dash="dash", line_color="gray")
    fig2.update_layout(
        title=f"각 시그널 추종 시 누적 수익률 ({horizon}일 보유 가정)",
        yaxis_title="누적 수익률 (%)",
        height=400,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Individual signal details
    st.subheader("개별 시그널 상세")
    sig_filter = st.multiselect("시그널 필터",
                                 options=valid_bt["signal"].unique().tolist(),
                                 default=valid_bt["signal"].unique().tolist())
    etf_filter = st.multiselect("ETF 필터",
                                 options=valid_bt["etf"].unique().tolist(),
                                 default=valid_bt["etf"].unique().tolist())

    filtered = valid_bt[
        (valid_bt["signal"].isin(sig_filter)) &
        (valid_bt["etf"].isin(etf_filter))
    ].sort_values(ret_col, ascending=False)

    display_cols = ["date", "etf", "signal", "name", "weight", "detail",
                    "return_1d", "return_3d", "return_5d", "return_10d", "return_20d"]
    display_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(
        filtered[display_cols].round(2),
        use_container_width=True,
        height=500,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Page: 워크포워드 검증
# ══════════════════════════════════════════════════════════════════════════════

elif page == "워크포워드 검증":
    from signals import walk_forward_backtest, compute_equity_curve

    st.header("워크포워드 백테스트 (Out-of-Sample 검증)")
    st.caption(
        "각 시점에서 '그 시점까지의 데이터만' 사용하여 시그널을 판단하고, "
        "이후 실제 수익률을 측정합니다. 미래 정보를 사용하지 않는 진짜 백테스트입니다."
    )

    sa = load_signal_analysis()
    prices = sa["prices"]
    ticker_map = sa["ticker_map"]

    hold_days = st.select_slider(
        "보유 기간", options=[5, 10, 20], value=10,
        format_func=lambda x: f"{x}일"
    )

    @st.cache_data(ttl=3600, show_spinner="워크포워드 백테스트 실행 중...")
    def run_walkforward(hold):
        tf = get_portfolio_data("timefolio")
        ko = get_portfolio_data("koact")
        tf_raw = load_all_data("timefolio")
        ko_raw = load_all_data("koact")
        tm = build_ticker_map(tf_raw, ko_raw)
        from datetime import date as d
        pr = fetch_all_prices(tm, d(2025, 10, 1), d.today())
        wf = walk_forward_backtest(tf, ko, pr, tm, hold_days=hold)
        return wf

    wf = run_walkforward(hold_days)

    if wf.empty:
        st.warning("데이터 부족")
        st.stop()

    eq = compute_equity_curve(wf)

    # Summary metrics
    total = len(wf)
    wins = (wf["return_pct"] > 0).sum()
    wr = wins / total * 100
    avg_ret = wf["return_pct"].mean()
    cumul = eq["cumulative"].iloc[-1] if not eq.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 거래 수", total)
    c2.metric("승률", f"{wr:.1f}%")
    c3.metric("평균 수익률", f"{avg_ret:+.2f}%")
    c4.metric("누적 수익률", f"{cumul:+.1f}%")

    # Equity curve
    if not eq.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=eq["decision_date"], y=eq["cumulative"],
            fill="tozeroy", name="누적 수익률",
            line=dict(color="#2ecc71" if cumul > 0 else "#e74c3c"),
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            title=f"시그널 추종 전략 누적 수익률 ({hold_days}일 보유)",
            yaxis_title="누적 수익률 (%)",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # By signal type
    st.subheader("시그널 유형별 성과")
    for sig in wf["signal"].unique():
        s = wf[wf["signal"] == sig]
        sig_wr = (s["return_pct"] > 0).mean() * 100
        sig_avg = s["return_pct"].mean()
        label = signal_labels.get(sig, sig)
        color = "green" if sig_avg > 0 else "red"
        st.markdown(
            f"**{label}**: {len(s)}건 | 승률 **{sig_wr:.0f}%** | "
            f"평균 수익률 :{color}[**{sig_avg:+.2f}%**]"
        )

    # By ETF
    st.subheader("ETF별 시그널 정확도 비교")
    for etf in wf["etf"].unique():
        e = wf[wf["etf"] == etf]
        etf_wr = (e["return_pct"] > 0).mean() * 100
        etf_avg = e["return_pct"].mean()
        etf_label = "타임폴리오" if etf == "Timefolio" else "Koact"
        st.metric(etf_label, f"승률 {etf_wr:.0f}%", delta=f"평균 {etf_avg:+.2f}%")

    # Trade detail scatter
    st.subheader("개별 거래 분포")
    fig2 = px.scatter(
        wf, x="decision_date", y="return_pct",
        color="signal", size=abs(wf["return_pct"]).clip(0.5, 20),
        hover_data=["name", "etf", "entry_price", "exit_price"],
        color_discrete_map={
            "NEW_ENTRY": "#3498db", "EXIT": "#e74c3c",
            "BIG_INCREASE": "#2ecc71", "BIG_DECREASE": "#f39c12",
            "CONVICTION_BUY": "#9b59b6",
        },
        title="개별 거래 수익률 분포",
        labels={"return_pct": "수익률 (%)", "decision_date": "거래일"},
    )
    fig2.add_hline(y=0, line_dash="dash", line_color="gray")
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("전체 거래 내역"):
        display = wf.copy()
        display["decision_date"] = pd.to_datetime(display["decision_date"]).dt.strftime("%Y-%m-%d")
        st.dataframe(
            display[["decision_date", "action", "signal", "etf", "name",
                      "entry_price", "exit_price", "return_pct"]].round(2),
            use_container_width=True, height=500,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Page: 개요
# ══════════════════════════════════════════════════════════════════════════════

elif page == "개요":
    col1, col2 = st.columns(2)

    for col, df, etf_name in [(col1, tf_df, "타임폴리오"), (col2, ko_df, "Koact")]:
        with col:
            st.subheader(etf_name)
            if df.empty:
                st.info("데이터 없음")
                continue
            dates = get_dates(df)
            latest = df[df["date"] == dates[-1]].sort_values("weight", ascending=False)
            st.caption(f"기준일: {dates[-1].strftime('%Y-%m-%d')} | 종목 수: {len(latest)}")
            st.dataframe(
                latest[["name", "weight", "quantity"]].reset_index(drop=True),
                use_container_width=True,
                height=400,
            )

    if not tf_df.empty and not ko_df.empty:
        st.subheader("공통 보유 vs 차별 종목")
        comparison = compare_holdings(tf_df, ko_df)

        c1, c2, c3 = st.columns(3)
        c1.metric("공통 종목", len(comparison["common"]))
        c2.metric("타임폴리오만", len(comparison["timefolio_only"]))
        c3.metric("Koact만", len(comparison["koact_only"]))

        if comparison["common"]:
            st.markdown("**공통 보유 종목 비중 비교**")
            common_df = pd.DataFrame(comparison["common"])
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=common_df["name"], x=common_df["tf_weight"],
                name="타임폴리오", orientation="h", marker_color="#636EFA"
            ))
            fig.add_trace(go.Bar(
                y=common_df["name"], x=common_df["ko_weight"],
                name="Koact", orientation="h", marker_color="#EF553B"
            ))
            fig.update_layout(barmode="group", height=max(300, len(common_df)*30),
                              yaxis=dict(autorange="reversed"), margin=dict(l=0))
            st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            if comparison["timefolio_only"]:
                st.markdown("**타임폴리오만 보유**")
                st.dataframe(pd.DataFrame(comparison["timefolio_only"]).sort_values("weight", ascending=False))
        with col_b:
            if comparison["koact_only"]:
                st.markdown("**Koact만 보유**")
                st.dataframe(pd.DataFrame(comparison["koact_only"]).sort_values("weight", ascending=False))


# ══════════════════════════════════════════════════════════════════════════════
# Page: 위너/루저 분석
# ══════════════════════════════════════════════════════════════════════════════

elif page == "위너/루저 분석":
    lookback = st.slider("분석 기간 (일)", 7, 180, 30)

    for df, etf_name in [(tf_df, "타임폴리오"), (ko_df, "Koact")]:
        st.subheader(f"{etf_name} 위너/루저")
        if df.empty:
            st.info("데이터 없음")
            continue

        wl = classify_winners_losers(df, lookback_days=lookback)
        if wl.empty:
            st.info("분석 데이터 부족")
            continue

        cats = wl["category"].value_counts()
        cols = st.columns(5)
        for i, cat in enumerate(["winner", "loser", "new_entry", "exited", "active_increase"]):
            cols[i].metric(cat.replace("_", " ").title(), cats.get(cat, 0))

        wl_display = wl[wl["category"].isin(["winner", "loser", "new_entry", "exited"])].copy()
        color_map = {
            "winner": "#2ecc71", "loser": "#e74c3c",
            "new_entry": "#3498db", "exited": "#95a5a6",
            "active_increase": "#f39c12", "active_decrease": "#e67e22",
            "neutral": "#bdc3c7",
        }
        fig = px.bar(
            wl_display.head(30), x="weight_change", y="name",
            color="category", orientation="h",
            color_discrete_map=color_map,
            title=f"비중 변화 (최근 {lookback}일)",
        )
        fig.update_layout(
            height=max(400, len(wl_display.head(30))*25),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("상세 데이터"):
            st.dataframe(wl.round(2), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Page: 종목별 상세
# ══════════════════════════════════════════════════════════════════════════════

elif page == "종목별 상세":
    all_stocks = set()
    stock_names = {}
    for df in [tf_df, ko_df]:
        if not df.empty:
            for _, row in df[["name", "norm_name"]].drop_duplicates().iterrows():
                all_stocks.add(row["norm_name"])
                stock_names[row["norm_name"]] = row["name"]

    if not all_stocks:
        st.info("데이터 없음")
        st.stop()

    stock_list = sorted(all_stocks, key=lambda n: stock_names.get(n, n))
    display_names = [f"{stock_names.get(n, n)}" for n in stock_list]
    selected_idx = st.selectbox("종목 선택", range(len(stock_list)),
                                format_func=lambda i: display_names[i])
    selected_norm = stock_list[selected_idx]

    col1, col2 = st.columns(2)

    for col, df, etf_name, color in [
        (col1, tf_df, "타임폴리오", "#636EFA"),
        (col2, ko_df, "Koact", "#EF553B")
    ]:
        with col:
            st.subheader(etf_name)
            if df.empty:
                st.info("데이터 없음")
                continue

            ts = get_weight_timeseries(df, selected_norm)
            if ts.empty:
                st.info(f"{stock_names.get(selected_norm, selected_norm)} — 미보유")
                continue

            fig = px.line(ts, x="date", y="weight", title="비중(%) 변화",
                          color_discrete_sequence=[color])
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.line(ts, x="date", y="quantity", title="수량 변화",
                           color_discrete_sequence=[color])
            fig2.update_layout(height=250)
            st.plotly_chart(fig2, use_container_width=True)

            hp = get_holding_periods(df)
            stock_hp = hp[hp["norm_name"] == selected_norm] if not hp.empty else pd.DataFrame()
            if not stock_hp.empty:
                st.markdown("**보유 기간**")
                for _, row in stock_hp.iterrows():
                    status = "보유중" if row["still_held"] else "편출"
                    entry = pd.Timestamp(row["entry_date"]).strftime("%Y-%m-%d")
                    exit_ = pd.Timestamp(row["exit_date"]).strftime("%Y-%m-%d")
                    st.write(f"- {entry} ~ {exit_} ({row['duration_days']}일) [{status}]")
                    st.write(f"  비중: {row['entry_weight']:.2f}% → {row['exit_weight']:.2f}% "
                             f"(최대 {row['max_weight']:.2f}%, 최소 {row['min_weight']:.2f}%)")

    if not tf_df.empty and not ko_df.empty:
        tf_ts = get_weight_timeseries(tf_df, selected_norm)
        ko_ts = get_weight_timeseries(ko_df, selected_norm)
        if not tf_ts.empty and not ko_ts.empty:
            st.subheader("비중 변화 비교")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tf_ts["date"], y=tf_ts["weight"],
                                     name="타임폴리오", line=dict(color="#636EFA")))
            fig.add_trace(go.Scatter(x=ko_ts["date"], y=ko_ts["weight"],
                                     name="Koact", line=dict(color="#EF553B")))
            fig.update_layout(height=350, yaxis_title="비중 (%)")
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Page: 편입/편출 타이밍
# ══════════════════════════════════════════════════════════════════════════════

elif page == "편입/편출 타이밍":
    tab1, tab2 = st.tabs(["보유 기간 (Gantt)", "편입/편출 타이밍 비교"])

    with tab1:
        selected_etf = st.radio("ETF 선택", ["타임폴리오", "Koact"], horizontal=True)
        df = tf_df if selected_etf == "타임폴리오" else ko_df

        if df.empty:
            st.info("데이터 없음")
        else:
            hp = get_holding_periods(df)
            if hp.empty:
                st.info("분석 데이터 부족")
            else:
                hp_filtered = hp[hp["duration_days"] > 1].copy()
                hp_filtered["status_label"] = hp_filtered["still_held"].apply(
                    lambda x: "보유중" if x else "편출")

                fig = px.timeline(
                    hp_filtered, x_start="entry_date", x_end="exit_date",
                    y="name", color="status_label",
                    color_discrete_map={"보유중": "#2ecc71", "편출": "#e74c3c"},
                    title=f"{selected_etf} 종목별 보유 기간",
                )
                fig.update_layout(
                    height=max(500, len(hp_filtered)*22),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if tf_df.empty or ko_df.empty:
            st.info("양쪽 ETF 데이터가 필요합니다")
        else:
            timing = compare_entry_exit_timing(tf_df, ko_df)
            if timing.empty:
                st.info("공통 종목 없음")
            else:
                st.markdown("**편입 타이밍 비교** (음수 = 타임폴리오가 먼저, 양수 = Koact가 먼저)")

                timing_display = timing.dropna(subset=["entry_diff_days"]).copy()
                fig = px.bar(
                    timing_display, x="entry_diff_days", y="name",
                    color="earlier_entry", orientation="h",
                    color_discrete_map={"Timefolio": "#636EFA", "Koact": "#EF553B", "Same": "#95a5a6"},
                    title="편입 시점 차이 (일)",
                )
                fig.update_layout(
                    height=max(400, len(timing_display)*25),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("상세 데이터"):
                    st.dataframe(timing, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Page: 비중 히트맵
# ══════════════════════════════════════════════════════════════════════════════

elif page == "비중 히트맵":
    selected_etf = st.radio("ETF 선택", ["타임폴리오", "Koact"], horizontal=True, key="heatmap_etf")
    df = tf_df if selected_etf == "타임폴리오" else ko_df

    if df.empty:
        st.info("데이터 없음")
    else:
        pivot = get_all_weight_timeseries(df)
        if pivot.empty:
            st.info("데이터 부족")
        else:
            top_n = st.slider("표시할 상위 종목 수", 5, 50, 20)
            avg_weights = pivot.mean().sort_values(ascending=False)
            top_stocks = avg_weights.head(top_n).index.tolist()
            pivot_top = pivot[top_stocks]

            fig = px.imshow(
                pivot_top.T,
                labels=dict(x="날짜", y="종목", color="비중(%)"),
                aspect="auto",
                color_continuous_scale="YlOrRd",
                title=f"{selected_etf} 상위 {top_n} 종목 비중 히트맵",
            )
            fig.update_layout(height=max(400, top_n * 25))
            st.plotly_chart(fig, use_container_width=True)
