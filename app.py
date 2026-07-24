from __future__ import annotations

import datetime as dt

import streamlit as st

from gewissbahn.gtfs import loader, station_mapping
from gewissbahn.pipeline import plan_journey

st.set_page_config(page_title="Gewissbahn-GB", page_icon="🚆", layout="centered")


@st.cache_resource
def load_gtfs():
    con = loader.connect()
    mapping = station_mapping.get_mapping(con)
    return con, mapping


gtfs_con, mapping = load_gtfs()
station_names = sorted(mapping["station_name"].dropna().unique().tolist())


def _default_index(preferred: str) -> int:
    return station_names.index(preferred) if preferred in station_names else 0


st.title("🚆 Gewissbahn-GB")
st.caption("Reliability-aware train routing for Germany — ranks connections by the odds you actually make every transfer, not just by scheduled time.")

col1, col2 = st.columns(2)
with col1:
    origin = st.selectbox("From", station_names, index=_default_index("Köln Hbf"))
with col2:
    destination = st.selectbox("To", station_names, index=_default_index("Berlin Hbf"))

col3, col4, col5 = st.columns(3)
with col3:
    date = st.date_input("Date", value=dt.date.today())
with col4:
    time = st.time_input("Depart after", value=dt.datetime.now().time().replace(second=0, microsecond=0))
with col5:
    count = st.slider("Options", min_value=1, max_value=5, value=3)

search = st.button("Search", type="primary", use_container_width=True)

if search:
    when = dt.datetime.combine(date, time)
    with st.spinner("Searching connections, checking live data, scoring reliability..."):
        result = plan_journey(origin, destination, when, count=count, gtfs_con=gtfs_con, mapping=mapping)

    if result.error:
        st.error(result.error)
    else:
        for s in result.summaries:
            with st.container(border=True):
                st.subheader(f"{s.label}  ·  {s.departure} → {s.arrival}  ({s.duration_min} min)")
                badge = "🟢" if s.confidence >= 0.7 else ("🟡" if s.confidence >= 0.4 else "🔴")
                st.write(f"{badge} **{s.confidence:.0%} confidence**  ·  {s.n_transfers} transfer(s)")
                for d in s.leg_details:
                    st.markdown(f"🚆 {d}")
                for d in s.transfer_details:
                    st.markdown(f"🔀 {d}")
                if s.live_note:
                    if "no live confirmation" in s.live_note:
                        st.warning(s.live_note)
                    elif "delay" in s.live_note:
                        st.warning(f"⏱ {s.live_note}")
                    else:
                        st.success(s.live_note)

        st.subheader("🧭 Recommendation")
        st.write(result.explanation)

st.divider()
st.caption(f"{len(station_names)} stations loaded from the historical dataset, mapped to the nationwide GTFS network.")
