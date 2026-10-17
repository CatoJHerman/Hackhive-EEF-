"""One-click examples using the same in-memory input path as uploads."""
import io
import streamlit as st


def example_input(upload, key, examples):
    """Examples are (label, Path, provenance). Actual uploads always take priority."""
    ready = [(label, path, note) for label, path, note in examples if path.is_file()]
    state_key = key + "_example"
    if ready:
        columns = st.columns(len(ready) + 1)
        for column, (label, path, note) in zip(columns, ready):
            if column.button(label, key=key + "_load_" + path.stem, disabled=upload is not None):
                st.session_state[state_key] = (str(path), note)
        if columns[-1].button("Clear example", key=key + "_clear"):
            st.session_state.pop(state_key, None)
    if upload is not None:
        st.caption("Source: uploaded recording. Acquisition conditions and labels are not independently verified. Remove the upload to enable examples.")
        return upload
    selected = st.session_state.get(state_key)
    if selected:
        from pathlib import Path
        path, note = selected
        try:
            result = io.BytesIO(Path(path).read_bytes())
        except OSError as exc:
            st.warning(f"Example unavailable: {exc}")
            return None
        result.name = Path(path).name
        st.info("Example source: " + note)
        return result
    return None
