from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("OCT_API_URL", "http://localhost:8000").rstrip("/")
DISCLAIMER = (
    "Clinical decision-support prototype only. This application does not provide a diagnosis, "
    "is not cleared for patient care, and requires review by a qualified ophthalmologist."
)


def api_get(path: str):
    response = requests.get(f"{API_URL}{path}", timeout=20)
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner=False, max_entries=256)
def fetch_media(path: str) -> bytes:
    # Fetch server-side: the browser cannot resolve internal API hosts such as http://api:8000.
    response = requests.get(f"{API_URL}{path}", timeout=20)
    response.raise_for_status()
    return response.content


def show_media(column, path: str | None, caption: str, **kwargs) -> None:
    if not path:
        return
    try:
        column.image(fetch_media(path), caption=caption, **kwargs)
    except requests.RequestException as exc:
        column.warning(f"{caption} is unavailable: {exc}")


def saliency_caption(saliency_class: str | None) -> str:
    target = f" for {saliency_class}" if saliency_class else ""
    return f"Grad-CAM attention overlay{target} (supporting evidence, not an explanation)"


def render_evidence(result: dict) -> None:
    confidence = result.get("confidence")
    st.metric("Model confidence", "Not available" if confidence is None else f"{confidence:.1%}")
    st.write("Findings driving priority:", ", ".join(result["findings"]))
    if result.get("escalation_reasons"):
        st.warning("Escalated because: " + ", ".join(result["escalation_reasons"]))
    if result.get("requires_repeat_acquisition"):
        st.warning("The quality gate recommends repeat acquisition and immediate human review.")
    if result.get("class_probabilities"):
        st.caption(
            "Aggregated class evidence (max across views for pathology, min for NORMAL; "
            "not calibrated probabilities)"
        )
        st.dataframe(
            pd.DataFrame(
                sorted(result["class_probabilities"].items(), key=lambda item: -item[1]),
                columns=["class", "evidence"],
            ),
            hide_index=True,
        )


def submit_scan(uploaded, study_id: str):
    files = {
        "file": (
            uploaded.name,
            uploaded.getvalue(),
            uploaded.type or "application/octet-stream",
        )
    }
    data = {"study_id": study_id} if study_id.strip() else {}
    response = requests.post(f"{API_URL}/v1/triage", files=files, data=data, timeout=180)
    response.raise_for_status()
    return response.json()


def tier_badge(tier: str) -> None:
    if tier == "URGENT":
        st.error(f"Priority: {tier}")
    elif tier == "SEMI_URGENT":
        st.warning(f"Priority: {tier}")
    else:
        st.success(f"Priority: {tier}")


def render_result(result: dict) -> None:
    tier_badge(result["triage_tier"])
    render_evidence(result)
    columns = st.columns(2)
    show_media(columns[0], result["thumbnail_url"], "Submitted scan")
    show_media(
        columns[1], result.get("saliency_map_url"), saliency_caption(result.get("saliency_class"))
    )
    st.caption(f"Model: {result['model_version']} · Audit ID: {result['audit_id']}")
    st.error(result["disclaimer"])


def render_worklist() -> None:
    st.subheader("Prioritized worklist")
    try:
        payload = api_get("/v1/worklist")
    except requests.RequestException as exc:
        st.info(f"The worklist API is not available: {exc}")
        return
    items = payload["items"]
    if not items:
        st.info("No scans have been submitted.")
        return
    table = pd.DataFrame(
        [
            {
                "study_id": item["study_id"],
                "tier": item["effective_tier"],
                "model_tier": item["triage_tier"],
                "confidence": item["confidence"],
                "findings": ", ".join(item["findings"]),
                "received": item["created_at"],
                "audit_id": item["audit_id"],
            }
            for item in items
        ]
    )
    st.dataframe(table, hide_index=True, use_container_width=True)

    selected = st.selectbox(
        "Select a scan for clinician review",
        options=items,
        format_func=lambda item: f"{item['effective_tier']} · {item['study_id']}",
    )
    try:
        detail = api_get(f"/v1/predictions/{selected['audit_id']}")
    except requests.RequestException as exc:
        st.error(f"The audit record could not be loaded: {exc}")
        return
    prediction = detail["prediction"]
    tier_badge(detail["effective_tier"])
    if detail["effective_tier"] != prediction["triage_tier"]:
        st.caption(f"Model tier was {prediction['triage_tier']}; changed by clinician override.")
    columns = st.columns(2)
    show_media(columns[0], selected["thumbnail_url"], "Submitted scan", width=420)
    show_media(
        columns[1],
        selected.get("saliency_map_url"),
        saliency_caption(prediction.get("saliency_class")),
        width=420,
    )
    render_evidence(prediction)
    if detail["overrides"]:
        st.markdown("#### Override history")
        st.dataframe(
            pd.DataFrame(detail["overrides"])[
                ["timestamp", "original_tier", "new_tier", "clinician_id", "reason"]
            ],
            hide_index=True,
            use_container_width=True,
        )
    st.caption(f"Model: {prediction['model_version']} · Audit ID: {prediction['audit_id']}")

    with st.form("clinician_override"):
        st.markdown("#### Clinician override")
        clinician_id = st.text_input("Clinician identifier")
        new_tier = st.selectbox("Reviewed priority", ["URGENT", "SEMI_URGENT", "ROUTINE"])
        reason = st.text_area("Reason for override")
        submitted = st.form_submit_button("Record override")
        if submitted:
            try:
                response = requests.post(
                    f"{API_URL}/v1/overrides",
                    json={
                        "audit_id": selected["audit_id"],
                        "new_tier": new_tier,
                        "clinician_id": clinician_id,
                        "reason": reason,
                    },
                    timeout=20,
                )
                response.raise_for_status()
                st.success("Override recorded in the audit log.")
                st.rerun()
            except requests.RequestException as exc:
                st.error(f"Override was not recorded: {exc}")


def main() -> None:
    st.set_page_config(page_title="OCT Triage Worklist", page_icon="👁️", layout="wide")
    st.title("Retinal OCT Triage Worklist")
    st.error(DISCLAIMER)
    upload_tab, worklist_tab = st.tabs(["Submit scan", "Worklist and review"])
    with upload_tab:
        st.write("Upload one OCT B-scan image or DICOM study for queue-priority decision support.")
        study_id = st.text_input("Study identifier (use a de-identified value)")
        uploaded = st.file_uploader(
            "OCT scan",
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "dcm", "dicom"],
        )
        if st.button("Run triage", type="primary", disabled=uploaded is None):
            try:
                result = submit_scan(uploaded, study_id)
                st.session_state["latest_result"] = result
            except requests.RequestException as exc:
                detail = (
                    getattr(exc.response, "text", str(exc))
                    if exc.response is not None
                    else str(exc)
                )
                st.error(f"The scan was not processed: {detail}")
        if "latest_result" in st.session_state:
            render_result(st.session_state["latest_result"])
    with worklist_tab:
        render_worklist()


if __name__ == "__main__":
    main()
