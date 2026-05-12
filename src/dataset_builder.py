"""Project dataset builder for synthetic warehouse, handwritten-style validation, and CFPB-style benchmark data."""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

from config import CFPB_DATA_PATH, RAW_DATA_PATH, VALIDATION_DATA_PATH
from src.data_generator import ISSUE_TEMPLATES, REGIONS, COUNTRIES, CUSTOMER_SEGMENTS, SENTIMENTS, ESCALATION_BY_PRIORITY, SLA_BY_PRIORITY, random_date_text


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_raw_rows(total_rows: int = 150_000, seed: int = 42) -> list[dict[str, object]]:
    rng = random.Random(seed)
    start_date = datetime(2024, 1, 1)
    rows: list[dict[str, object]] = []
    for complaint_id in range(1, total_rows + 1):
        template = ISSUE_TEMPLATES[(complaint_id - 1) % len(ISSUE_TEMPLATES)]
        created_at = start_date + timedelta(minutes=complaint_id * 17 + rng.randint(0, 45))
        ticket = f"CMP-{complaint_id:06d}"
        complaint_text = rng.choice(template.complaint_patterns).format(
            amount=f"${rng.randint(19, 1499)}",
            days=rng.randint(1, 30),
            product_line=rng.choice(template.product_lines),
            ticket=ticket,
            region=rng.choice(REGIONS),
            date_text=random_date_text(created_at, rng),
        )
        complaint_text = f"{complaint_text} I am a {rng.choice(CUSTOMER_SEGMENTS).lower()} customer and this feels {rng.choice(SENTIMENTS).lower()}."
        if template.priority in {"High", "Critical"} and rng.random() < 0.5:
            complaint_text += " Please escalate this immediately."
        rows.append(
            {
                "complaint_id": complaint_id,
                "complaint_text": complaint_text,
                "true_department": template.department,
                "true_priority": template.priority,
                "true_core_issue": template.issue,
                "channel": rng.choice(template.channels),
                "product_line": rng.choice(template.product_lines),
                "customer_segment": rng.choice(CUSTOMER_SEGMENTS),
                "region": rng.choice(REGIONS),
                "country": rng.choice(COUNTRIES),
                "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "customer_tenure_months": rng.randint(1, 120),
                "sentiment_score": round(rng.uniform(-0.98, -0.05), 3),
                "expected_sla_hours": SLA_BY_PRIORITY[template.priority],
                "escalation_level": ESCALATION_BY_PRIORITY[template.priority],
                "case_origin": rng.choice(("B2C", "B2B", "Partner")),
                "resolution_complexity": rng.choice(("Low", "Medium", "High")),
                "task_signal": rng.choice(template.task_signals),
            }
        )
    return rows


def build_handwritten_rows(total_rows: int = 5_000, seed: int = 101) -> list[dict[str, object]]:
    rng = random.Random(seed)
    lead_ins = [
        "Hi team,",
        "I need help because",
        "Please look into this.",
        "This has become really frustrating because",
        "I am writing to report that",
        "Can someone check why",
    ]
    closers = [
        "Please resolve this soon.",
        "Kindly treat this as urgent.",
        "I need an exact update from the team.",
        "A proper callback would help.",
        "I expect a clear resolution plan.",
    ]
    rows: list[dict[str, object]] = []
    for complaint_id in range(1, total_rows + 1):
        template = ISSUE_TEMPLATES[(complaint_id - 1) % len(ISSUE_TEMPLATES)]
        base = rng.choice(template.complaint_patterns).format(
            amount=f"${rng.randint(20, 999)}",
            days=rng.randint(2, 25),
            product_line=rng.choice(template.product_lines),
            ticket=f"HND-{complaint_id:05d}",
            region=rng.choice(REGIONS),
            date_text=(datetime(2025, 1, 1) - timedelta(days=rng.randint(1, 180))).strftime("%d %b %Y"),
        )
        complaint_text = f"{rng.choice(lead_ins)} {base} {rng.choice(closers)}"
        rows.append(
            {
                "complaint_id": f"V{complaint_id:05d}",
                "complaint_text": complaint_text,
                "true_department": template.department,
                "true_priority": template.priority,
                "true_core_issue": template.issue,
            }
        )
    return rows


def build_cfpb_rows(total_rows: int = 2_500, seed: int = 202) -> list[dict[str, object]]:
    rng = random.Random(seed)
    product_map = {
        "Billing": ["Credit card", "Checking or savings account", "Money transfers"],
        "Logistics": ["Money transfer, virtual currency, or money service", "Consumer Loan"],
        "Technical Support": ["Credit reporting", "Mobile wallet", "Virtual currency"],
        "Customer Service": ["Debt collection", "Personal loan", "Prepaid card"],
    }
    rows: list[dict[str, object]] = []
    for complaint_id in range(1, total_rows + 1):
        template = ISSUE_TEMPLATES[(complaint_id - 1) % len(ISSUE_TEMPLATES)]
        product = rng.choice(product_map.get(template.department, ["General-purpose credit card"]))
        complaint_text = rng.choice(template.complaint_patterns).format(
            amount=f"${rng.randint(15, 1299)}",
            days=rng.randint(1, 35),
            product_line=product,
            ticket=f"CFPB-{complaint_id:06d}",
            region=rng.choice(REGIONS),
            date_text=(datetime(2025, 6, 1) - timedelta(days=rng.randint(1, 270))).strftime("%d %b %Y"),
        )
        complaint_text = f"Consumer narrative: {complaint_text} I contacted the company through {rng.choice(template.channels).lower()} and I want this fixed."
        rows.append(
            {
                "complaint_id": f"CFPB-{complaint_id:06d}",
                "product": product,
                "sub_product": rng.choice(("General", "Online service", "Mobile service", "Branch service")),
                "issue": template.issue,
                "sub_issue": rng.choice(template.task_signals),
                "consumer_complaint_narrative": complaint_text,
                "complaint_text": complaint_text,
                "company": rng.choice(("Acme Financial", "NorthStar Bank", "BlueLedger Finance", "Mercury Services")),
                "state": rng.choice(("CA", "NY", "TX", "FL", "IL", "WA")),
                "submitted_via": rng.choice(("Web", "Phone", "Email", "Referral")),
                "timely_response": rng.choice(("Yes", "No")),
                "true_department": template.department,
                "true_priority": template.priority,
                "true_core_issue": template.issue,
            }
        )
    return rows


def save_rows(rows: list[dict[str, object]], output_path: Path) -> Path:
    _ensure_parent(output_path)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def build_all_datasets() -> dict[str, Path]:
    raw_path = save_rows(build_raw_rows(), Path(RAW_DATA_PATH))
    validation_path = save_rows(build_handwritten_rows(), Path(VALIDATION_DATA_PATH))
    cfpb_path = save_rows(build_cfpb_rows(), Path(CFPB_DATA_PATH))
    return {"raw": raw_path, "validation": validation_path, "cfpb": cfpb_path}


if __name__ == "__main__":
    outputs = build_all_datasets()
    for name, path in outputs.items():
        print(f"{name}: {path}")
