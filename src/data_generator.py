"""Synthetic dataset generator for large-scale complaint analysis experiments."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from config import RAW_DATA_PATH


@dataclass(frozen=True)
class IssueTemplate:
    issue: str
    department: str
    priority: str
    product_lines: tuple[str, ...]
    channels: tuple[str, ...]
    complaint_patterns: tuple[str, ...]
    task_signals: tuple[str, ...]


ISSUE_TEMPLATES: tuple[IssueTemplate, ...] = (
    IssueTemplate(
        issue="Wrong charge",
        department="Billing",
        priority="High",
        product_lines=("Subscription", "Payments", "Enterprise Billing"),
        channels=("Email", "Phone", "Web", "Chat"),
        complaint_patterns=(
            "I was charged {amount} twice for my {product_line} plan and nobody has fixed it yet.",
            "Your system applied an unauthorized {amount} charge on my account after I already paid.",
            "Billing collected the wrong amount for invoice {ticket} and support keeps delaying the correction.",
        ),
        task_signals=("refund", "invoice correction", "charge reversal"),
    ),
    IssueTemplate(
        issue="Refund delay",
        department="Billing",
        priority="Medium",
        product_lines=("Subscription", "Marketplace", "Payments"),
        channels=("Email", "Web", "Chat"),
        complaint_patterns=(
            "My refund was approved {days} days ago for {product_line} and the money is still missing.",
            "I canceled my order but the reimbursement for {amount} has not reached my bank yet.",
            "Finance confirmed a refund for ticket {ticket}, but I am still waiting after {days} days.",
        ),
        task_signals=("reimbursement status", "refund queue", "payment gateway check"),
    ),
    IssueTemplate(
        issue="Invoice error",
        department="Billing",
        priority="Medium",
        product_lines=("Enterprise Billing", "Subscription", "Payments"),
        channels=("Email", "Phone", "Web"),
        complaint_patterns=(
            "The invoice for my {product_line} account has the wrong tax calculation for region {region}.",
            "Invoice {ticket} lists duplicate line items and the total does not match my contract.",
            "Your billing statement shows a failed payment even though my bank confirms the transfer succeeded.",
        ),
        task_signals=("tax validation", "statement review", "invoice correction"),
    ),
    IssueTemplate(
        issue="Delivery delay",
        department="Logistics",
        priority="High",
        product_lines=("Marketplace", "Devices", "Accessories"),
        channels=("Phone", "Web", "Chat"),
        complaint_patterns=(
            "My {product_line} shipment was supposed to arrive on {date_text} but it is still stuck in transit.",
            "The courier has not updated tracking for {days} days and my order has still not arrived.",
            "This delivery delay is affecting my work schedule and no one can tell me where package {ticket} is.",
        ),
        task_signals=("shipment trace", "courier escalation", "delivery recovery"),
    ),
    IssueTemplate(
        issue="Damaged shipment",
        department="Logistics",
        priority="High",
        product_lines=("Devices", "Accessories", "Marketplace"),
        channels=("Phone", "Chat", "Web"),
        complaint_patterns=(
            "The {product_line} item arrived with a cracked screen and a damaged box.",
            "Package {ticket} was crushed on arrival and the product inside is broken.",
            "I received the order today, but the unit is physically damaged and unusable.",
        ),
        task_signals=("replacement shipment", "damage claim", "reverse pickup"),
    ),
    IssueTemplate(
        issue="Damaged product",
        department="Logistics",
        priority="High",
        product_lines=("Glassware", "Electronics", "Home Appliance", "Furniture", "Marketplace"),
        channels=("Web", "Chat", "Phone", "Email"),
        complaint_patterns=(
            "The delivered {product_line} has shattered glass and multiple visible cracks, so the product is unusable.",
            "I opened package {ticket} and found the product body dented, broken, and damaged right out of the box.",
            "The item I received today has a smashed screen and physical damage even though I expected a new working product.",
            "My {product_line} arrived with broken parts, cracked glass, and signs of impact damage on the product itself.",
            "The product is damaged on arrival with chipped edges, fractured glass, and unsafe condition for use.",
        ),
        task_signals=("damaged product replacement", "product damage inspection", "courier and packaging review"),
    ),
    IssueTemplate(
        issue="Wrong item delivered",
        department="Logistics",
        priority="Medium",
        product_lines=("Marketplace", "Accessories", "Devices"),
        channels=("Web", "Chat", "Email"),
        complaint_patterns=(
            "I ordered one {product_line} item but received the wrong color and wrong model instead.",
            "The order linked to ticket {ticket} contains a completely different product from what I purchased.",
            "Your warehouse sent the wrong item and now I need an exchange urgently.",
        ),
        task_signals=("exchange", "reverse pickup", "warehouse audit"),
    ),
    IssueTemplate(
        issue="App or website failure",
        department="Technical Support",
        priority="High",
        product_lines=("Mobile App", "Web Portal", "Checkout"),
        channels=("Web", "Chat", "Phone"),
        complaint_patterns=(
            "The {product_line} crashes every time I try to submit my order and shows a 500 error.",
            "Your website keeps failing during checkout, so I cannot complete the payment flow.",
            "The platform is not loading for my team since {date_text}, and this is blocking revenue.",
        ),
        task_signals=("incident response", "bug reproduction", "service recovery"),
    ),
    IssueTemplate(
        issue="Login or account access issue",
        department="Technical Support",
        priority="High",
        product_lines=("Web Portal", "Mobile App", "Identity"),
        channels=("Phone", "Chat", "Email"),
        complaint_patterns=(
            "I cannot log in to my {product_line} account because the OTP never arrives.",
            "My account was locked after a password reset attempt and support has not restored access.",
            "The sign-in page keeps rejecting valid credentials for ticket {ticket}.",
        ),
        task_signals=("identity verification", "access recovery", "authentication bug"),
    ),
    IssueTemplate(
        issue="Device or integration issue",
        department="Technical Support",
        priority="Medium",
        product_lines=("API Platform", "Devices", "Partner Integration"),
        channels=("Email", "Web", "Phone"),
        complaint_patterns=(
            "Our {product_line} sync failed repeatedly after yesterday's release and the integration is down.",
            "The device disconnected from your service and now every API call returns an error.",
            "Integration ticket {ticket} is blocked because the webhook events stopped arriving after {days} days of instability.",
        ),
        task_signals=("compatibility fix", "integration diagnostics", "API investigation"),
    ),
    IssueTemplate(
        issue="Poor support experience",
        department="Customer Service",
        priority="High",
        product_lines=("Support Desk", "Premium Care", "Marketplace"),
        channels=("Phone", "Chat", "Email"),
        complaint_patterns=(
            "Your agent was rude, disconnected the call, and never followed up on my complaint.",
            "I have waited {days} days for a response from customer service and still feel ignored.",
            "Support case {ticket} has been bounced around with no clear owner and no resolution.",
        ),
        task_signals=("quality audit", "callback", "service recovery"),
    ),
    IssueTemplate(
        issue="Cancellation or return issue",
        department="Customer Service",
        priority="Medium",
        product_lines=("Marketplace", "Subscription", "Devices"),
        channels=("Email", "Chat", "Web"),
        complaint_patterns=(
            "I submitted a cancellation request for my {product_line} order, but the system still shows it as active.",
            "The return request linked to ticket {ticket} has stalled and no pickup has been scheduled.",
            "Your return policy workflow is unclear, and I still cannot send the product back after {days} days.",
        ),
        task_signals=("return workflow", "policy exception", "pickup scheduling"),
    ),
    IssueTemplate(
        issue="Security or fraud concern",
        department="Technical Support",
        priority="Critical",
        product_lines=("Identity", "Payments", "Web Portal"),
        channels=("Phone", "Email", "Web"),
        complaint_patterns=(
            "I found a suspicious login on my {product_line} account and I think someone hijacked it.",
            "There is a possible fraud incident on transaction {ticket} and I need the account locked immediately.",
            "My payment profile shows activity I did not authorize, and I am worried about a breach.",
        ),
        task_signals=("security incident", "fraud lock", "breach triage"),
    ),
)

REGIONS = ("North", "South", "East", "West", "Central")
COUNTRIES = ("India", "USA", "UK", "UAE", "Singapore")
CUSTOMER_SEGMENTS = ("Student", "SMB", "Enterprise", "Premium", "Standard")
SENTIMENTS = ("Frustrated", "Disappointed", "Concerned", "Angry", "Urgent")
SLA_BY_PRIORITY = {"Medium": 48, "High": 24, "Critical": 4}
ESCALATION_BY_PRIORITY = {"Medium": 1, "High": 2, "Critical": 3}


def random_date_text(reference: datetime, rng: random.Random) -> str:
    dt = reference - timedelta(days=rng.randint(1, 120))
    return dt.strftime("%d %b %Y")


def generate_rows(total_rows: int = 5000, seed: int = 42) -> list[dict[str, object]]:
    rng = random.Random(seed)
    start_date = datetime(2024, 1, 1)
    rows: list[dict[str, object]] = []

    for complaint_id in range(1, total_rows + 1):
        template = ISSUE_TEMPLATES[(complaint_id - 1) % len(ISSUE_TEMPLATES)]
        created_at = start_date + timedelta(hours=complaint_id * 3 + rng.randint(0, 18))
        amount = f"${rng.randint(19, 499)}"
        days = rng.randint(2, 21)
        ticket = f"CMP-{complaint_id:05d}"
        region = rng.choice(REGIONS)
        country = rng.choice(COUNTRIES)
        channel = rng.choice(template.channels)
        product_line = rng.choice(template.product_lines)
        segment = rng.choice(CUSTOMER_SEGMENTS)
        sentiment_label = rng.choice(SENTIMENTS)
        complaint_text = rng.choice(template.complaint_patterns).format(
            amount=amount,
            days=days,
            product_line=product_line,
            ticket=ticket,
            region=region,
            date_text=random_date_text(created_at, rng),
        )
        complaint_text = f"{complaint_text} I am a {segment.lower()} customer and this feels {sentiment_label.lower()}."
        if template.priority in {"High", "Critical"} and rng.random() < 0.45:
            complaint_text += " Please escalate this immediately."

        rows.append(
            {
                "complaint_id": complaint_id,
                "complaint_text": complaint_text,
                "true_department": template.department,
                "true_priority": template.priority,
                "true_core_issue": template.issue,
                "channel": channel,
                "product_line": product_line,
                "customer_segment": segment,
                "region": region,
                "country": country,
                "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "customer_tenure_months": rng.randint(1, 84),
                "sentiment_score": round(rng.uniform(-0.95, -0.15), 3),
                "expected_sla_hours": SLA_BY_PRIORITY[template.priority],
                "escalation_level": ESCALATION_BY_PRIORITY[template.priority],
                "case_origin": rng.choice(("B2C", "B2B", "Partner")),
                "resolution_complexity": rng.choice(("Low", "Medium", "High")),
                "task_signal": rng.choice(template.task_signals),
            }
        )

    return rows


def save_dataset(rows: list[dict[str, object]], output_path: Path = Path(RAW_DATA_PATH)) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


if __name__ == "__main__":
    dataset_rows = generate_rows(total_rows=5000, seed=42)
    saved_to = save_dataset(dataset_rows)
    print(f"Generated {len(dataset_rows)} complaints at {saved_to}")
