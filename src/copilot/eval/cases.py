"""Hand-written evaluation cases.

ROUTING is a regression set for the supervisor: each message has the intent a
buyer would expect. QUESTIONS are paraphrased contract questions per topic,
worded the way a buyer talks and not the way the contract is written.
"""

ROUTING = [
    ("Top 5 vendors by spend in 2025", "top_vendors"),
    ("Who do we spend the most money with?", "top_vendors"),
    ("Show me our biggest suppliers for copper wire", "top_vendors"),
    ("How much did we buy by category last year?", "spend_by_category"),
    ("Break down 2024 spend per category", "spend_by_category"),
    ("Which vendors charge the most above should-cost for steel sheet?", "expensive_vendors"),
    ("Who is overcharging us?", "expensive_vendors"),
    ("List the most expensive suppliers in 2025", "expensive_vendors"),
    ("Who delivers late the most?", "late_vendors"),
    ("Which suppliers have the worst on-time record in 2025?", "late_vendors"),
    ("How is V-0009 performing?", "vendor_profile"),
    ("Tell me about Siam Supply", "vendor_profile"),
    ("Is V-0046 reliable?", "vendor_profile"),
    ("What is the score of V-0021?", "vendor_profile"),
    ("What are the payment terms with V-0003?", "contract_question"),
    ("What happens if Siam Supply delivers late?", "contract_question"),
    ("How many days notice do we need to terminate the agreement with V-0010?",
     "contract_question"),
    ("Is the price with V-0007 linked to an index?", "contract_question"),
    ("How long is the warranty from V-0015?", "contract_question"),
    ("Can we cancel the contract with V-0033?", "contract_question"),
    ("Are we close to a rebate with V-0002 in 2025?", "rebate_status"),
    ("Has V-0021 earned us a volume rebate this year?", "rebate_status"),
    ("Is 1.62 USD fair for 5000 kg of MAT-0001 from V-0002?", "check_quote"),
    ("V-0010 quoted 95 THB for 2000 pcs of MAT-0003. Is that too high?", "check_quote"),
    ("MAT-0012 at 4.10 for 800 pcs, good price?", "check_quote"),
    ("Check this quote: MAT-0005, 10000 kg, 2.35 USD", "check_quote"),
    ("What should MAT-0020 cost for 3000 pcs if they ask 7.5?", "check_quote"),
    ("Compare the quotes in RFQ-01990", "rfq_compare"),
    ("Which offer is best in RFQ-01850?", "rfq_compare"),
    ("Build a negotiation playbook for that quote", "negotiate"),
    ("How should I negotiate 1.62 USD for 5000 kg of MAT-0001 from V-0002?", "negotiate"),
    ("What should we offer back to V-0002?", "negotiate"),
    ("Give me talking points to push back on this price", "negotiate"),
    ("How do I negotiate RFQ-01990?", "negotiate"),
    ("Let me practise against the vendor", "simulate_start"),
    ("Can you role-play the supplier so I can rehearse?", "simulate_start"),
    ("Simulate the negotiation with me", "simulate_start"),
    ("Run the negotiation automatically", "simulate_auto"),
    ("Compare strategies for me", "simulate_auto"),
    ("hello", "help"),
    ("What can you do?", "help"),
    ("thanks", "help"),
]

QUESTIONS = {
    "Payment Terms": [
        "How long do we have to pay an invoice?",
        "When is the bill due?",
        "Do we get anything for paying faster?",
    ],
    "Volume Rebate": [
        "Do we get money back if we buy a lot in a year?",
        "Is there a refund for high annual spend?",
        "What bonus applies once purchases pass a threshold?",
    ],
    "Delivery and Late Delivery": [
        "What happens if they deliver late?",
        "What compensation do we get for a delayed shipment?",
        "Is there a penalty when goods are overdue?",
    ],
    "Term and Termination": [
        "How much notice do we need to give to cancel?",
        "Can we exit this contract early?",
        "How long does the contract run?",
    ],
    "Price and Price Adjustment": [
        "Can the price change when the commodity market moves?",
        "How often is pricing renegotiated?",
        "Are prices tied to an index?",
    ],
    "Warranty": [
        "How long are the goods guaranteed?",
        "What cover do we have against faulty material?",
        "Who is responsible if a defect appears after delivery?",
    ],
}
