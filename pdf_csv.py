import pdfplumber
import pandas as pd
import re


def convert_pdf_to_csv(pdf_path, output_csv):

    transactions = []

    with pdfplumber.open(pdf_path) as pdf:

        for page in pdf.pages:

            text = page.extract_text()

            if not text:
                continue

            lines = text.split("\n")

            i = 0
            while i < len(lines):

                line = lines[i].strip()

                # Pattern: transaction number + date
                match = re.match(r'^(\d+)\s+(\d{2}\.\d{2}\.\d{4})', line)

                if match:

                    date = match.group(2)

                    amounts = re.findall(r'\d+\.\d{2}', line)

                    withdrawal = None
                    deposit = None
                    balance = None

                    # Usually last value is balance
                    if len(amounts) >= 2:
                        withdrawal = amounts[-2]
                        balance = amounts[-1]

                    # Check next line for remarks
                    remarks = ""

                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()

                        if not re.match(r'^\d+\s+\d{2}\.\d{2}\.\d{4}', next_line):
                            remarks = next_line
                            i += 1

                    transactions.append({
                        "value_date": date,
                        "transaction_date": date,
                        "remarks": remarks,
                        "withdrawal": withdrawal,
                        "deposit": 0,
                        "balance": balance
                    })

                i += 1

    df = pd.DataFrame(transactions)

    df.to_csv(output_csv, index=False)

    print("CSV created:", output_csv)

    return output_csv