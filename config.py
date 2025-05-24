TARGET_ENTITIES = [
    "full_name", "email", "phone_number", "dob",
    "aadhar_num", "credit_debit_no", "cvv_no", "expiry_no"
]

REGEX_PATTERNS = {
    "email": r"[a-zA-Z0-9._%+-]+@[a-zAET0-9.-]+\\.[a-zA-Z]{2,}",
    "phone_number": r"(\\+91[\\-\\s]?)?[789]\\d{9}",
    "dob": r"\\b\\d{1,2}[/-]\\d{1,2}[/-]\\d{2,4}\\b",
    "aadhar_num": r"\\b\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}\\b",
    "credit_debit_no": r"\\b(?:\\d[ -]*?){13,16}\\b",
    "cvv_no": r"\\b\\d{3}\\b",
    "expiry_no": r"\\b(0[1-9]|1[0-2])\\/?([0-9]{2}|[0-9]{4})\\b"
}
