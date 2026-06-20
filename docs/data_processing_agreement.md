# Data Processing Agreement (DPA)

**Between:**

**[COMPANY NAME]** ("Processor")
[Company Address]
[Registration Number / VAT Number]

**And:**

**[CUSTOMER NAME]** ("Controller")
[Customer Address]
[Registration Number / VAT Number]

**Effective Date:** [INSERT DATE]

---

## Recitals

This Data Processing Agreement ("Agreement") governs the processing of personal data by Processor on behalf of Controller in connection with the InsightLens legal research platform ("Service"), in accordance with:
- Regulation (EU) 2016/679 (General Data Protection Regulation, "GDPR")
- UK GDPR (as retained by the Data Protection Act 2018)
- California Consumer Privacy Act (CCPA) / CPRA, as applicable

---

## 1. Definitions

- **"Controller"** means the entity that determines the purposes and means of processing personal data.
- **"Processor"** means the entity that processes personal data on behalf of the Controller.
- **"Personal Data"** has the meaning given in Article 4(1) GDPR.
- **"Processing"** has the meaning given in Article 4(2) GDPR.
- **"Data Subject"** means any identified or identifiable natural person whose personal data is processed under this Agreement.
- **"Sub-processor"** means any third party engaged by the Processor to carry out processing activities on behalf of the Controller.
- **"Security Incident"** means any accidental or unlawful destruction, loss, alteration, unauthorised disclosure of, or access to, Personal Data.

---

## 2. Scope and Nature of Processing

### 2.1 Subject Matter
Processor provides an AI-powered legal document analysis platform. Controller's employees or clients upload legal documents, which Processor processes to extract text, generate semantic embeddings, store structured chunks, and respond to natural-language queries.

### 2.2 Duration
Processing continues for the term of the Controller's subscription to the Service, and ceases upon termination in accordance with Section 9.

### 2.3 Categories of Data Subjects
- The Controller's employees, associates, and contractors who use the Service
- Individuals whose personal data appears within documents uploaded by the Controller (e.g., parties to litigation, clients, witnesses)

### 2.4 Categories of Personal Data
- Account data: email addresses, display names, login history
- Document content: any personal data within uploaded PDFs (names, dates, financial data, medical information, etc. — scope determined by the Controller)
- Usage data: query text, AI responses, conversation history

### 2.5 Purpose of Processing
To provide the InsightLens Service as described in the Terms of Service, including document ingestion, semantic search, AI-assisted question answering, and case management.

---

## 3. Controller Obligations

The Controller shall:

1. Ensure a lawful basis exists for any Personal Data provided to the Processor.
2. Ensure Data Subjects are informed that their data may be processed by the Processor.
3. Comply with all applicable data protection laws with respect to data supplied to the Processor.
4. Not instruct the Processor to perform any processing that would violate applicable law.

---

## 4. Processor Obligations

The Processor shall:

1. **Process only on instructions.** Process Personal Data only on documented instructions from the Controller (including those set out in this Agreement), unless required by applicable law.
2. **Confidentiality.** Ensure that persons authorised to process Personal Data are bound by confidentiality obligations.
3. **Security.** Implement the technical and organisational measures described in Section 6.
4. **Sub-processing.** Not engage sub-processors other than those listed in Annex A without prior written notice to the Controller (30 days' notice for new sub-processors).
5. **Data subject rights.** Assist the Controller in responding to Data Subject requests within 5 business days of receipt.
6. **Security assistance.** Assist the Controller in meeting obligations under Articles 32–36 GDPR (security, breach notification, DPIAs).
7. **Deletion or return.** At the Controller's choice, delete or return all Personal Data upon termination per Section 9.
8. **Audit.** Make available all information necessary to demonstrate compliance and allow for audits or inspections by the Controller (or a mandated auditor) on 30 days' written notice, at the Controller's cost.
9. **Legal disclosure.** Promptly inform the Controller if Processor is required by law to disclose Personal Data, unless prohibited from doing so.

---

## 5. International Transfers

### 5.1 Transfers Outside the EEA/UK
Personal Data may be transferred to the United States for processing by Processor and its sub-processors. Such transfers are made on the basis of:
- Standard Contractual Clauses (SCCs) adopted by the European Commission (Commission Implementing Decision (EU) 2021/914), incorporated by reference into this Agreement.
- For UK transfers: the UK International Data Transfer Addendum (IDTA) to the EU SCCs, where applicable.

### 5.2 Adequacy Decisions
Where an adequacy decision exists for the destination country, the Processor may rely on it in lieu of SCCs.

---

## 6. Technical and Organisational Security Measures

The Processor implements the following measures (per Article 32 GDPR):

### Access Control
- Authentication via Firebase (OAuth 2.0 / OIDC). All API endpoints require valid ID tokens.
- Admin endpoints require a separately managed API key transmitted only via HTTPS.
- Staff access to production systems is limited to authorised personnel and logged.

### Data in Transit
- All communications between clients and the Service use TLS 1.2 or higher.
- Database connections use SSL/TLS.

### Data at Rest
- PostgreSQL database hosted on Neon with encryption at rest (AES-256).
- Uploaded documents are stored only transiently during ingestion and are deleted from disk after processing.

### Availability and Resilience
- Database is managed by Neon with automated backups and point-in-time recovery.
- Application is deployed in containers with health checks and automated restart policies.

### Incident Response
- Security Incidents are logged and triaged within 4 hours of detection.
- The Controller is notified within 72 hours of the Processor becoming aware of a Security Incident affecting the Controller's Personal Data.

### Vulnerability Management
- Dependencies are reviewed for known CVEs as part of the deployment pipeline.
- Critical security patches are applied within 72 hours of availability.

---

## 7. Sub-Processors

The Controller grants general authorisation to engage the sub-processors listed in **Annex A**. Processor will provide 30 days' notice before adding or replacing a sub-processor. The Controller may object in writing within that period; if the parties cannot resolve the objection, either party may terminate this Agreement.

---

## 8. Security Incident Notification

Upon becoming aware of a Security Incident affecting Personal Data processed under this Agreement, Processor will:

1. Notify the Controller **within 72 hours** at the contact provided in the signature block.
2. Provide, to the extent then known: nature of the incident, categories and approximate number of Data Subjects and records affected, likely consequences, and measures taken or proposed.
3. Cooperate fully with the Controller's investigation and remediation efforts.

---

## 9. Deletion on Termination

Upon expiry or termination of the Controller's subscription, Processor will:

1. Within **30 days**, delete all Personal Data processed on behalf of the Controller from live systems.
2. Retain encrypted backups for up to **90 days** after which they are permanently purged.
3. Provide written confirmation of deletion upon request.

Billing records required by applicable accounting law (typically 7 years) are retained in anonymised or pseudonymised form where possible.

---

## 10. Liability

The liability of each party under this Agreement is subject to and does not exceed the limits of liability set out in the Master Services Agreement or Terms of Service between the parties. Where no such agreement exists, liability is limited to the greater of (i) fees paid by the Controller in the 12 months preceding the incident or (ii) €10,000 / £10,000 / $10,000.

---

## 11. Term and Termination

This Agreement is co-terminus with the Controller's subscription to the Service. Either party may terminate this Agreement on 30 days' written notice if the other party materially breaches its data protection obligations and fails to cure such breach within 14 days of notice.

---

## 12. Governing Law

This Agreement is governed by the law of [GOVERNING LAW JURISDICTION], and the parties submit to the exclusive jurisdiction of the courts of [JURISDICTION]. For EEA/UK customers, this Agreement is interpreted to comply with GDPR / UK GDPR as a floor minimum.

---

## 13. Signatures

**Processor:**
[COMPANY NAME]

Signed: ___________________________
Name: ___________________________
Title: ___________________________
Date: ___________________________

**Controller:**
[CUSTOMER NAME]

Signed: ___________________________
Name: ___________________________
Title: ___________________________
Date: ___________________________

---

## Annex A — Approved Sub-Processors

| Sub-processor | Role | Processing Location | DPA / Safeguard |
|---|---|---|---|
| **Anthropic, Inc.** | AI text generation (Claude API) | United States | Anthropic API Terms + Data Processing Addendum |
| **Neon, Inc.** | PostgreSQL hosting (document chunks, embeddings, metadata) | AWS us-east-1 | Neon DPA + AWS SCCs |
| **Google LLC (Firebase)** | User authentication and identity management | United States | Google Cloud DPA + SCCs |
| **Voyage AI** | Text embedding generation | United States | Voyage AI Terms |
| **Zep Cloud** | Conversation memory storage | United States | Zep Privacy Policy + DPA |
| **Stripe, Inc.** | Payment processing | United States | Stripe DPA + SCCs |

---

## Annex B — Standard Contractual Clauses

The Standard Contractual Clauses for the transfer of Personal Data to third countries pursuant to Commission Implementing Decision (EU) 2021/914 (Module Two: Controller to Processor) are incorporated herein by reference. In the event of any conflict between the body of this Agreement and the SCCs, the SCCs shall prevail.

[Attach completed SCCs as Exhibit B-1 prior to execution.]
