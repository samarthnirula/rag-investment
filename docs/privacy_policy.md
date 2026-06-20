# Privacy Policy

**InsightLens — Legal Research AI Platform**
**Effective Date:** [INSERT DATE]
**Last Updated:** [INSERT DATE]

---

## 1. Introduction

InsightLens ("we," "us," or "our") provides an AI-powered legal research platform designed for legal professionals. This Privacy Policy describes how we collect, use, store, and protect information when you use our services at [YOUR DOMAIN] (the "Service").

By accessing the Service, you acknowledge that you have read and understood this Privacy Policy.

---

## 2. Information We Collect

### 2.1 Account Information
- Email address and display name (via Firebase Authentication)
- Account creation date and last login timestamp
- Subscription plan and billing status

### 2.2 Content You Upload
- PDF documents and images you submit for analysis
- Document metadata (filename, page count, upload date)
- Extracted text, headings, footnotes, and image descriptions derived from your documents

### 2.3 Usage Data
- Queries you submit to the AI assistant
- AI-generated responses and confidence scores
- Retrieval results and source citations returned to you
- Session identifiers and conversation history (stored via Zep memory service)

### 2.4 Technical Data
- IP address (used solely for anonymous demo rate limiting)
- HTTP request logs retained for a maximum of 30 days
- Error traces for debugging (stripped of document content before logging)

### 2.5 Billing Data
- Payment processing is handled by Stripe. We store only Stripe customer IDs and subscription identifiers. We do **not** store payment card numbers or bank details.

---

## 3. How We Use Your Information

| Purpose | Legal Basis (GDPR) |
|---|---|
| Providing AI-powered document analysis | Performance of contract |
| Storing and retrieving your documents and queries | Performance of contract |
| Enforcing usage limits and preventing abuse | Legitimate interest |
| Sending transactional emails (billing, security) | Performance of contract |
| Improving service reliability and debugging errors | Legitimate interest |
| Complying with legal obligations | Legal obligation |

We do **not** use your legal documents to train AI models. We do **not** sell, rent, or share your data with third-party advertisers.

---

## 4. Attorney-Client Privilege Notice

InsightLens is designed for use by legal professionals. You are responsible for ensuring that any use of the Service is consistent with your professional obligations, including rules governing attorney-client privilege and confidentiality.

We treat all uploaded documents as potentially privileged. Our staff does not access document content except in the narrow circumstance of investigating a reported security incident, and only with appropriate controls in place.

---

## 5. Data Storage and Sub-Processors

Your data is processed by the following sub-processors:

| Sub-processor | Role | Data Location |
|---|---|---|
| **Neon** (neon.tech) | PostgreSQL database — stores document chunks, metadata, embeddings | AWS us-east-1 |
| **Anthropic** | AI text generation (Claude API) | United States |
| **Voyage AI** | Document embedding generation | United States |
| **Firebase / Google** | Authentication and identity | United States |
| **Zep** | Conversation memory storage | United States |
| **Stripe** | Payment processing | United States |
| **Redis** (self-hosted) | Job queue and rate limiting | Same host as application server |

All sub-processors are contractually bound to process data only as instructed and to implement appropriate security measures.

---

## 6. Data Retention

| Data type | Retention period |
|---|---|
| Uploaded documents and extracted chunks | Until you delete them or close your account |
| Conversation history | Until you clear chat history or close your account |
| Billing records | 7 years (statutory accounting requirement) |
| Audit logs | 90 days rolling |
| HTTP access logs | 30 days rolling |
| Deleted account data | Purged within 30 days of account closure |

When you delete a document or close your account, associated chunks, embeddings, and images are cascade-deleted from the database.

---

## 7. Your Rights (GDPR / CCPA)

If you are located in the European Economic Area, United Kingdom, or California, you have the following rights:

- **Access:** Request a copy of the personal data we hold about you.
- **Correction:** Request correction of inaccurate personal data.
- **Deletion:** Request deletion of your account and associated data ("right to be forgotten").
- **Portability:** Request your data in a machine-readable format.
- **Restriction:** Request that we restrict processing of your data.
- **Objection:** Object to processing based on legitimate interest.
- **Withdraw consent:** Where processing is based on consent, withdraw it at any time.

To exercise any of these rights, contact us at [PRIVACY_CONTACT_EMAIL]. We will respond within 30 days.

You may also delete your account directly from the Profile page in the application, which triggers immediate cascade deletion of your documents, chunks, and chat history.

---

## 8. Security

We implement the following security measures:

- All data transmitted between your browser and our servers is encrypted via TLS 1.2+.
- Database connections use SSL/TLS.
- API endpoints require authentication via Firebase ID tokens.
- The admin API is protected by a separately managed API key.
- Document files are stored only temporarily during processing and are not retained on disk after ingestion.

Despite these measures, no method of transmission or storage is 100% secure. In the event of a personal data breach, we will notify affected users and relevant supervisory authorities within 72 hours as required by GDPR Article 33.

---

## 9. Cookies and Tracking

The Service uses only functional cookies required for authentication (Firebase Auth session tokens). We do not use analytics cookies, advertising pixels, or cross-site tracking technologies.

---

## 10. Children's Privacy

The Service is not directed to individuals under the age of 16. We do not knowingly collect personal data from minors. If you become aware that a minor has provided us with personal data, contact us immediately.

---

## 11. International Data Transfers

Your data is processed in the United States. If you are accessing the Service from outside the United States, your data may be transferred to, stored, and processed in the US. We rely on Standard Contractual Clauses (SCCs) approved by the European Commission for transfers from the EEA.

---

## 12. Changes to This Policy

We may update this Privacy Policy from time to time. We will notify you of material changes by email or by displaying a notice within the Service at least 14 days before the change takes effect. Continued use of the Service after the effective date constitutes acceptance of the revised policy.

---

## 13. Contact Us

If you have questions about this Privacy Policy or wish to exercise your data rights, contact:

**[COMPANY NAME]**
**Privacy Contact:** [PRIVACY_CONTACT_EMAIL]
**Address:** [COMPANY ADDRESS]
**Data Protection Officer (if applicable):** [DPO_EMAIL]

For EU/UK inquiries, you also have the right to lodge a complaint with your local supervisory authority.
