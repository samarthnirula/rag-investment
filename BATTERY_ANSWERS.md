# InsightLens — 24 Question Battery Reference

Expected system behaviour for each question based on current architecture.

**Status key:** ✅ Pass · ⚠️ Partial · ❌ Fail (acknowledged gap)

---

## Group A — Version & Temporal Awareness

---

### Q1 · What is Digital Realty's total IT capacity?

**Status:** ⚠️ Partial Pass

**What the system does:**
Retrieves chunks from both the December 2025 and March 2026 DLR decks. The March 2026 doc is marked CURRENT VERSION (supersedes December 2025) and surfaces first due to the 15% score boost. The December 2025 doc is marked HISTORICAL VERSION and gets a 20% penalty but still appears because both documents are in the corpus.

**Expected answer shape:**
> As of the March 2026 deck [CURRENT VERSION, Source 1], Digital Realty reports 3 GW of in-place capacity with a 5 GW future pipeline, implying approximately 8 GW total under a broad definition. The December 2025 deck [HISTORICAL VERSION, Source 2] reported 2.9 GW in-place plus 730 MW under construction. The definitions of "total capacity" differ between the two decks — the March figure includes a development pipeline while the December figure reflects in-service and near-term capacity only.

**Why partial:** The two decks define "total capacity" differently and the system presents both with attribution, but a human analyst would need to reconcile the definitions. The system correctly flags this rather than silently picking one.

---

### Q2 · What is BXP's projected 2026 occupancy?

**Status:** ✅ Pass

**What the system does:**
Both BXP documents are ingested. The Q4 2025 Presentation is labelled "(type: Q4 Update)" and the Morning Session Deck is "(type: Investor Day)". Prompt rule 12 instructs Claude that Q4 Update figures supersede Investor Day figures on the same metric. The Q4 figure is presented as the current guidance with the Investor Day range noted as earlier guidance.

**Expected answer shape:**
> BXP's Q4 2025 Presentation [Source 1, type: Q4 Update] sets 2026 occupancy guidance at 88% (point estimate). This tightened from the range of 87.25% to 88.0% provided at the June 2025 Investor Day [Source 2, type: Investor Day], which was based on assumptions made several months earlier. The Q4 figure is the more current guidance.

---

### Q3 · What changed in BXP's strategy between Investor Day and Q4 2025?

**Status:** ⚠️ Partial Pass

**What the system does:**
Coverage guarantee ensures both BXP documents appear in results. The system compares the two and surfaces concrete changes — occupancy guidance tightened, dividend reset completed, FFO guidance became public.

**Expected answer shape:**
> The overall strategic framework is unchanged between the two presentations. What advanced: (1) the dividend reset announced at Investor Day [Source 2] was completed by Q4 [Source 1]; (2) occupancy guidance tightened from a 87.25–88.0% range to an 88% point estimate; (3) 2026 FFO guidance of $6.96/share midpoint was disclosed in the Q4 deck but was not yet public at Investor Day; (4) CBD share of NOI rose slightly from 90.5% to 90.8%.

**Why partial:** Strategy slides are narrative and the comparison depends on retrieval surfacing the right pages from both documents. The cross-encoder reranker improves precision but doesn't guarantee both relevant slides are in the top-K.

---

### Q4 · What is the economic impact of Simon's shopping centers on local communities?

**Status:** ✅ Pass

**What the system does:**
The Simon report (November 2018) triggers the staleness warning — version_date is more than 2 years before today. Claude receives the source header marked "⚠ STALE SOURCE (published 2018, data may be outdated)" and prompt rule 5 requires flagging this before presenting any figures.

**Expected answer shape:**
> ⚠ The source for this answer is the Simon Property Group report published in November 2018 using 2017 data — approximately 7–8 years old. The figures below reflect conditions from that period and should not be treated as current. According to that report [Source 1], Simon's shopping centers supported X jobs and contributed $Y billion to local economies...

---

## Group B — Intra-document Inconsistency & Footnote Awareness

---

### Q5 · How many customers does Digital Realty have?

**Status:** ⚠️ Partial Pass

**What the system does:**
Default top_k is raised to 8, which increases the chance that both page 3 (5,500+ customers) and page 23 (5,000+ customers) of the March 2026 deck are retrieved simultaneously. Prompt rule 11 requires Claude to surface both values with page numbers and flag the inconsistency explicitly.

**Expected answer shape:**
> The March 2026 Digital Realty deck is internally inconsistent on this figure. Page 3 [Source 1] states "5,500+ customers" while page 23 [Source 2] states "5,000+ global customers" — both cite December 31, 2025 as the as-of date. The document itself does not reconcile this discrepancy. The December 2025 deck [Source 3, HISTORICAL VERSION] reported 5,000+.

**Why partial:** Both pages must rank in the top 8 for this to work. If page 23 ranks outside the retrieval window the inconsistency is missed.

---

### Q6 · What is BXP's dividend yield?

**Status:** ⚠️ Partial Pass

**What the system does:**
Two BXP documents with different stock prices at their respective dates produce two different yield calculations from the same reset dividend. Both are retrieved and presented with attribution.

**Expected answer shape:**
> BXP's dividend yield depends on the stock price at the time of each document. The Investor Day [Source 1, Aug 2025] calculated approximately 3.9% yield based on the post-reset $0.70 quarterly dividend divided by the August 2025 stock price. By the Q4 2025 Presentation [Source 2, Mar 2026] the same dividend implies approximately 5.2% yield because the stock price had moved lower. The dividend rate is the same in both cases — the yield difference reflects the price change.

---

### Q7 · What was BXP's actual market dividend yield as of August 29, 2025?

**Status:** ⚠️ Partial Pass

**What the system does:**
The [FOOTNOTE] tagger runs during ingestion and marks footnote lines in the bottom quarter of each slide. If footnote 2 on the Investor Day dividend slide was detected and tagged, Claude receives it with the [FOOTNOTE] prefix and prompt rule 10 instructs it to treat that as the authoritative override of the chart headline.

**Expected answer shape:**
> According to footnote 2 on the dividend slide of the BXP Investor Day presentation [Source 1, FOOTNOTE], the actual market-implied dividend yield as of August 29, 2025 was 5.47%. This is distinct from the 3.9% figure shown in the chart body, which represents a normalised construct based on the post-reset dividend divided by a reference price — not the market yield on that specific date.

**Why partial:** Depends on whether the footnote line matched the detection regex and was tagged during the re-ingestion. If the footnote text began with a pattern not covered by the regex it may not have been tagged.

---

### Q8 · How has Digital Realty's customer credit-quality profile changed?

**Status:** ❌ Acknowledged Gap

**What the system does:**
Retrieves chunks from both DLR decks. May surface the 50–51% investment-grade headline from both. If the footnote describing the measurement basis change (top-10 vs top-100 customers) was tagged during ingestion, Claude will present it. However, the system has no mechanism to automatically detect that the underlying measurement methodology changed — it can only surface what the footnotes say.

**Expected answer shape:**
> The investment-grade percentage of Digital Realty's customer base appears stable at approximately 50–51% across both decks. However, a footnote in the March 2026 deck [Source 1, FOOTNOTE] indicates this figure is now based on the top-100 customers representing $6.1B in revenue, compared to the December 2025 measurement which was based on the top-10 customers representing $4.5B in revenue. These two numbers are not directly comparable — the measurement scope expanded 10x between the two reports.

**Honest gap:** Automatic detection of methodology changes requires a structured fact-extraction layer that parses metric definitions, not just metric values. The system surfaces the footnotes but cannot proactively flag this without a human reading both carefully.

---

## Group C — Document-type and Scope Distinctions

---

### Q9 · In how many countries does Realty Income operate?

**Status:** ✅ Pass

**What the system does:**
The Realty Income Q4 2025 presentation body text contains both "10 countries" and "9 European countries" with scope qualifiers intact. Prompt rule 4 requires Claude to preserve scope qualifiers and never strip them to make numbers look comparable.

**Expected answer shape:**
> Realty Income operates in 10 countries total [Source 1] — the United States plus 9 European countries including the UK and 8 continental European markets. The figure of 9 countries refers specifically to the European footprint excluding the US. Both answers are correct under their respective scope definitions.

---

### Q10 · How big is EastGroup's portfolio?

**Status:** ⚠️ Partial Pass

**What the system does:**
The slide-aware chunker keeps each page intact so scope qualifiers remain in the chunk text. Top_k of 8 increases the chance that both page 3 (65M SF including development) and page 9 (65M SF under ownership) are retrieved. Rule 4 requires preserving the qualifiers.

**Expected answer shape:**
> EastGroup's portfolio is approximately 65 million square feet under two different scope definitions used within the same document. Page 3 [Source 1] states "approximately 65 million square feet" inclusive of development projects and value-add acquisitions in lease-up and under construction. Page 9 [Source 2] states "65 million square feet" referring to properties under ownership only. The numbers look identical but the scope is different.

---

### Q11 · What is Public Storage's NOI margin?

**Status:** ✅ Pass

**What the system does:**
Both PSA documents are labelled with specific types — "(type: Company Update)" and "(type: Merger Presentation)". Rule 12 instructs Claude that these are concurrent documents covering different scopes, not replacements of each other. All three margin figures are sourced and attributed separately.

**Expected answer shape:**
> Three NOI margin figures exist across the two concurrent PSA documents. PSA standalone margin: 78% [Source 1, Company Update, March 2026]. NSA standalone margin: 69% [Source 2, Merger Presentation]. Pro-forma combined margin post-merger: 77% [Source 2, Merger Presentation]. Presenting 77% as "PSA's margin" would be incorrect — that is the projected post-merger combined entity figure, not PSA on a standalone basis.

---

### Q12 · What is PSA's outlook for 2026?

**Status:** ✅ Pass

**What the system does:**
Two concurrent documents provide additive (not conflicting) outlooks. Coverage guarantee ensures both appear. Document type labels help Claude attribute correctly.

**Expected answer shape:**
> PSA's 2026 outlook has two additive components. Organic standalone guidance from the Company Update [Source 1]: revenue -2.2% to 0.0%, same-store NOI -3.9% to -0.5%. NSA acquisition impact from the Merger Presentation [Source 2]: FFO-neutral in 2026 (integration costs offset by new income), then $0.10–0.20 accretive in 2027, growing to $0.35–0.50 at stabilisation. Both outlooks are simultaneously true — the organic performance and the deal impact are separate line items.

---

### Q13 · What percentage of BXP's portfolio is in CBD markets?

**Status:** ⚠️ Partial Pass

**What the system does:**
Both BXP documents are retrieved with specific type labels. Multiple metrics exist across pages — NOI percentage, ARO percentage, and different as-of dates. Rule 4 preserves the qualifiers on each figure.

**Expected answer shape:**
> BXP's CBD concentration is reported under three different metrics across the two presentations. 90.5% of NOI as of Q2 2025 [Source 1, Investor Day]. Approximately 90% of Annualized Rental Obligations [Source 2, Q4 Update, page 7]. 90.8% of NOI as of Q4 2025 [Source 2, Q4 Update, page 9]. The NOI figure increased from 90.5% to 90.8% between the Investor Day and Q4 2025 — reflecting leasing activity in the second half of 2025.

---

### Q14 · What is the occupancy difference between PSA and NSA same-store?

**Status:** ✅ Pass

**What the system does:**
The 3-column comparison table on page 4 of the PSA Merger Presentation is extracted by pdfplumber as a structured financial_table chunk. All four figures are in one chunk.

**Expected answer shape:**
> From the comparison table on page 4 of the PSA Merger Presentation [Source 1]: PSA same-store occupancy: 92.0%. NSA same-store occupancy: 84.3%. Gap: 7.7 percentage points. Pro-forma combined occupancy: 90.3%. The occupancy gap reflects NSA's weaker asset base, which is part of the rationale for the merger — PSA's operating platform is expected to improve NSA's metrics over time.

---

## Group D — Chart, Table, and Map Parsing

---

### Q15 · What are self-storage REIT same-store NOI growth and margin figures?

**Status:** ⚠️ Partial Pass

**What the system does:**
The bar chart grid on PSA page 7 triggers vision extraction (sparse page). Claude vision reads the chart and returns labelled data points. However, the bar chart re-sorts companies by performance in each subchart, so the legend order does not map directly to bar order.

**Expected answer shape:**
> From the comparative bar chart on page 7 of the PSA Company Update [Source 1, AI VISION EXTRACTION]: Same-store NOI growth — PSA: [value], CubeSmart: [value], Extra Space: [value], NSA: [value]. Same-store NOI margin — PSA: [value], CubeSmart: [value], Extra Space: [value], NSA: [value]. Note: the bar chart re-sorts companies by ranking within each metric, so the mapping of bar positions to company names is based on vision extraction and should be verified against the source document.

---

### Q16 · Who is VICI's largest tenant by rent percentage?

**Status:** ✅ Pass

**What the system does:**
The tenant table on the VICI deck uses logo images for company names. Vision extraction reads the table and extracts both the logo names and their associated rent percentages. Company names also appear in body text on pages 5, 13, and 14 as confirmation.

**Expected answer shape:**
> Caesars Entertainment is VICI's largest tenant, representing approximately 39% of annualised rent [Source 1, AI VISION EXTRACTION — tenant table]. MGM Resorts is the second largest at approximately 34% [Source 1]. Together these two tenants account for approximately 73% of VICI's total rent, making tenant concentration a key consideration for the portfolio.

---

### Q17 · What are VICI's 10 Las Vegas Strip trophy assets?

**Status:** ⚠️ Partial Pass

**What the system does:**
Vision extraction reads the Las Vegas map on page 14. Body text on pages 13 and 16 names several assets explicitly. Combined, the system surfaces 8–10 asset names.

**Expected answer shape:**
> Based on body text and vision extraction from the Las Vegas Strip map [Sources 1–2]: Caesars Palace, The Venetian Resort, MGM Grand, Mandalay Bay, Park MGM, New York-New York, Excalibur, Luxor. The map on page 14 [AI VISION EXTRACTION] may contain additional asset labels — if any names read as unclear in the vision output, they should be verified against the source document directly.

---

### Q18 · What share of Realty Income's ABR comes from each US region vs Europe?

**Status:** ⚠️ Partial Pass

**What the system does:**
The geographic map on page 15 triggers vision extraction. Vision reads regional labels and their associated ABR values and percentages. Spatial adjacency is preserved by vision better than raw text extraction, but some region-to-value pairings may be uncertain for smaller regions.

**Expected answer shape:**
> From the geographic breakdown on page 15 of the Realty Income Q4 2025 presentation [Source 1, AI VISION EXTRACTION]: US regions — Pacific Northwest: ~1.6%, Midwest: ~20.3%, Northeast: ~9.2% [remaining regions listed]. Europe total: [percentage]. Note: values are extracted from a geographic map image — figures for regions with smaller text labels should be verified against the source document.

---

### Q19 · Which markets does EastGroup operate in, ranked by importance?

**Status:** ✅ Pass

**What the system does:**
Page 8 of the EGP roadshow contains a state-level ABR breakdown in text format which is fully extractable. Page 6 contains a dot-size map which vision extraction reads for city-level data.

**Expected answer shape:**
> By state ABR from page 8 [Source 1]: Texas 35%, Florida 25%, California 15%, Arizona 8%, North Carolina 5%, Other 12%. City-level distribution from the map on page 6 [Source 2, AI VISION EXTRACTION]: major concentrations in Dallas-Fort Worth, Houston, Tampa, Orlando, Los Angeles, Phoenix, and Charlotte, with dot size representing relative ABR contribution.

---

## Group E — Ingestion Quality

---

### Q20 · What is EastGroup's strategy for property selection?

**Status:** ✅ Pass

**What the system does:**
The is_likely_visual threshold is raised to 120 characters so photo pages with short captions are flagged for vision processing. Strategy content pages (pages 3, 6, 9, 11) have sufficient text and are chunked normally by the slide-aware chunker. Visual property photo pages are supplemented by vision extraction.

**Expected answer shape:**
> EastGroup focuses on Sunbelt industrial markets with an emphasis on infill locations in high-growth cities near major highways and population centres [Source 1, page 3]. The selection criteria prioritise functional Class A product in markets with strong demand fundamentals — primarily Texas, Florida, California, Arizona, and the Carolinas [Source 2, page 9]. The strategy avoids speculative development in markets with oversupply and targets assets with proximity to the end customer to support last-mile logistics.

---

### Q21 · What is BXP's key strategy?

**Status:** ✅ Pass

**What the system does:**
The slide-aware chunker keeps the BXP Investor Day summary slide intact as a single chunk. The two-column format is preserved in the text extraction.

**Expected answer shape:**
> From the strategy summary slide of the BXP Investor Day [Source 1]: Current strengths — CBD-focused portfolio, life science and technology tenant base, strong balance sheet, premier asset quality, and experienced management team. Action plan — active leasing to close vacancy gap, capital recycling from non-core dispositions, selective development, dividend reset to align payout with cash flow, balance sheet optimisation, and ESG leadership. The Q4 2025 presentation [Source 2] confirms this framework is unchanged with the dividend reset now completed.

---

## Group F — Cross-Document Synthesis

---

### Q22 · How is AI affecting demand across different real estate sectors?

**Status:** ✅ Pass

**What the system does:**
Per-document quota caps Digital Realty at 3 chunks so it cannot fill all top-K slots. The company coverage guarantee runs secondary retrievals for any company named or implied in the query. Result: all companies are represented.

**Expected answer shape:**
> AI is affecting each sector differently. Digital Realty [Source 1]: AI is the primary demand driver — data centre power demand is growing 2.7x, directly fuelling new construction and lease-up of hyperscale campuses [CURRENT VERSION]. BXP [Source 2]: AI indirectly benefits CBD office markets by concentrating high-paying technology and life science jobs in gateway cities, supporting premium office demand. Public Storage [Source 3]: uses AI as an internal efficiency tool (pricing, yield management) — it is not a demand driver for self-storage. VICI, Realty Income, EastGroup, Simon: AI is not a material theme in their submitted documents.

---

### Q23 · How are VICI's and Realty Income's gaming portfolios different?

**Status:** ✅ Pass

**What the system does:**
Coverage guarantee ensures both VICI and Realty Income chunks appear even if one ranked lower. Both documents are retrieved and compared directly.

**Expected answer shape:**
> The two portfolios have fundamentally different exposures to gaming. VICI [Source 1] is entirely concentrated in gaming and experiential real estate — 54 gaming properties and 39 experiential assets, with Caesars at ~39% of rent and MGM at ~34%. Gaming is the entire investment thesis. Realty Income [Source 2] holds gaming assets as one diversification segment among many alongside retail, industrial, data centres, and credit investments. Gaming is not a core concentration and represents a minority of total ABR. Comparing the two as "gaming REITs" would overstate Realty Income's gaming exposure.

---

### Q24 · What is the 2026 FFO outlook for each REIT in the corpus?

**Status:** ✅ Pass

**What the system does:**
Prompt rule 1 prohibits gap-filling from training data. Prompt rule 8 requires addressing each company separately and naming those absent from sources. Coverage guarantee ensures each company has at least one chunk retrieved.

**Expected answer shape:**
> 2026 FFO guidance by company, based solely on submitted documents: **BXP** [Source 1]: $6.96/share midpoint 2026 FFO guidance (range disclosed in Q4 2025 presentation). **PSA** [Source 2]: Organic revenue guidance -2.2% to 0.0%, same-store NOI -3.9% to -0.5%; NSA acquisition impact is FFO-neutral in 2026 then $0.10–0.20 accretive in 2027. **Digital Realty, VICI, Realty Income, EastGroup, Simon**: 2026 FFO per share guidance is not disclosed in their submitted documents. These figures may exist in earnings calls or SEC filings not included in this corpus — the system will not estimate them from general knowledge.

---

## Summary

| Group | Q | Status |
|---|---|---|
| A — Version & Temporal | Q1 | ⚠️ Partial |
| | Q2 | ✅ Pass |
| | Q3 | ⚠️ Partial |
| | Q4 | ✅ Pass |
| B — Footnote & Inconsistency | Q5 | ⚠️ Partial |
| | Q6 | ⚠️ Partial |
| | Q7 | ⚠️ Partial |
| | Q8 | ❌ Gap |
| C — Document-type Scope | Q9 | ✅ Pass |
| | Q10 | ⚠️ Partial |
| | Q11 | ✅ Pass |
| | Q12 | ✅ Pass |
| | Q13 | ⚠️ Partial |
| | Q14 | ✅ Pass |
| D — Vision & Charts | Q15 | ⚠️ Partial |
| | Q16 | ✅ Pass |
| | Q17 | ⚠️ Partial |
| | Q18 | ⚠️ Partial |
| | Q19 | ✅ Pass |
| E — Ingestion Quality | Q20 | ✅ Pass |
| | Q21 | ✅ Pass |
| F — Cross-Company | Q22 | ✅ Pass |
| | Q23 | ✅ Pass |
| | Q24 | ✅ Pass |

**✅ Pass: 13 · ⚠️ Partial: 10 · ❌ Gap: 1**
