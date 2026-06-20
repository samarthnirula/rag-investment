"""Secondary public-reference people context for the Epstein demo corpus.

This is not primary evidence. It exists so broad/person-identification
questions have a safe, sourced orientation layer when retrieval misses a name.
"""
from __future__ import annotations

from textwrap import dedent


EPSTEIN_PEOPLE_CONTEXT_TEXT = dedent(
    """
    Public-reference people index for the Epstein matter.
    Treat this as secondary context only. A person being named, mentioned, photographed,
    subpoenaed, associated with Epstein, or appearing in released files does not by itself
    mean the person committed a crime or participated in Epstein's abuse. Use primary
    documents, filings, transcripts, and exhibits for any legal conclusion.

    Sources used for this orientation index:
    - Associated Press, "From Elon Musk to the former Prince Andrew, a who's who of powerful people named in Epstein files," Feb. 1, 2026, https://apnews.com/article/682447e50bf9a3643a36c9b54ccdfa22
    - Associated Press, "Unsealing of documents related to decades of Jeffrey Epstein's sexual abuse of girls concludes," Jan. 10, 2024, https://apnews.com/article/58a3b2c4a7690d8956890ca08fc07723
    - Associated Press, "Justice Department releases largest batch yet of Epstein documents, says it totals 3 million pages," Jan. 30, 2026, https://apnews.com/article/ed743598c320b94bd9d91631618678d9
    - U.S. DOJ Office of Professional Responsibility, Executive Summary on the Acosta/Epstein NPA review, Nov. 2020.

    Key people and why they may appear in questions:
    - Jeffrey Epstein: financier and convicted sex offender at the center of the criminal, civil, and investigative record. He died in federal custody in 2019 while awaiting trial on federal sex-trafficking charges.
    - Ghislaine Maxwell: Epstein associate convicted in federal court in 2021 for sex-trafficking-related offenses and sentenced to 20 years in prison.
    - Virginia Giuffre: public survivor-witness and plaintiff whose litigation against Maxwell led to major document-unsealing disputes. AP reports she accused Epstein of sexual abuse and alleged she was directed to have encounters with powerful men; several named men denied her allegations.
    - Bill Clinton: former U.S. president. AP reports he spent time with Epstein more than two decades ago, including occasional flights on Epstein's private jet and visits/contacts connected to the White House. Clinton's representatives have said he denied knowledge of Epstein's wrongdoing, broke off relations after the 2006 charges, and no Epstein victim has publicly accused Clinton of involvement in Epstein's crimes.
    - Donald Trump: public figure and U.S. president whose name appears in Epstein-related public reporting/files. AP reports Trump had prior social association with Epstein and denied wrongdoing or involvement in Epstein's abuse.
    - Andrew Mountbatten-Windsor / Prince Andrew: former British royal figure long associated in public reporting with Epstein and Maxwell. Giuffre alleged Epstein trafficked her to Andrew when she was 17; Andrew denied wrongdoing and settled Giuffre's civil suit in 2022 without admission of liability.
    - Alan Dershowitz: lawyer and law professor who represented Epstein. Giuffre accused him and later withdrew her claims in 2022, saying she may have made a mistake; Dershowitz denied the allegations.
    - Jean-Luc Brunel: French modeling scout and Epstein associate. AP reports he was named in Giuffre-related allegations, denied wrongdoing, and died by suicide in a Paris jail in 2022 while awaiting trial on rape charges.
    - George Mitchell: former U.S. senator named in Giuffre's 2014 legal filing according to AP; he denied her allegation.
    - Bill Richardson: former New Mexico governor named in Giuffre's 2014 legal filing according to AP; he denied her allegation.
    - Glenn Dubin: billionaire financier named in Giuffre's 2014 legal filing according to AP; he denied her allegation.
    - Les Wexner: billionaire retail executive and former Epstein client/associate who appears in public reporting and investigative context; verify any specific claim against primary documents.
    - Lesley Groff: Epstein's longtime assistant. She appears in investigative and public-reporting context concerning scheduling, records, and possible witness/co-conspirator issues; verify specific allegations against filings or investigative records.
    - Sarah Kellen: Epstein assistant and alleged recruiter/scheduler in public reporting and litigation context; verify specific claims against primary filings.
    - Nadia Marcinkova: Epstein associate named in public reporting and in discussions of alleged assistants/co-conspirators; verify specific claims against primary filings.
    - Darren Indyke: attorney and Epstein estate co-executor who appears in estate, civil-settlement, and records-release context.
    - Richard Kahn: accountant and Epstein estate co-executor who appears in estate, civil-settlement, and records-release context.
    - Alexander Acosta: former U.S. Attorney for the Southern District of Florida who approved the 2007-2008 non-prosecution agreement; DOJ OPR later criticized aspects of the resolution as poor judgment but did not find professional misconduct.
    - Elon Musk: technology executive named in AP coverage of the 2026 file release. AP reports released materials showed Epstein-related contacts/invitations, while Musk said he refused an island invitation and denied wrongdoing.
    - Bill Gates: Microsoft co-founder and philanthropist who has publicly expressed regret over meeting Epstein; use current primary testimony or reputable reporting for any detailed claim.
    - Larry Summers: former U.S. Treasury Secretary and former Harvard president identified by AP as a longtime acquaintance in the released-files context; verify specific communications against primary records.
    - Ehud Barak: former Israeli prime minister identified by AP as appearing in released-file correspondence/visit logistics; Barak has said he did not observe inappropriate behavior.
    - Richard Branson: Virgin Group founder identified in AP coverage as appearing in released-file correspondence; verify details against primary files.
    - Steven Tisch: New York Giants co-owner identified by AP as mentioned in the 2026 released files; verify details against primary files.
    - Howard Lutnick: finance executive and public official/business figure reported in released-file context; verify specific meetings or island-visit claims against primary files.
    - Steve Bannon: political strategist reported in released-file context as having communications with Epstein; verify specific claims against primary records.
    """
).strip()


def epstein_people_context_text() -> str:
    return EPSTEIN_PEOPLE_CONTEXT_TEXT
